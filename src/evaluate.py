"""Compare raw classification against confidence- and energy-based OOD rejection."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.data import make_eval_dataset
from src.model import load_checkpoint
from src.scoring import (
    detection_metrics,
    energy_score,
    expected_calibration_error,
    max_softmax_confidence,
)


def infer_dataset(model, dataset, device, batch_size=128, num_workers=2):
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=False,
    )
    logits_list, labels_list = [], []
    model.eval()
    with torch.inference_mode():
        for images, labels in tqdm(loader, desc="Inference", leave=False):
            logits_list.append(model(images.to(device)).cpu())
            labels_list.append(torch.as_tensor(labels).cpu())
    return torch.cat(logits_list), torch.cat(labels_list).long()


def safe_float(value):
    value = float(value)
    return value if np.isfinite(value) else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="artifacts/best_model.pt")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="artifacts/evaluation")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--ood-datasets", nargs="+", choices=["svhn", "cifar100"],
                        default=["svhn", "cifar100"])
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, checkpoint = load_checkpoint(args.checkpoint, device)
    image_size = int(checkpoint.get("image_size", 128))
    classes = list(checkpoint["class_names"])
    temperature = float(checkpoint.get("temperature", 1.0))
    conf_threshold = float(checkpoint.get("confidence_threshold", 0.5))
    energy_threshold = float(checkpoint.get("energy_threshold", 0.0))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Device: {device}; temperature={temperature:.4f}")
    print(f"Confidence threshold={conf_threshold:.4f}; energy threshold={energy_threshold:.4f}")

    datasets_to_eval = ["cifar10"] + args.ood_datasets
    all_rows, report = [], {"checkpoint": str(args.checkpoint), "temperature": temperature}
    score_cache = {}
    for name in datasets_to_eval:
        dataset = make_eval_dataset(name, args.data_dir, image_size)
        logits, raw_labels = infer_dataset(
            model, dataset, device, args.batch_size, args.num_workers
        )
        pred = logits.argmax(dim=1).numpy()
        conf = max_softmax_confidence(logits, temperature).numpy()
        energy = energy_score(logits, temperature).numpy()
        is_ood_dataset = name != "cifar10"
        rejected_conf = conf < conf_threshold
        rejected_energy = energy > energy_threshold

        score_cache[name] = {
            "logits": logits, "labels": raw_labels, "pred": pred,
            "confidence": conf, "energy": energy,
        }
        row = {
            "dataset": name,
            "n_samples": int(len(raw_labels)),
            "role": "OOD" if is_ood_dataset else "known test",
            "confidence_rejection_rate": float(rejected_conf.mean()),
            "energy_rejection_rate": float(rejected_energy.mean()),
            "mean_max_softmax_confidence": float(conf.mean()),
            "mean_energy": float(energy.mean()),
        }
        if not is_ood_dataset:
            labels = raw_labels.numpy()
            row.update({
                "raw_top1_accuracy": float(accuracy_score(labels, pred)),
                "confidence_accepted_accuracy": safe_float(
                    accuracy_score(labels[~rejected_conf], pred[~rejected_conf])
                ) if (~rejected_conf).any() else None,
                "energy_accepted_accuracy": safe_float(
                    accuracy_score(labels[~rejected_energy], pred[~rejected_energy])
                ) if (~rejected_energy).any() else None,
                "ece_before_temperature": expected_calibration_error(logits, raw_labels),
                "ece_after_temperature": expected_calibration_error(
                    logits / temperature, raw_labels
                ),
                "confidence_known_false_rejection_rate": float(rejected_conf.mean()),
                "energy_known_false_rejection_rate": float(rejected_energy.mean()),
            })
            report["known_classification_report"] = classification_report(
                labels, pred, labels=list(range(len(classes))), target_names=classes,
                output_dict=True, zero_division=0,
            )
            cm = confusion_matrix(labels, pred, labels=list(range(len(classes))))
            pd.DataFrame(cm, index=classes, columns=classes).to_csv(
                output_dir / "known_confusion_matrix.csv"
            )
        else:
            # Raw OOD labels are from a different label space and must not be used
            # as CIFAR-10 classification targets.
            row["raw_classifier_forced_known_label_rate"] = 1.0
            row["raw_ood_false_accept_rate_without_rejection"] = 1.0
            row["ood_recall_confidence_threshold"] = float(rejected_conf.mean())
            row["ood_recall_energy_threshold"] = float(rejected_energy.mean())

        report.setdefault("datasets", []).append(row)
        print(json.dumps(row, indent=2))

        # Save per-example audit rows so false accepts and false rejects can be inspected.
        labels_np = raw_labels.numpy()
        if name == "cifar10":
            true_labels = [classes[int(i)] for i in labels_np]
            correct = (pred == labels_np).tolist()
        else:
            if name == "cifar100":
                ood_class_names = dataset.classes
                true_labels = [f"cifar100:{ood_class_names[int(i)]}" for i in labels_np]
            else:
                true_labels = [f"svhn:digit_{int(i)}" for i in labels_np]
            correct = [None] * len(labels_np)
        for i in range(len(labels_np)):
            all_rows.append({
                "dataset": name,
                "sample_index": i,
                "true_label": true_labels[i],
                "raw_prediction": classes[int(pred[i])],
                "raw_prediction_correct": correct[i],
                "max_softmax_confidence": float(conf[i]),
                "energy_score": float(energy[i]),
                "confidence_decision": "UNKNOWN" if rejected_conf[i] else classes[int(pred[i])],
                "energy_decision": "UNKNOWN" if rejected_energy[i] else classes[int(pred[i])],
                "confidence_rejected": bool(rejected_conf[i]),
                "energy_rejected": bool(rejected_energy[i]),
            })

    # Detection ranking metrics: 1 means OOD, and larger score means more OOD-like.
    known = score_cache["cifar10"]
    for method in ("confidence", "energy"):
        all_scores, all_targets = [], []
        if method == "confidence":
            all_scores.extend((-known["confidence"]).tolist())
        else:
            all_scores.extend(known["energy"].tolist())
        all_targets.extend([0] * len(known["confidence"]))
        for name in args.ood_datasets:
            item = score_cache[name]
            if method == "confidence":
                all_scores.extend((-item["confidence"]).tolist())
            else:
                all_scores.extend(item["energy"].tolist())
            all_targets.extend([1] * len(item["confidence"]))
        metrics = detection_metrics(np.asarray(all_targets), np.asarray(all_scores))
        report[f"combined_id_vs_ood_{method}"] = metrics
        print(f"Combined ID-vs-OOD {method}: {json.dumps(metrics)}")

    pd.DataFrame(report["datasets"]).to_csv(output_dir / "metrics.csv", index=False)
    pd.DataFrame(all_rows).to_csv(output_dir / "per_sample_predictions.csv", index=False)
    with (output_dir / "report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    # Confidence and energy distributions help show overlap and threshold failures.
    for field, threshold, filename, x_label in [
        ("confidence", conf_threshold, "confidence_distributions.png", "Max softmax confidence"),
        ("energy", energy_threshold, "energy_distributions.png", "Energy score (higher = more OOD-like)"),
    ]:
        plt.figure(figsize=(9, 5))
        for name, values in score_cache.items():
            plt.hist(values[field], bins=40, density=True, alpha=0.4, label=name)
        plt.axvline(threshold, linestyle="--", linewidth=2, label="known-validation threshold")
        plt.xlabel(x_label)
        plt.ylabel("Density")
        plt.title(f"{x_label}: known vs excluded categories")
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / filename, dpi=160)
        plt.close()

    print(f"\nSaved evaluation outputs to: {output_dir.resolve()}")
    print("Inspect per_sample_predictions.csv for high-confidence OOD false accepts and known false rejects.")


if __name__ == "__main__":
    main()
