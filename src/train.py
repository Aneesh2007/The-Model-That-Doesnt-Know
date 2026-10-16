"""Train a CIFAR-10 classifier and calibrate OOD rejection thresholds."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm.auto import tqdm

from src.data import make_loaders
from src.model import build_model
from src.scoring import fit_thresholds


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def collect_logits(model, loader, device):
    model.eval()
    all_logits, all_labels = [], []
    with torch.inference_mode():
        for images, labels in loader:
            images = images.to(device)
            all_logits.append(model(images).cpu())
            all_labels.append(labels.cpu())
    return torch.cat(all_logits), torch.cat(all_labels)


def run_epoch(model, loader, criterion, device, optimizer=None, description=""):
    training = optimizer is not None
    model.train(training)
    total_loss, total_correct, total_count = 0.0, 0, 0
    context = torch.enable_grad() if training else torch.inference_mode()
    with context:
        for images, labels in tqdm(loader, desc=description, leave=False):
            images, labels = images.to(device), labels.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            if training:
                loss.backward()
                optimizer.step()
            count = labels.size(0)
            total_loss += float(loss.item()) * count
            total_correct += int(logits.argmax(1).eq(labels).sum().item())
            total_count += count
    return total_loss / max(total_count, 1), total_correct / max(total_count, 1)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="artifacts")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-fraction", type=float, default=0.1,
                        help="Fraction reserved for epoch/model selection (default 0.1).")
    parser.add_argument("--calibration-fraction", type=float, default=0.1,
                        help="Fraction reserved for temperature/threshold fitting (default 0.1).")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--known-rejection", type=float, default=0.05,
                        help="Target fraction of known validation samples rejected (default 0.05).")
    parser.add_argument("--no-pretrained", action="store_true",
                        help="Train from random initialization; substantially more epochs may be needed.")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.epochs < 1:
        raise ValueError("--epochs must be >= 1")
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Device: {device}")
    print("Known classes: CIFAR-10 (10 classes)")
    print("OOD evaluation datasets: SVHN digits and CIFAR-100 (kept out of training)")

    train_loader, val_loader, calibration_loader, class_names = make_loaders(
        root=args.data_dir,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_fraction=args.val_fraction,
        calibration_fraction=args.calibration_fraction,
        seed=args.seed,
    )
    model = build_model(num_classes=len(class_names), pretrained=not args.no_pretrained).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_acc = -1.0
    best_path = output_dir / "best_model.pt"
    history = []
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(
            model, train_loader, criterion, device, optimizer,
            description=f"Epoch {epoch}/{args.epochs} train",
        )
        val_loss, val_acc = run_epoch(
            model, val_loader, criterion, device,
            description=f"Epoch {epoch}/{args.epochs} val",
        )
        scheduler.step()
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "val_loss": val_loss,
            "val_accuracy": val_acc,
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        history.append(row)
        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train loss={train_loss:.4f} acc={train_acc:.4f} | "
            f"val loss={val_loss:.4f} acc={val_acc:.4f}"
        )
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "model_state_dict": model.state_dict(),
                "class_names": class_names,
                "image_size": args.image_size,
                "seed": args.seed,
                "epoch": epoch,
                "val_accuracy": val_acc,
            }, best_path)

    with (output_dir / "training_history.json").open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    # Calibrate temperature and thresholds on a separate known-class holdout.
    checkpoint = torch.load(best_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    calibration_logits, calibration_labels = collect_logits(model, calibration_loader, device)
    thresholds = fit_thresholds(
        calibration_logits, calibration_labels, target_known_rejection=args.known_rejection
    )
    checkpoint.update({
        "temperature": thresholds["temperature"],
        "confidence_threshold": thresholds["confidence_threshold"],
        "energy_threshold": thresholds["energy_threshold"],
        "target_known_rejection": thresholds["target_known_rejection"],
        "threshold_calibration": "separate known-class calibration split only",
        "seed": args.seed,
    })
    torch.save(checkpoint, best_path)
    with (output_dir / "thresholds.json").open("w", encoding="utf-8") as f:
        json.dump({k: float(v) for k, v in thresholds.items()}, f, indent=2)
    print(f"\nBest validation accuracy: {best_val_acc:.4f}")
    print(f"Saved checkpoint: {best_path}")
    print("Fitted thresholds (separate known-class calibration split):")
    print(json.dumps(thresholds, indent=2))
    print("Next: python -m src.evaluate --checkpoint artifacts/best_model.pt")


if __name__ == "__main__":
    main()
