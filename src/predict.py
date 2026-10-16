"""Run single-image inference with optional reject-as-unknown behavior."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from PIL import Image, UnidentifiedImageError

from src.data import make_transforms
from src.model import load_checkpoint
from src.scoring import energy_score, max_softmax_confidence


def predict_image(image_path: str | Path, checkpoint_path: str | Path, method: str = "confidence"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, checkpoint = load_checkpoint(checkpoint_path, device)
    _, eval_transform = make_transforms(int(checkpoint.get("image_size", 128)))
    try:
        with Image.open(image_path) as im:
            image = eval_transform(im.convert("RGB")).unsqueeze(0).to(device)
    except (FileNotFoundError, UnidentifiedImageError) as exc:
        raise ValueError(f"Cannot open image: {image_path}") from exc

    with torch.inference_mode():
        logits = model(image)
        probs = torch.softmax(logits / float(checkpoint.get("temperature", 1.0)), dim=1)[0]
        confidence, class_index = probs.max(dim=0)
        energy = energy_score(logits, float(checkpoint.get("temperature", 1.0)))[0]

    names = checkpoint["class_names"]
    raw_label = names[int(class_index.item())]
    conf = float(confidence.item())
    en = float(energy.item())
    method = method.lower()
    if method == "confidence":
        threshold = float(checkpoint.get("confidence_threshold", 0.5))
        is_unknown = conf < threshold
        score_name, score, comparator = "max_softmax_confidence", conf, f">= {threshold:.6f} to accept"
    elif method == "energy":
        threshold = float(checkpoint.get("energy_threshold", 0.0))
        is_unknown = en > threshold
        score_name, score, comparator = "energy", en, f"<= {threshold:.6f} to accept"
    else:
        raise ValueError("method must be 'confidence' or 'energy'")

    return {
        "image": str(image_path),
        "decision": "UNKNOWN / NEEDS REVIEW" if is_unknown else raw_label,
        "raw_top_prediction": raw_label,
        "max_softmax_confidence": round(conf, 6),
        "energy_score": round(en, 6),
        "method": method,
        "decision_score": score_name,
        "decision_score_value": round(score, 6),
        "decision_rule": comparator,
        "temperature": float(checkpoint.get("temperature", 1.0)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, help="Path to an image file")
    parser.add_argument("--checkpoint", default="artifacts/best_model.pt")
    parser.add_argument("--method", choices=["confidence", "energy"], default="confidence")
    args = parser.parse_args()
    print(json.dumps(predict_image(args.image, args.checkpoint, args.method), indent=2))


if __name__ == "__main__":
    main()
