"""Uncertainty scores, threshold calibration, and evaluation metrics."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score


def max_softmax_confidence(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """Top-class probability after optional temperature scaling; larger means more ID-like."""
    probs = F.softmax(logits / max(float(temperature), 1e-6), dim=1)
    return probs.max(dim=1).values


def energy_score(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """Energy score E(x)=-T logsumexp(logits/T); larger often means more OOD-like."""
    t = max(float(temperature), 1e-6)
    return -t * torch.logsumexp(logits / t, dim=1)


def fit_temperature(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Fit a single temperature on labelled known-class validation data only."""
    logits = logits.detach()
    labels = labels.detach().long()
    log_t = torch.zeros(1, device=logits.device, requires_grad=True)
    optimizer = torch.optim.LBFGS([log_t], lr=0.05, max_iter=50)

    def closure():
        optimizer.zero_grad()
        temperature = log_t.exp().clamp(0.05, 20.0)
        loss = F.cross_entropy(logits / temperature, labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(log_t.detach().exp().clamp(0.05, 20.0).item())


def fit_thresholds(
    calibration_logits: torch.Tensor,
    calibration_labels: torch.Tensor,
    target_known_rejection: float = 0.05,
) -> dict[str, float]:
    """Fit temperature and score thresholds from a known-class calibration split only.

    Thresholds are selected so approximately 95% of calibration-known examples are
    accepted. This avoids using the OOD test set to tune the rejection boundary.
    """
    temperature = fit_temperature(calibration_logits, calibration_labels)
    confidences = max_softmax_confidence(calibration_logits, temperature).cpu().numpy()
    energies = energy_score(calibration_logits, temperature).cpu().numpy()
    q = float(np.clip(target_known_rejection, 0.0, 0.5))
    return {
        "temperature": temperature,
        "confidence_threshold": float(np.quantile(confidences, q)),
        "energy_threshold": float(np.quantile(energies, 1.0 - q)),
        "target_known_rejection": q,
    }


def expected_calibration_error(logits: torch.Tensor, labels: torch.Tensor, bins: int = 15) -> float:
    """Compute top-label ECE: confidence/accuracy gap, weighted by bin size."""
    probs = F.softmax(logits, dim=1)
    confidence, prediction = probs.max(dim=1)
    correct = prediction.eq(labels)
    edges = torch.linspace(0.0, 1.0, bins + 1, device=logits.device)
    ece = torch.zeros((), device=logits.device)
    for i in range(bins):
        # Include the right edge in the last bin only.
        if i == bins - 1:
            mask = (confidence >= edges[i]) & (confidence <= edges[i + 1])
        else:
            mask = (confidence >= edges[i]) & (confidence < edges[i + 1])
        if mask.any():
            ece += mask.float().mean() * torch.abs(
                confidence[mask].mean() - correct[mask].float().mean()
            )
    return float(ece.item())


def detection_metrics(is_ood: np.ndarray, ood_score: np.ndarray) -> dict[str, float]:
    """Calculate AUROC/AP where label 1 means OOD and higher scores mean more OOD-like."""
    is_ood = np.asarray(is_ood, dtype=np.int32)
    ood_score = np.asarray(ood_score, dtype=np.float64)
    if len(np.unique(is_ood)) < 2:
        return {"ood_auroc": float("nan"), "ood_aupr": float("nan")}
    return {
        "ood_auroc": float(roc_auc_score(is_ood, ood_score)),
        "ood_aupr": float(average_precision_score(is_ood, ood_score)),
    }
