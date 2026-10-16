import torch

from src.scoring import energy_score, max_softmax_confidence


def test_confidence_is_between_zero_and_one():
    logits = torch.tensor([[4.0, 1.0, -2.0], [0.0, 0.0, 0.0]])
    confidence = max_softmax_confidence(logits)
    assert torch.all(confidence >= 0)
    assert torch.all(confidence <= 1)
    assert confidence[0] > confidence[1]


def test_energy_score_shape_and_finite_values():
    logits = torch.tensor([[2.0, 0.0], [0.0, 2.0]])
    energy = energy_score(logits)
    assert energy.shape == (2,)
    assert torch.isfinite(energy).all()
