"""Occlusion saliency, canonical model saliency and the faithfulness metrics (Secs. 3 and 5.1)."""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn

import cansal.saliency.maps as maps
from cansal.saliency.maps import aggregate_cms, explanation_metrics, negative_explanation, occlusion_cis


class RegionModel(nn.Module):
    """Class-1 confidence is a sigmoid of a weighted pixel sum, so the weights say what matters."""

    def __init__(self, weights: torch.Tensor):
        super().__init__()
        self.weights = weights

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        p = torch.sigmoid((x[:, 0] * self.weights).sum(dim=(1, 2)))
        return torch.stack([1 - p, p], dim=1)


class MeanModel(nn.Module):
    """Class-1 confidence is the mean pixel value."""

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        p = x.mean(dim=(1, 2, 3)).clamp(0, 1)
        return torch.stack([1 - p, p], dim=1)


class AnyOccludedModel(nn.Module):
    """Confidence drops by the same amount whichever patch is occluded."""

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        p = torch.sigmoid(5 * x.amin(dim=(1, 2, 3)))
        return torch.stack([1 - p, p], dim=1)


def image(size: int = 32) -> torch.Tensor:
    return torch.ones(1, 3, size, size)


def test_saliency_follows_the_region_the_model_uses():
    """Eq. (1): importance is the drop in confidence when a patch is occluded."""
    weights = torch.zeros(32, 32)
    weights[:8, :8] = 0.02
    cis = occlusion_cis(RegionModel(weights), image(), 1, patch=7, stride=4)
    assert cis[:8, :8].mean() > 5 * cis[16:, 16:].mean()


def test_confidence_increases_are_not_clipped():
    """Eq. (1) is a plain difference, so occluding counter-evidence gives negative saliency."""
    weights = torch.zeros(32, 32)
    weights[:8, :8] = 0.02
    weights[24:, 24:] = -0.02
    cis = occlusion_cis(RegionModel(weights), image(), 1, patch=7, stride=4)
    assert cis[26:, 26:].mean() < 0


def test_density_normalisation_divides_by_the_occlusion_count():
    """Eq. (5): a constant drop gives a flat map once each pixel's sum is divided by its count."""
    cis = occlusion_cis(AnyOccludedModel(), image(), 1, patch=7, stride=4)
    assert np.allclose(cis, cis.max(), rtol=1e-4)


def test_model_saliency_is_the_mean_of_image_saliency():
    """Eq. (6): the CMS map is the average CIS map."""
    rng = np.random.default_rng(0)
    cis = [rng.random((8, 8)).astype(np.float32) for _ in range(5)]
    cms = aggregate_cms(cis)
    mean = np.mean(cis, axis=0)
    assert np.allclose(cms / cms.max(), mean / mean.max(), atol=1e-6)


def test_heatmaps_of_one_image_darken_the_same_total_amount():
    """Eq. (10): heatmaps are standardised to the same sum, so a broad map gains no advantage."""
    spot = np.zeros((16, 16), dtype=np.float32)
    spot[6:10, 6:10] = 1.0
    broad = np.zeros((16, 16), dtype=np.float32)
    broad[2:14, 2:14] = 1.0
    img = image(16)
    darkened = [(img - negative_explanation(img, h, mass=10.0)).sum().item() for h in (spot, broad)]
    assert darkened[0] == pytest.approx(darkened[1], rel=1e-3)


def test_negative_explanation_darkens_the_highlighted_pixels():
    """Eq. (9): E = (1 - H) * I, so the most salient pixel is removed and the least is kept."""
    heat = np.zeros((16, 16), dtype=np.float32)
    heat[8, 8] = 1.0
    e = negative_explanation(image(16), heat)
    assert e[0, 0, 8, 8] < e[0, 0, 0, 0]
    assert e[0, 0, 0, 0] == pytest.approx(1.0)


@pytest.mark.parametrize(
    "explained,drop,increase",
    [(0.6, 25.0, 0.0), (0.9, 0.0, 100.0)],
)
def test_average_drop_and_increase_in_confidence(monkeypatch, explained, drop, increase):
    """Eqs. (11)-(12) for one image whose original confidence is 0.8."""
    monkeypatch.setattr(maps, "negative_explanation", lambda img, *_: torch.full_like(img, explained))
    result = explanation_metrics(MeanModel(), torch.full((1, 3, 4, 4), 0.8), 1, np.ones((4, 4)))
    assert result["avg_drop_pct"] == pytest.approx(drop, abs=1e-4)
    assert result["increase_conf"] == pytest.approx(increase)


def test_patch_size_matches_the_paper(default_cfg, targets):
    assert default_cfg["eval"]["patch"] == targets["patch"]
