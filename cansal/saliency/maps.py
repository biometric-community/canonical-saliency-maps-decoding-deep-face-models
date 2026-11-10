"""Occlusion CIS/CMS + Grad-CAM baselines."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F


@torch.no_grad()
def occlusion_cis(
    model,
    image: torch.Tensor,
    class_idx: int,
    patch: int = 15,
    stride: int = 8,
) -> np.ndarray:
    """
    Canonical Image Saliency via occlusion (paper Eq.1 confidence drop).
    image: (1,3,H,W). CelebA is treated as approximate canonical frame (no PRNet).
    """
    model.eval()
    _, _, h, w = image.shape
    base = float(model.predict_proba(image)[0, class_idx].item())
    acc = np.zeros((h, w), dtype=np.float64)
    counts = np.zeros((h, w), dtype=np.float64)
    half = patch // 2
    for y in range(0, h, stride):
        for x in range(0, w, stride):
            y0, y1 = max(0, y - half), min(h, y + half + 1)
            x0, x1 = max(0, x - half), min(w, x + half + 1)
            occ = image.clone()
            occ[:, :, y0:y1, x0:x1] = 0.0
            conf = float(model.predict_proba(occ)[0, class_idx].item())
            drop = base - conf  # Eq. 1: plain difference, increases stay negative
            acc[y0:y1, x0:x1] += drop
            counts[y0:y1, x0:x1] += 1.0
    dens = acc / (counts + 1e-8)  # Eq. 5: density normalisation
    peak = np.abs(dens).max()
    if peak > 0:
        dens = dens / peak
    return dens.astype(np.float32)


def grad_cam(model, image: torch.Tensor, class_idx: int) -> np.ndarray:
    """Grad-CAM on last conv features."""
    model.eval()
    feats = {}
    grads = {}

    def fwd_hook(_m, _i, o):
        feats["v"] = o

    def bwd_hook(_m, _gi, go):
        grads["v"] = go[0]

    h1 = model.features[-1].register_forward_hook(fwd_hook)
    h2 = model.features[-1].register_full_backward_hook(bwd_hook)
    image = image.detach().requires_grad_(True)
    logits, _ = model(image)
    score = logits[0, class_idx]
    model.zero_grad(set_to_none=True)
    score.backward()
    h1.remove()
    h2.remove()
    f = feats["v"]  # (1,C,h,w)
    g = grads["v"]
    weights = g.mean(dim=(2, 3), keepdim=True)
    cam = F.relu((weights * f).sum(dim=1, keepdim=True))
    cam = F.interpolate(cam, size=image.shape[-2:], mode="bilinear", align_corners=False)
    cam = cam[0, 0].detach().cpu().numpy()
    if cam.max() > 0:
        cam = cam / cam.max()
    return cam.astype(np.float32)


def aggregate_cms(cis_list) -> np.ndarray:
    """Canonical Model Saliency = mean CIS."""
    stack = np.stack(cis_list, axis=0)
    m = stack.mean(axis=0)
    if m.max() > 0:
        m = m / m.max()
    return m.astype(np.float32)


def normalised_heatmap(heatmap: np.ndarray) -> np.ndarray:
    """Eq. 10, first step: H' = (h - min h) / (max h - min h)."""
    h = np.asarray(heatmap, dtype=np.float64)
    return (h - h.min()) / (h.max() - h.min() + 1e-8)


def shared_mass(*heatmaps: np.ndarray) -> float:
    """The scalar s of Eq. 10 for one image: the smallest pixel sum among its normalised
    heatmaps, so every standardised map stays within [0, 1]."""
    return float(min(normalised_heatmap(h).sum() for h in heatmaps))


def negative_explanation(image: torch.Tensor, heatmap: np.ndarray, mass: Optional[float] = None) -> torch.Tensor:
    """Eq. 9: E = (1 - H) * I, with H = s * H' / sum(H') (Eq. 10) so that all methods for the
    same image darken the same total amount. Without `mass`, H = H'."""
    h = normalised_heatmap(heatmap)
    if mass is not None:
        h = h * (mass / (h.sum() + 1e-8))
    h = torch.from_numpy(h).to(image.device, dtype=image.dtype)
    return image * (1.0 - h.view(1, 1, *h.shape))


@torch.no_grad()
def explanation_metrics(
    model,
    image: torch.Tensor,
    class_idx: int,
    heatmap: np.ndarray,
    mass: Optional[float] = None,
) -> Dict[str, float]:
    """Average Drop % and Increase in Confidence (paper Eqs. 11–12)."""
    model.eval()
    base = float(model.predict_proba(image)[0, class_idx].item())
    e = negative_explanation(image, heatmap, mass)
    conf_e = float(model.predict_proba(e)[0, class_idx].item())
    drop = max(0.0, (base - conf_e) / (base + 1e-8)) * 100.0
    increase = 100.0 if conf_e > base else 0.0
    return {"avg_drop_pct": drop, "increase_conf": increase, "base": base, "expl": conf_e}
