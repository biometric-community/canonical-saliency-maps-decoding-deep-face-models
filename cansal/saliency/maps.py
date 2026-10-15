"""Occlusion CIS/CMS + Grad-CAM baselines."""

from __future__ import annotations

from typing import Dict, Tuple

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
            drop = max(0.0, base - conf)
            acc[y0:y1, x0:x1] += drop
            counts[y0:y1, x0:x1] += 1.0
    dens = acc / (counts + 1e-8)
    if dens.max() > 0:
        dens = dens / dens.max()
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


def negative_explanation(image: torch.Tensor, heatmap: np.ndarray) -> torch.Tensor:
    """E = (1-H)*I with H normalized to equal pixel sum across methods."""
    h = torch.from_numpy(heatmap).to(image.device, dtype=image.dtype)
    h = h / (h.sum() + 1e-8) * heatmap.size  # equal sum mass
    h = (h - h.min()) / (h.max() - h.min() + 1e-8)
    return image * (1.0 - h.view(1, 1, *h.shape))


@torch.no_grad()
def explanation_metrics(
    model,
    image: torch.Tensor,
    class_idx: int,
    heatmap: np.ndarray,
) -> Dict[str, float]:
    """Average Drop % and Increase in Confidence (paper Eqs. 11–12)."""
    model.eval()
    base = float(model.predict_proba(image)[0, class_idx].item())
    e = negative_explanation(image, heatmap)
    conf_e = float(model.predict_proba(e)[0, class_idx].item())
    drop = max(0.0, (base - conf_e) / (base + 1e-8)) * 100.0
    increase = 100.0 if conf_e > base else 0.0
    return {"avg_drop_pct": drop, "increase_conf": increase, "base": base, "expl": conf_e}
