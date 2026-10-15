"""Gender accuracy + CIS/CMS vs Grad-CAM explanation metrics."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from cansal.data.attrs import CelebAAttrDataset, resolve_celeba_root
from cansal.models.gender_net import GenderNet
from cansal.saliency.maps import aggregate_cms, explanation_metrics, grad_cam, occlusion_cis


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _device(cfg: dict, override: Optional[str] = None) -> torch.device:
    name = override or cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    if str(name).startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(name)


@torch.no_grad()
def eval_gender_acc(model, ds, device, batch_size, nw) -> Dict:
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=nw)
    correct, total = 0, 0
    model.eval()
    for images, _sid, male, _nd in tqdm(loader, desc="gender-acc", leave=False):
        images = images.to(device)
        labels = male.to(device).long()
        logits, _ = model(images)
        correct += int((logits.argmax(1) == labels).sum().item())
        total += labels.numel()
    acc = correct / max(total, 1)
    print(f"[gender] acc={acc:.4f} n={total}")
    return {"accuracy": acc, "n": int(total)}


def evaluate(cfg: dict, device_override: Optional[str] = None) -> Dict:
    device = _device(cfg, device_override)
    ckpt = Path(cfg["paths"]["checkpoint_dir"]) / "last.pt"
    blob = torch.load(ckpt, map_location=device, weights_only=False)
    model = GenderNet(pretrained=False).to(device)
    model.load_state_dict(blob["model"])
    model.eval()

    root = resolve_celeba_root(cfg)
    image_size = int(cfg["model"].get("image_size", 112))
    max_n = cfg["data"].get("max_images")
    max_n = int(max_n) if max_n is not None else None
    val_ds = CelebAAttrDataset(
        root,
        split="val",
        image_size=image_size,
        train_ratio=float(cfg["data"].get("train_ratio", 0.7)),
        max_n=max_n,
        seed=int(cfg["train"].get("seed", 42)),
        train=False,
    )
    bs = int(cfg["eval"].get("batch_size", 128))
    nw = int(cfg["train"].get("num_workers", 4))
    results: Dict = {"gender": eval_gender_acc(model, val_ds, device, bs, nw)}

    n_sal = int(cfg["eval"].get("n_saliency_images", 128))
    patch = int(cfg["eval"].get("patch", 15))
    stride = int(cfg["eval"].get("stride", 8))
    idxs = list(range(min(len(val_ds), n_sal)))
    sub = Subset(val_ds, idxs)

    cis_maps = []
    met_cis = {"avg_drop_pct": [], "increase_conf": []}
    met_gcam = {"avg_drop_pct": [], "increase_conf": []}
    wins = 0

    for i in tqdm(range(len(sub)), desc="saliency"):
        img, _sid, male, _nd = sub[i]
        x = img.unsqueeze(0).to(device)
        cls = int(male)
        with torch.no_grad():
            pred = int(model.predict_proba(x).argmax(1).item())
        # use predicted class for explanation (paper-style)
        cls = pred
        cis = occlusion_cis(model, x, cls, patch=patch, stride=stride)
        gcam = grad_cam(model, x, cls)
        cis_maps.append(cis)
        m_c = explanation_metrics(model, x, cls, cis)
        m_g = explanation_metrics(model, x, cls, gcam)
        for k in ("avg_drop_pct", "increase_conf"):
            met_cis[k].append(m_c[k])
            met_gcam[k].append(m_g[k])
        if m_c["avg_drop_pct"] > m_g["avg_drop_pct"]:
            wins += 1

    cms = aggregate_cms(cis_maps)
    sal = {
        "n_images": len(cis_maps),
        "patch": patch,
        "stride": stride,
        "cis": {
            "avg_drop_pct": float(np.mean(met_cis["avg_drop_pct"])),
            "increase_conf_pct": float(np.mean(met_cis["increase_conf"])),
        },
        "gradcam": {
            "avg_drop_pct": float(np.mean(met_gcam["avg_drop_pct"])),
            "increase_conf_pct": float(np.mean(met_gcam["increase_conf"])),
        },
        "win_pct_cis_vs_gradcam": float(100.0 * wins / max(len(cis_maps), 1)),
        "cms_peak_yx": [int(x) for x in np.unravel_index(int(cms.argmax()), cms.shape)],
        "cms_center_mass": float(
            cms[
                cms.shape[0] // 4 : 3 * cms.shape[0] // 4,
                cms.shape[1] // 4 : 3 * cms.shape[1] // 4,
            ].mean()
        ),
    }
    print(
        f"[sal] CIS drop={sal['cis']['avg_drop_pct']:.2f}% "
        f"GradCAM drop={sal['gradcam']['avg_drop_pct']:.2f}% "
        f"win={sal['win_pct_cis_vs_gradcam']:.1f}%"
    )
    results["saliency"] = sal

    # persist CMS for report figure
    log_dir = Path(cfg["paths"]["log_dir"])
    log_dir.mkdir(parents=True, exist_ok=True)
    np.save(log_dir / "cms.npy", cms)
    out = log_dir / "eval.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
        f.write("\n")
    print(f"Wrote {out}")
    return results


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    evaluate(load_config(args.config), device_override=args.device)


if __name__ == "__main__":
    main()
