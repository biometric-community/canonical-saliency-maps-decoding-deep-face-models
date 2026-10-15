"""CIS + Grad-CAM for one image."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image
from torchvision import transforms

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from cansal.models.gender_net import GenderNet
from cansal.saliency.maps import grad_cam, occlusion_cis


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--image", required=True)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    device_name = args.device or cfg.get("device", "cpu")
    device = torch.device(
        device_name if not str(device_name).startswith("cuda") or torch.cuda.is_available() else "cpu"
    )
    blob = torch.load(Path(cfg["paths"]["checkpoint_dir"]) / "last.pt", map_location=device, weights_only=False)
    model = GenderNet(pretrained=False).to(device)
    model.load_state_dict(blob["model"])
    model.eval()
    size = int(cfg["model"].get("image_size", 112))
    tf = transforms.Compose(
        [
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )
    x = tf(Image.open(args.image).convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        cls = int(model.predict_proba(x).argmax(1).item())
    patch = int(cfg["eval"].get("patch", 15))
    stride = int(cfg["eval"].get("stride", 8))
    cis = occlusion_cis(model, x, cls, patch=patch, stride=stride)
    gcam = grad_cam(model, x, cls)
    out_dir = Path(cfg["paths"]["pred_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "cis.npy", cis)
    np.save(out_dir / "gradcam.npy", gcam)
    meta = {"pred_class": cls, "cis_peak": float(cis.max()), "gradcam_peak": float(gcam.max())}
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"Wrote {out_dir} pred={cls}")


if __name__ == "__main__":
    main()
