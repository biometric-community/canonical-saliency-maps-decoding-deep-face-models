"""Train CelebA gender classifier."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from cansal.data.attrs import CelebAAttrDataset, resolve_celeba_root
from cansal.models.gender_net import GenderNet


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _device(cfg: dict, override: Optional[str] = None) -> torch.device:
    name = override or cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    if str(name).startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(name)


def train(cfg: dict, device_override: Optional[str] = None) -> Dict:
    device = _device(cfg, device_override)
    root = resolve_celeba_root(cfg)
    seed = int(cfg["train"].get("seed", 42))
    torch.manual_seed(seed)
    np.random.seed(seed)

    max_n = cfg["data"].get("max_images")
    max_n = int(max_n) if max_n is not None else None
    image_size = int(cfg["model"].get("image_size", 112))

    train_ds = CelebAAttrDataset(
        root,
        split="train",
        image_size=image_size,
        train_ratio=float(cfg["data"].get("train_ratio", 0.7)),
        max_n=max_n,
        seed=seed,
        train=True,
    )
    loader = DataLoader(
        train_ds,
        batch_size=int(cfg["train"].get("batch_size", 64)),
        shuffle=True,
        num_workers=int(cfg["train"].get("num_workers", 4)),
        pin_memory=device.type == "cuda",
        drop_last=True,
    )
    model = GenderNet(pretrained=True).to(device)
    opt = torch.optim.SGD(
        model.parameters(), lr=float(cfg["train"].get("lr", 0.01)), momentum=0.9, weight_decay=5e-4
    )
    epochs = int(cfg["train"].get("epochs", 5))
    hist = []
    t0 = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        losses, correct, total = [], 0, 0
        pbar = tqdm(loader, desc=f"ep{epoch}/{epochs}", leave=False)
        for images, _sid, male, _nd in pbar:
            images = images.to(device)
            labels = male.to(device).long()
            logits, _ = model(images)
            loss = F.cross_entropy(logits, labels)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(float(loss.detach()))
            correct += int((logits.argmax(1) == labels).sum().item())
            total += labels.numel()
            pbar.set_postfix(loss=f"{losses[-1]:.3f}", acc=f"{correct/max(total,1):.3f}")
        mean = float(np.mean(losses))
        acc = correct / max(total, 1)
        hist.append({"epoch": epoch, "loss": mean, "acc": acc})
        print(f"[train] epoch={epoch} loss={mean:.4f} acc={acc:.4f}")

    elapsed = time.time() - t0
    ckpt_dir = Path(cfg["paths"]["checkpoint_dir"])
    log_dir = Path(cfg["paths"]["log_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "config": cfg}, ckpt_dir / "last.pt")
    summary = {
        "protocol": cfg["train"].get("protocol", "full"),
        "epochs": epochs,
        "n_train": len(train_ds),
        "elapsed_sec": elapsed,
        "history": hist,
        "checkpoint": str(ckpt_dir / "last.pt"),
    }
    with open(log_dir / "train_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")
    print(f"Wrote {log_dir / 'train_summary.json'} elapsed={elapsed:.1f}s")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    train(load_config(args.config), device_override=args.device)


if __name__ == "__main__":
    main()
