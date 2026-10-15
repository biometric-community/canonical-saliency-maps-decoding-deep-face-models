"""Figures + REPORT.md."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from cansal.plot_style import FIGSIZE, LINEWIDTH_MAIN, MULTI_SERIES_COLORS, apply_rcparams, apply_style


def plot_train(hist, out: Path) -> None:
    apply_rcparams()
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.plot([h["epoch"] for h in hist], [h["loss"] for h in hist], color=MULTI_SERIES_COLORS[0], linewidth=LINEWIDTH_MAIN, label="loss")
    ax2 = ax.twinx()
    ax2.plot([h["epoch"] for h in hist], [h["acc"] * 100 for h in hist], color=MULTI_SERIES_COLORS[1], linewidth=LINEWIDTH_MAIN, label="acc")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("CE loss")
    ax2.set_ylabel("Train acc (%)")
    ax.set_title("Gender classifier training (CelebA)")
    apply_style(ax)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, format="svg")
    plt.close(fig)


def plot_metrics(sal: dict, out: Path) -> None:
    apply_rcparams()
    fig, ax = plt.subplots(figsize=FIGSIZE)
    methods = ["CIS (ours)", "Grad-CAM"]
    drops = [sal["cis"]["avg_drop_pct"], sal["gradcam"]["avg_drop_pct"]]
    x = np.arange(len(methods))
    ax.bar(x, drops, color=MULTI_SERIES_COLORS[0], edgecolor="#002080")
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.set_ylabel("Average Drop %")
    ax.set_title(f"Explanation fidelity (Win% CIS={sal['win_pct_cis_vs_gradcam']:.1f})")
    apply_style(ax, grid_axis="y")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, format="svg")
    plt.close(fig)


def plot_cms(cms: np.ndarray, out: Path) -> None:
    apply_rcparams()
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(cms, cmap="hot")
    ax.set_title("Canonical Model Saliency (gender)")
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.046)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, format="svg")
    plt.close(fig)


def write_report(cfg, train_s, eval_res, fig_dir: Path, path: Path) -> None:
    protocol = cfg["train"].get("protocol", "full")
    sal = eval_res.get("saliency", {})
    gen = eval_res.get("gender", {})
    lines = [
        "# REPORT — Canonical Saliency Maps",
        "",
        f"**Protocol:** `{protocol}`",
        "",
        "Lab twin of John et al.: occlusion CIS aggregated to CMS on aligned CelebA",
        "(canonical-frame proxy), compared to Grad-CAM on a gender classifier.",
        "",
        "## Training",
        "",
        f"- Epochs: {train_s.get('epochs')}",
        f"- Train images: {train_s.get('n_train')}",
        f"- Final train acc: {train_s['history'][-1]['acc']*100:.2f}%",
        f"- Elapsed (s): {train_s.get('elapsed_sec', 0):.1f}",
        "",
        f"![Training]({fig_dir.as_posix()}/train_loss.svg)",
        "",
        "## Gender accuracy (val)",
        "",
        f"- Accuracy: **{gen.get('accuracy', 0)*100:.2f}%** (n={gen.get('n', 0)})",
        "",
        "## Saliency metrics",
        "",
        f"- Images: {sal.get('n_images')}, patch={sal.get('patch')}, stride={sal.get('stride')}",
        f"- CIS Average Drop %: **{sal.get('cis', {}).get('avg_drop_pct', 0):.2f}**",
        f"- Grad-CAM Average Drop %: **{sal.get('gradcam', {}).get('avg_drop_pct', 0):.2f}**",
        f"- Win % (CIS > Grad-CAM): **{sal.get('win_pct_cis_vs_gradcam', 0):.1f}**",
        f"- CMS center mass: {sal.get('cms_center_mass', 0):.4f}",
        "",
        f"![Metrics]({fig_dir.as_posix()}/saliency_metrics.svg)",
        "",
        f"![CMS]({fig_dir.as_posix()}/cms.svg)",
        "",
        "## Notes",
        "",
        "- Paper uses PRNet 3D→canonical projection; we use CelebA aligned crops as proxy.",
        "- See `DEVIATIONS.md`.",
        "",
    ]
    path.write_text("\n".join(lines))
    print(f"Wrote {path}")


def generate(cfg: dict) -> None:
    log_dir = Path(cfg["paths"]["log_dir"])
    fig_dir = Path(cfg["paths"]["figure_dir"])
    fig_dir.mkdir(parents=True, exist_ok=True)
    train_s = json.loads((log_dir / "train_summary.json").read_text())
    eval_res = json.loads((log_dir / "eval.json").read_text())
    plot_train(train_s["history"], fig_dir / "train_loss.svg")
    plot_metrics(eval_res.get("saliency", {}), fig_dir / "saliency_metrics.svg")
    cms_path = log_dir / "cms.npy"
    if cms_path.is_file():
        plot_cms(np.load(cms_path), fig_dir / "cms.svg")
    write_report(cfg, train_s, eval_res, fig_dir, Path("REPORT.md"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args()
    generate(yaml.safe_load(Path(args.config).read_text()))


if __name__ == "__main__":
    main()
