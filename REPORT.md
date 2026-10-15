# REPORT — Canonical Saliency Maps

**Protocol:** `full`

Lab twin of John et al.: occlusion CIS aggregated to CMS on aligned CelebA
(canonical-frame proxy), compared to Grad-CAM on a gender classifier.

## Training

- Epochs: 5
- Train images: 141610
- Final train acc: 98.76%
- Elapsed (s): 718.3

![Training](outputs/figures/full/train_loss.svg)

## Gender accuracy (val)

- Accuracy: **96.95%** (n=60989)

## Saliency metrics

- Images: 128, patch=15, stride=8
- CIS Average Drop %: **8.54**
- Grad-CAM Average Drop %: **19.96**
- Win % (CIS > Grad-CAM): **14.1**
- CMS center mass: 0.2151

![Metrics](outputs/figures/full/saliency_metrics.svg)

![CMS](outputs/figures/full/cms.svg)

## Notes

- Paper uses PRNet 3D→canonical projection; we use CelebA aligned crops as proxy.
- See `DEVIATIONS.md`.
