# IMPLEMENTATION_PLAN — Canonical Saliency Maps

| Component | Choice |
|-----------|--------|
| Task model | ResNet-18 gender (CelebA Male) |
| CIS | Occlusion confidence drop, density-normalized |
| CMS | Mean CIS over val images (aligned CelebA = canonical proxy) |
| Baseline | Grad-CAM |
| Metrics | Average Drop %, Increase in Confidence, Win % |
