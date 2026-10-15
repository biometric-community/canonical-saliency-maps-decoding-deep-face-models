# DEVIATIONS — Canonical Saliency Maps

| ID | Item | Paper | Ours | Reason |
|----|------|-------|------|--------|
| D1 | Canonical alignment | PRNet + 3DMMM dense UV | CelebA aligned crops as frame | PRNet stack not in monorepo |
| D2 | Backbone | VGG-Face multi-task | ResNet-18 gender only | Lab twin scope |
| D3 | Baselines | Grad-CAM / ++ / Score-CAM | Grad-CAM | Primary comparison |
| D4 | Eval scale | 22K CelebA subset | 128 / smoke-8 saliency images | Runtime gate |
