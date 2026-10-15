"""CelebA images + 40 binary attributes (Male = demographic; rest = non-demographic)."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

# Re-export root resolver compatible with celeba.py layout
def resolve_celeba_root(cfg: dict) -> Path:
    root = Path(cfg["data"].get("celeba_root", "../../datasets/celeba/extracted/celeba"))
    if not root.is_absolute():
        root = (Path(__file__).resolve().parents[2] / root).resolve()
    if not (root / "img_align_celeba").is_dir():
        raise FileNotFoundError(root)
    if not (root / "list_attr_celeba.txt").is_file():
        raise FileNotFoundError(root / "list_attr_celeba.txt")
    return root


def load_celeba_attributes(root: Path) -> Tuple[List[str], Dict[str, np.ndarray]]:
    """
    Parse list_attr_celeba.txt.
    Returns (attr_names, {filename: int8 vector in {0,1}}).
    """
    path = Path(root) / "list_attr_celeba.txt"
    with open(path) as f:
        _ = f.readline()  # count
        names = f.readline().split()
        mapping: Dict[str, np.ndarray] = {}
        for line in f:
            toks = line.split()
            fn = toks[0]
            # file uses -1/1 → map to 0/1
            vals = np.array([(1 if int(v) > 0 else 0) for v in toks[1:]], dtype=np.int8)
            mapping[fn] = vals
    return names, mapping


def _tf(image_size: int, train: bool) -> transforms.Compose:
    ops = [transforms.Resize((image_size, image_size))]
    if train:
        ops.append(transforms.RandomHorizontalFlip())
    ops += [
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ]
    return transforms.Compose(ops)


class CelebAAttrDataset(Dataset):
    """Identity labels for FR train + attribute vector + gender flag."""

    def __init__(
        self,
        root: Path,
        split: str = "train",
        image_size: int = 112,
        train_ratio: float = 0.7,
        max_n: Optional[int] = None,
        seed: int = 42,
        train: bool = False,
    ) -> None:
        self.root = Path(root)
        self.img_dir = self.root / "img_align_celeba"
        self.transform = _tf(image_size, train=train)
        self.attr_names, attr_map = load_celeba_attributes(self.root)
        if "Male" not in self.attr_names:
            raise RuntimeError("CelebA attrs missing Male")
        self.male_idx = self.attr_names.index("Male")
        self.nd_indices = [i for i, n in enumerate(self.attr_names) if n != "Male"]
        self.nd_names = [self.attr_names[i] for i in self.nd_indices]

        id_map: Dict[str, int] = {}
        with open(self.root / "identity_CelebA.txt") as f:
            for line in f:
                fn, sid = line.split()
                id_map[fn] = int(sid)

        subjects = sorted(set(id_map.values()))
        rng = np.random.RandomState(seed)
        rng.shuffle(subjects)
        n_train = int(round(len(subjects) * train_ratio))
        train_subs = set(subjects[:n_train])
        want = split == "train"
        rows = [(fn, sid) for fn, sid in id_map.items() if (sid in train_subs) == want and fn in attr_map]
        if max_n is not None:
            rows = rows[: int(max_n)]
        uniq = sorted({r[1] for r in rows})
        remap = {s: i for i, s in enumerate(uniq)}
        self.samples = []
        for fn, sid in rows:
            a = attr_map[fn]
            self.samples.append((fn, remap[sid], int(a[self.male_idx]), a[self.nd_indices].copy()))
        self.num_classes = len(uniq)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        fn, sid, male, nd = self.samples[idx]
        img = self.transform(Image.open(self.img_dir / fn).convert("RGB"))
        return img, sid, male, torch.from_numpy(nd.astype(np.float32))
