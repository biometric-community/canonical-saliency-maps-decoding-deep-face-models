"""ResNet-18 gender classifier (CelebA Male)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class GenderNet(nn.Module):
    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        net = models.resnet18(weights=weights)
        self.features = nn.Sequential(*list(net.children())[:-2])  # → (B,512,H',W')
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, 2)

    def forward(self, x: torch.Tensor):
        feat = self.features(x)
        pooled = torch.flatten(self.pool(feat), 1)
        logits = self.fc(pooled)
        return logits, feat

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        logits, _ = self.forward(x)
        return F.softmax(logits, dim=1)
