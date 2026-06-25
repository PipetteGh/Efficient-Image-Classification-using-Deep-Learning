"""
model.py
========
Modified MobileNetV3 (Large and Small variants) for CSCD618/DSCD604 Assignment 2.

Uses torchvision's MobileNetV3 with ImageNet pretrained weights.
The heavy classifier head is stripped and replaced with a single Linear layer.
"""

import torch
import torch.nn as nn
from torchvision import models

class MobileLiteNet(nn.Module):
    def __init__(self, num_classes: int = 16, dropout: float = 0.2,
                 pretrained: bool = True, model_type: str = "large"):
        super().__init__()
        self.model_type = model_type
        
        if pretrained:
            if model_type == "large":
                weights = models.MobileNet_V3_Large_Weights.IMAGENET1K_V2
            else:
                weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
        else:
            weights = None

        if model_type == "large":
            base = models.mobilenet_v3_large(weights=weights)
            last_channel = 960
        else:
            base = models.mobilenet_v3_small(weights=weights)
            last_channel = 576

        self.features = base.features
        self.pool = nn.AdaptiveAvgPool2d(1)

        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(last_channel, num_classes),
        )
        self._init_classifier()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def _init_classifier(self) -> None:
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def freeze_backbone(self) -> None:
        for param in self.features.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self) -> None:
        for param in self.features.parameters():
            param.requires_grad = True


def build_model(num_classes: int = 16, dropout: float = 0.2,
                pretrained: bool = True, model_type: str = "large") -> MobileLiteNet:
    return MobileLiteNet(num_classes=num_classes, dropout=dropout,
                         pretrained=pretrained, model_type=model_type)

def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def count_all_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())

if __name__ == "__main__":
    net_l = build_model(num_classes=16, model_type="large")
    net_s = build_model(num_classes=16, model_type="small")
    total = count_all_parameters(net_l) + count_all_parameters(net_s)
    print(f"Total Ensemble Parameters: {total:,} (Budget < 4,000,000)")
    assert total < 4_000_000, "Budget exceeded!"
    print("[OK] Model architecture passed.")
