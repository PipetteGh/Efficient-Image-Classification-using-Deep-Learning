"""
model.py
========
Modified MobileNetV3-Large for CSCD618/DSCD604 Assignment 2 – Image Classification.

Uses torchvision's MobileNetV3-Large with ImageNet pretrained weights (IMAGENET1K_V2).
The heavy 1.2M parameter classifier head is completely stripped and replaced with a 
single Linear layer to keep the total parameters under 3,000,000.
Supports freezing/unfreezing the backbone for two-phase fine-tuning.
Parameter budget: < 4 000 000.
"""

import torch
import torch.nn as nn
from torchvision import models


# ──────────────────────────────────────────────────────────────────────────────
# Main Architecture
# ──────────────────────────────────────────────────────────────────────────────

class MobileLiteNet(nn.Module):
    """
    Modified MobileNetV3-Large with pretrained ImageNet weights.

    Architecture summary
    --------------------
    Backbone : MobileNetV3-Large features (pretrained on ImageNet)
    Head     : global avg pool → dropout → FC(960, num_classes)
    """

    def __init__(self, num_classes: int = 16, dropout: float = 0.2,
                 pretrained: bool = True, width_mult: float = 1.0):
        super().__init__()

        # Load pretrained MobileNetV3-Large
        if pretrained:
            weights = models.MobileNet_V3_Large_Weights.IMAGENET1K_V2
        else:
            weights = None

        base = models.mobilenet_v3_large(weights=weights)

        # Use the feature extractor
        self.features = base.features
        self.pool = nn.AdaptiveAvgPool2d(1)

        # New highly-efficient classifier head for our 16 classes
        last_channel = 960  # MobileNetV3-Large features output 960 channels
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(last_channel, num_classes),
        )

        # Initialize only the new classifier head
        self._init_classifier()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def _init_classifier(self) -> None:
        """Initialize only the classifier head weights."""
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def freeze_backbone(self) -> None:
        """Freeze the pretrained feature extractor."""
        for param in self.features.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self) -> None:
        """Unfreeze the pretrained feature extractor for full fine-tuning."""
        for param in self.features.parameters():
            param.requires_grad = True


# ──────────────────────────────────────────────────────────────────────────────
# Factory + sanity check
# ──────────────────────────────────────────────────────────────────────────────

def build_model(num_classes: int = 16, dropout: float = 0.2,
                pretrained: bool = True, width_mult: float = 1.0) -> MobileLiteNet:
    """Return an initialised MobileLiteNet (pretrained MobileNetV3-Large)."""
    model = MobileLiteNet(num_classes=num_classes, dropout=dropout,
                          pretrained=pretrained, width_mult=width_mult)
    return model


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_all_parameters(model: nn.Module) -> int:
    """Count ALL parameters (including frozen ones)."""
    return sum(p.numel() for p in model.parameters())


if __name__ == "__main__":
    import sys
    net = build_model(num_classes=16)
    total = count_all_parameters(net)
    trainable = count_parameters(net)
    dummy = torch.randn(2, 3, 224, 224)
    out = net(dummy)
    print(f"Output shape      : {out.shape}")
    print(f"Total parameters  : {total:,}")
    print(f"Trainable params  : {trainable:,}")
    assert total < 4_000_000, f"Parameter budget exceeded! {total:,} >= 4,000,000"
    assert out.shape == (2, 16), "Unexpected output shape!"
    print("[OK] Model architecture check passed.")
