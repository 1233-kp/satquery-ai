"""
backend/app/services/models/fusion_net.py
==========================================
Dual-branch gated optical+SAR fusion network for weakly-supervised
land-cover composition estimation.

Design:
  - Two SEPARATE lightweight conv encoders, one per modality (optical
    3-channel RGB, SAR 2-channel VV/VH backscatter). Not shared weights
    (unlike the Siamese change detector) -- these are genuinely different
    sensing modalities with different statistics, not two time steps of
    the same sensor.
  - Each encoder downsamples via strided convs to a small spatial map,
    global-average-pools it, and projects to a shared-size embedding.
  - The two embeddings are combined via a LEARNED SIGMOID GATE conditioned
    on both modalities -- gate = sigmoid(Linear([opt_emb, sar_emb])),
    fused = gate * opt_emb + (1 - gate) * sar_emb -- not concatenation.
    This lets the network down-weight whichever modality is less
    informative for a given scene (e.g. SAR under speckle noise, optical
    under cloud) rather than always trusting both equally.
  - A linear classifier head outputs per-class logits over BigEarthNet's
    19 official land-cover classes (multi-label, trained with BCE). The
    fusion_service.py caller aggregates these into the 3 coarse buckets
    (built-up/water/vegetation) the deterministic answer template reports
    -- this model only ever outputs verifiable classification logits, no
    free text.
"""
from __future__ import annotations

import torch
import torch.nn as nn

NUM_CLASSES = 19  # official BigEarthNet-19 label set


def _conv_bn_relu(in_ch: int, out_ch: int, stride: int = 2) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class ModalityEncoder(nn.Module):
    """4 stride-2 conv blocks + global average pool + linear projection.
    Input HxW must be divisible by 16 (128x128 is used throughout)."""

    def __init__(self, in_channels: int, channels=(32, 64, 128, 256), embed_dim: int = 256):
        super().__init__()
        layers = []
        c_prev = in_channels
        for c in channels:
            layers.append(_conv_bn_relu(c_prev, c, stride=2))
            c_prev = c
        self.conv = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Linear(channels[-1], embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.pool(x).flatten(1)
        return self.proj(x)


class GatedFusion(nn.Module):
    def __init__(self, embed_dim: int = 256):
        super().__init__()
        self.gate_fc = nn.Linear(embed_dim * 2, embed_dim)

    def forward(self, opt_emb: torch.Tensor, sar_emb: torch.Tensor) -> torch.Tensor:
        gate = torch.sigmoid(self.gate_fc(torch.cat([opt_emb, sar_emb], dim=1)))
        return gate * opt_emb + (1 - gate) * sar_emb


class OpticalSARFusionNet(nn.Module):
    """forward(optical, sar) -> (B, 19) multi-label logits.
    optical: (B, 3, H, W) in [0, 1]. sar: (B, 2, H, W) calibrated dB,
    normalized -- see geo_preprocessing.load_sar_backscatter."""

    def __init__(self, num_classes: int = NUM_CLASSES, embed_dim: int = 256, sar_channels: int = 2):
        super().__init__()
        self.optical_encoder = ModalityEncoder(3, embed_dim=embed_dim)
        self.sar_encoder = ModalityEncoder(sar_channels, embed_dim=embed_dim)
        self.fusion = GatedFusion(embed_dim)
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, optical: torch.Tensor, sar: torch.Tensor) -> torch.Tensor:
        opt_emb = self.optical_encoder(optical)
        sar_emb = self.sar_encoder(sar)
        fused = self.fusion(opt_emb, sar_emb)
        return self.classifier(fused)


def count_parameters(model: nn.Module) -> dict:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}


if __name__ == "__main__":
    model = OpticalSARFusionNet()
    counts = count_parameters(model)
    print(f"Total params: {counts['total']:,} ({counts['total'] / 1e6:.2f}M)")
    optical = torch.randn(2, 3, 128, 128)
    sar = torch.randn(2, 2, 128, 128)
    out = model(optical, sar)
    print(f"Output shape: {tuple(out.shape)} (expected (2, {NUM_CLASSES}))")
