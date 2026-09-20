"""
backend/app/services/models/siamese_unet.py
=============================================
FC-Siam-diff style bi-temporal change detector (Daudt, Le Saux & Boulch,
"Fully Convolutional Siamese Networks for Change Detection", 2018).

Design, spelled out because it's easy to accidentally build early-fusion
instead of a true Siamese network:

  - ONE encoder (weight-shared) is applied independently to image A and
    image B -- both passes use the identical Conv/BN weights. This is the
    "Siamese" part: the network learns a single feature representation
    space for either time step, rather than learning from a 6-channel
    concatenated input (which is early fusion, not this).
  - At every encoder resolution (4 skip levels + the bottleneck), we take
    the absolute difference |feat_A - feat_B| of the two branches' features
    at that level. This is the "diff" part -- it's what "FC-Siam-diff"
    means, as opposed to "FC-Siam-conc" (which concatenates instead of
    differencing).
  - The decoder is NOT shared: it's a single U-Net-style upsampling path
    that consumes those per-level difference features (deepest first),
    fusing each with the previous decoder stage via a skip concatenation,
    same as a standard U-Net decoder -- except the "skip" at each level is
    the difference tensor, not a single image's features.

Sized deliberately small (from-scratch training on a 6GB GPU, few-hour
budget): 4 encoder stages with channels [32, 64, 128, 256] and a 512-channel
bottleneck, ~7-8M parameters total (see count_parameters()).
"""
from __future__ import annotations

import torch
import torch.nn as nn


def conv_block(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class SharedEncoder(nn.Module):
    """Applied identically (same weights) to both time steps. Returns the
    4 pre-pool skip feature maps plus the bottleneck feature map."""

    def __init__(self, in_ch: int = 3, channels=(32, 64, 128, 256), bottleneck: int = 512):
        super().__init__()
        c1, c2, c3, c4 = channels
        self.enc1 = conv_block(in_ch, c1)
        self.enc2 = conv_block(c1, c2)
        self.enc3 = conv_block(c2, c3)
        self.enc4 = conv_block(c3, c4)
        self.bottleneck = conv_block(c4, bottleneck)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x: torch.Tensor):
        s1 = self.enc1(x)
        s2 = self.enc2(self.pool(s1))
        s3 = self.enc3(self.pool(s2))
        s4 = self.enc4(self.pool(s3))
        bn = self.bottleneck(self.pool(s4))
        return s1, s2, s3, s4, bn


class DecoderStage(nn.Module):
    """Upsamples the previous decoder feature, concatenates it with the
    per-level |A-B| difference skip, and fuses with a conv block."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.fuse = conv_block(out_ch + skip_ch, out_ch)

    def forward(self, x: torch.Tensor, skip_diff: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        x = torch.cat([x, skip_diff], dim=1)
        return self.fuse(x)


class SiameseUNet(nn.Module):
    """FC-Siam-diff change detector. forward(image_a, image_b) -> logits
    (B, 1, H, W), pre-sigmoid -- use BCEWithLogitsLoss for training and
    torch.sigmoid(...) > 0.5 for inference (matches change_service.py)."""

    def __init__(self, in_ch: int = 3, channels=(32, 64, 128, 256), bottleneck: int = 512):
        super().__init__()
        self.encoder = SharedEncoder(in_ch, channels, bottleneck)
        c1, c2, c3, c4 = channels

        self.dec4 = DecoderStage(bottleneck, c4, c4)
        self.dec3 = DecoderStage(c4, c3, c3)
        self.dec2 = DecoderStage(c3, c2, c2)
        self.dec1 = DecoderStage(c2, c1, c1)
        self.head = nn.Conv2d(c1, 1, kernel_size=1)

    def forward(self, image_a: torch.Tensor, image_b: torch.Tensor) -> torch.Tensor:
        a1, a2, a3, a4, a_bn = self.encoder(image_a)
        b1, b2, b3, b4, b_bn = self.encoder(image_b)  # same weights as image_a's pass

        diff_bn = torch.abs(a_bn - b_bn)
        diff4 = torch.abs(a4 - b4)
        diff3 = torch.abs(a3 - b3)
        diff2 = torch.abs(a2 - b2)
        diff1 = torch.abs(a1 - b1)

        x = self.dec4(diff_bn, diff4)
        x = self.dec3(x, diff3)
        x = self.dec2(x, diff2)
        x = self.dec1(x, diff1)
        return self.head(x)


def count_parameters(model: nn.Module) -> dict:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}


if __name__ == "__main__":
    model = SiameseUNet()
    counts = count_parameters(model)
    print(f"Total params: {counts['total']:,} ({counts['total'] / 1e6:.2f}M)")
    a = torch.randn(2, 3, 256, 256)
    b = torch.randn(2, 3, 256, 256)
    out = model(a, b)
    print(f"Output shape: {tuple(out.shape)} (expected (2, 1, 256, 256))")
