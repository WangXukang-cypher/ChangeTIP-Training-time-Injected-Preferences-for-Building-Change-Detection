"""Shared building blocks for ViT-based change-detection backbones.

These are intentionally small — the heavy lifting happens in the post-training
stage (PRM + GRPO). The base decoder here is just strong enough to produce a
sensible base probability that the residual head can refine.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_ch, out_ch, k=3, p=1):
        super().__init__(
            nn.Conv2d(in_ch, out_ch, k, padding=p, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )


class ResUnit(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.body = nn.Sequential(
            ConvBNReLU(channels, channels, 3, 1),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x):
        return F.relu(x + self.body(x), inplace=False)


class FusionBlock(nn.Module):
    """DPT-style top-down: upsample coarse, fuse with skip, refine."""

    def __init__(self, channels):
        super().__init__()
        self.skip_unit = ResUnit(channels)
        self.out_unit = ResUnit(channels)

    def forward(self, x, skip=None):
        if skip is not None:
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = x + self.skip_unit(skip)
        return self.out_unit(x)


class CrossTimeFusion(nn.Module):
    """Fuse pre/post features into a change feature at a single stage.

    Inputs are at the same resolution and channel count.
    Output is the same resolution / channels as input.
    """

    def __init__(self, channels):
        super().__init__()
        self.fuse = nn.Sequential(
            nn.Conv2d(channels * 3, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            ResUnit(channels),
        )

    def forward(self, pre_feat, post_feat):
        diff = post_feat - pre_feat
        return self.fuse(torch.cat([pre_feat, post_feat, diff], dim=1))


class ChangeDecoder(nn.Module):
    """Multi-scale top-down decoder producing a single-channel sigmoid map.

    Mimics the role of Change3D's ``ChangeDecoder``: takes a list of stage
    features (fine→coarse) and outputs a probability at input resolution.
    """

    def __init__(self, in_dim, channels=128, has_sigmoid=True):
        super().__init__()
        self.has_sigmoid = has_sigmoid
        self.projs = nn.ModuleList([nn.Conv2d(c, channels, 1) for c in in_dim])
        self.fusions = nn.ModuleList([FusionBlock(channels) for _ in range(len(in_dim) - 1)])
        self.out = nn.Sequential(
            ConvBNReLU(channels, channels, 3, 1),
            nn.Conv2d(channels, 1, 1),
        )

    def forward(self, features):
        # features: list[B, C_i, H_i, W_i] from finest to coarsest
        projected = [p(f) for p, f in zip(self.projs, features)]
        x = projected[-1]
        for fusion, skip in zip(self.fusions, reversed(projected[:-1])):
            x = fusion(x, skip)
        # Upsample to input resolution (finest stage is at 1/4)
        x = F.interpolate(x, scale_factor=4, mode="bilinear", align_corners=False)
        x = self.out(x)
        if self.has_sigmoid:
            x = torch.sigmoid(x)
        return x
