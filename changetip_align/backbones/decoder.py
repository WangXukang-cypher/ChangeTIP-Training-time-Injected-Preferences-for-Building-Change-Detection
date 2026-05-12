"""Shared building blocks for ViT-based change-detection backbones."""
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


class FRMFusion(nn.Module):
    """Feature Rectify Module — cross-temporal channel+spatial attention before fusion.

    Drop-in replacement for CrossTimeFusion. Inspired by NeXt2Former-CD (Simeoni
    et al., 2025): each temporal stream absorbs weighted signal from the other via
    learned channel-wise and spatial-wise attention gates, then the pair is merged.

    Compared with CrossTimeFusion (concat+1×1+ResUnit), FRM explicitly learns
    *which features* in T2 are informative for T1 (and vice versa), making it
    more tolerant to co-registration shifts and seasonal appearance change.
    """

    def __init__(self, channels, reduction=8):
        super().__init__()
        mid = max(channels // reduction, 8)
        # Channel attention: global-pooled [2C] → sigmoid gate of shape [2C]
        self.ch_fc = nn.Sequential(
            nn.Linear(channels * 2, mid),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels * 2),
            nn.Sigmoid(),
        )
        # Spatial attention: [2C, H, W] → 2 per-pixel weight maps
        self.sp_conv = nn.Sequential(
            nn.Conv2d(channels * 2, mid, 1, bias=False),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, 2, 1),
            nn.Sigmoid(),
        )
        # Learnable injection strength (initialized to 0.5 matching NeXt2Former)
        self.lambda_c = nn.Parameter(torch.tensor(0.5))
        self.lambda_s = nn.Parameter(torch.tensor(0.5))
        # Reduce rectified [2C] → [C]
        self.embed = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            ResUnit(channels),
        )

    def forward(self, pre_feat, post_feat):
        B, C, H, W = pre_feat.shape
        cat_feat = torch.cat([pre_feat, post_feat], dim=1)  # [B, 2C, H, W]

        # Channel-wise cross-temporal gate
        ch_w = self.ch_fc(cat_feat.mean(dim=(2, 3)))  # [B, 2C]
        cw_pre = ch_w[:, :C].view(B, C, 1, 1)    # weights from pre to inject into post
        cw_post = ch_w[:, C:].view(B, C, 1, 1)   # weights from post to inject into pre

        # Spatial cross-temporal gate
        sp_w = self.sp_conv(cat_feat)             # [B, 2, H, W]
        sw_pre = sp_w[:, 0:1]
        sw_post = sp_w[:, 1:2]

        # Rectify: each stream absorbs attention-weighted signal from the other
        rect_pre = pre_feat + self.lambda_c * cw_post * post_feat \
                            + self.lambda_s * sw_post * post_feat
        rect_post = post_feat + self.lambda_c * cw_pre * pre_feat \
                              + self.lambda_s * sw_pre * pre_feat

        return self.embed(torch.cat([rect_pre, rect_post], dim=1))


def build_fusion(channels: int, mode: str = "concat") -> nn.Module:
    """Factory: 'concat' → CrossTimeFusion, 'frm' → FRMFusion."""
    if mode == "frm":
        return FRMFusion(channels)
    return CrossTimeFusion(channels)


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
