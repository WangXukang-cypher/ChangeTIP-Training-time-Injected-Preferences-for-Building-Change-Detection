import torch
import torch.nn as nn
import torch.nn.functional as F


def prob_to_logit(prob):
    prob = prob.clamp(1e-6, 1.0 - 1e-6)
    return torch.log(prob) - torch.log1p(-prob)


class PreActResidualConvUnit(nn.Module):
    """Small residual unit used by the TIPS/DPT dense heads."""

    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.norm1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.norm2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        x = F.relu(x, inplace=False)
        x = self.norm1(self.conv1(x))
        x = F.relu(x, inplace=False)
        x = self.norm2(self.conv2(x))
        return x + residual


class FeatureFusionBlock(nn.Module):
    """DPT-style top-down feature fusion for CNN feature pyramids."""

    def __init__(self, channels):
        super().__init__()
        self.residual_unit = PreActResidualConvUnit(channels)
        self.output_unit = PreActResidualConvUnit(channels)

    def forward(self, x, residual=None):
        if residual is not None:
            if x.shape[-2:] != residual.shape[-2:]:
                x = F.interpolate(x, size=residual.shape[-2:], mode="bilinear", align_corners=False)
            x = x + self.residual_unit(residual)
        return self.output_unit(x)


class SpatialResidualChangeHead(nn.Module):
    """TIPS-inspired dense residual head adapted to Change3D CNN features.

    The head predicts a residual logit from the encoder feature pyramid and adds it
    to the original Change3D decoder probability. The final prediction is exactly
    the base decoder output at initialization because the last convolution is zero.
    """

    def __init__(self, in_channels=(24, 24, 48, 96), channels=64, init_scale=0.1):
        super().__init__()
        self.projections = nn.ModuleList([nn.Conv2d(c, channels, 1, bias=False) for c in in_channels])
        self.fusions = nn.ModuleList([FeatureFusionBlock(channels) for _ in range(len(in_channels) - 1)])
        self.refine = nn.Sequential(
            PreActResidualConvUnit(channels),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.out = nn.Conv2d(channels, 1, 3, padding=1)
        self.residual_scale = nn.Parameter(torch.tensor(float(init_scale)))
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, features, base_prob):
        projected = [proj(feat) for proj, feat in zip(self.projections, features)]
        x = projected[-1]
        for fusion, skip in zip(self.fusions, reversed(projected[:-1])):
            x = fusion(x, skip)
        x = self.refine(x)
        delta_logit = self.out(x)
        if delta_logit.shape[-2:] != base_prob.shape[-2:]:
            delta_logit = F.interpolate(delta_logit, size=base_prob.shape[-2:], mode="bilinear", align_corners=False)
        return torch.sigmoid(prob_to_logit(base_prob) + self.residual_scale * delta_logit)
