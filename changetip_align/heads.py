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

    Ablation knobs:
      zero_init: if False, use Kaiming init on the output conv (breaks the
                 "init equals base" property — used for the w/o-zero-init ablation).
      mode:      "residual" (default) → final = sigmoid(base_logit + alpha * delta).
                 "direct"   → final = sigmoid(delta), the head replaces the base
                              decoder entirely (used for the w/o-residual ablation).
    """

    def __init__(
        self,
        in_channels=(24, 24, 48, 96),
        channels=64,
        init_scale=0.1,
        zero_init: bool = True,
        mode: str = "residual",
    ):
        super().__init__()
        if mode not in ("residual", "direct"):
            raise ValueError(f"Unknown head mode: {mode}")
        self.channels = channels
        self.mode = mode
        self.num_stages = len(in_channels) + 1
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
        if zero_init:
            nn.init.zeros_(self.out.weight)
            nn.init.zeros_(self.out.bias)
        else:
            nn.init.kaiming_normal_(self.out.weight, nonlinearity="linear")
            nn.init.zeros_(self.out.bias)

    def forward(self, features, base_prob, return_stages=False):
        projected = [proj(feat) for proj, feat in zip(self.projections, features)]
        x = projected[-1]
        stages = [x]
        for fusion, skip in zip(self.fusions, reversed(projected[:-1])):
            x = fusion(x, skip)
            stages.append(x)
        x = self.refine(x)
        stages.append(x)
        delta_logit = self.out(x)
        if delta_logit.shape[-2:] != base_prob.shape[-2:]:
            delta_logit = F.interpolate(delta_logit, size=base_prob.shape[-2:], mode="bilinear", align_corners=False)
        base_logit = prob_to_logit(base_prob)
        if self.mode == "residual":
            final_prob = torch.sigmoid(base_logit + self.residual_scale * delta_logit)
        else:  # "direct" — head replaces base decoder
            final_prob = torch.sigmoid(delta_logit)
        if return_stages:
            return final_prob, base_logit, delta_logit, stages
        return final_prob

    @property
    def stage_channels(self):
        return [self.channels] * self.num_stages
