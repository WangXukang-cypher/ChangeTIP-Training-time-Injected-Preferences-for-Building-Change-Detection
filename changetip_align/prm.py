from typing import List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class StageRewardHead(nn.Module):
    """A small fully-convolutional discriminator that scores a (feature, mask) pair
    at one fusion stage. Mask and pre/post difference are bilinear-resized to the
    stage resolution before concatenation."""

    def __init__(self, feat_channels: int, hidden: int = 64):
        super().__init__()
        in_ch = feat_channels + 1 + 1  # mask + |diff|.mean(channel)
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, hidden, 3, padding=1),
            nn.GroupNorm(8, hidden),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.GroupNorm(8, hidden),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, 1, 1),
        )

    def forward(self, z: torch.Tensor, mask: torch.Tensor, diff: torch.Tensor):
        if mask.shape[-2:] != z.shape[-2:]:
            mask = F.interpolate(mask, size=z.shape[-2:], mode="bilinear", align_corners=False)
        if diff.shape[-2:] != z.shape[-2:]:
            diff = F.interpolate(diff, size=z.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([z, mask, diff], dim=1)
        return self.net(x)  # [B, 1, H_l, W_l] logit


class MultiStageProcessReward(nn.Module):
    """Process Reward Model decomposed across spatial-head stages.

    Each stage gets its own discriminator head; the final reward is an
    upsample-and-mix of all stage logits to the input resolution. The deep
    stages dominate via a monotonically increasing weight schedule that is
    consistent with the monotonicity regularizer used in training.
    """

    def __init__(
        self,
        stage_channels: Sequence[int],
        hidden: int = 64,
        weight_temperature: float = 1.0,
    ):
        super().__init__()
        self.heads = nn.ModuleList(
            [StageRewardHead(c, hidden=hidden) for c in stage_channels]
        )
        # Weights softplus + softmax-ed across stages so they stay positive and sum to 1.
        # Initialized to favor later stages: w_l proportional to exp(l).
        init = torch.linspace(-1.0, 1.0, len(stage_channels)) * weight_temperature
        self.stage_logits = nn.Parameter(init)

    def stage_weights(self) -> torch.Tensor:
        return F.softmax(self.stage_logits, dim=0)

    def forward(
        self,
        stages: List[torch.Tensor],
        mask: torch.Tensor,
        pre: torch.Tensor,
        post: torch.Tensor,
        return_stage: bool = False,
    ):
        diff = (post - pre).abs().mean(dim=1, keepdim=True)
        target_size = mask.shape[-2:]
        weights = self.stage_weights()
        per_stage = []
        fused = 0.0
        for w, head, z in zip(weights, self.heads, stages):
            r = head(z, mask, diff)
            per_stage.append(r)
            r_up = F.interpolate(r, size=target_size, mode="bilinear", align_corners=False)
            fused = fused + w * r_up
        if return_stage:
            return fused, per_stage
        return fused

    @torch.no_grad()
    def reward_for_candidates(
        self,
        stages: List[torch.Tensor],
        masks: torch.Tensor,
        pre: torch.Tensor,
        post: torch.Tensor,
    ) -> torch.Tensor:
        """Score K candidate masks. masks: [B, K, 1, H, W]; returns [B, K, H, W]."""
        b, k = masks.shape[:2]
        rewards = []
        for i in range(k):
            r = self.forward(stages, masks[:, i], pre, post)
            rewards.append(r.squeeze(1))
        return torch.stack(rewards, dim=1)


def perturb_mask(mask: torch.Tensor, p_flip: float = 0.05, blob_sigma: float = 4.0):
    """Generate a 'fake but plausible' mask by random pixel flips followed by smoothing.
    Returns a mask in {0,1} with the same shape as input."""
    rand = torch.rand_like(mask)
    flipped = torch.where(rand < p_flip, 1.0 - mask, mask)
    return flipped


def prm_discriminator_loss(
    real_reward: torch.Tensor,
    fake_reward: torch.Tensor,
    label_smoothing: float = 0.05,
):
    real_target = torch.full_like(real_reward, 1.0 - label_smoothing)
    fake_target = torch.full_like(fake_reward, label_smoothing)
    real_loss = F.binary_cross_entropy_with_logits(real_reward, real_target)
    fake_loss = F.binary_cross_entropy_with_logits(fake_reward, fake_target)
    return 0.5 * (real_loss + fake_loss)


def prm_monotonicity_loss(per_stage_real: List[torch.Tensor], margin: float = 0.05):
    """Encourage mean stage-reward of GT masks to be non-decreasing across stages.

    For real (GT) input we want: mean(r_{l+1}) >= mean(r_l) + margin.
    Implemented as a hinge over consecutive pairs.
    """
    if len(per_stage_real) < 2:
        return per_stage_real[0].new_zeros(())
    means = [r.mean() for r in per_stage_real]
    losses = []
    for prev, nxt in zip(means[:-1], means[1:]):
        losses.append(F.relu(margin + prev - nxt))
    return torch.stack(losses).mean()


def prm_iou_calibration_loss(
    fake_reward_image: torch.Tensor,  # [B] mean reward of fake mask
    iou: torch.Tensor,                # [B] IoU of fake mask vs GT
):
    """Calibrate that masks with higher IoU receive higher mean reward.
    Pearson-style soft correlation, optimized as 1 - corr."""
    if fake_reward_image.numel() < 2:
        return fake_reward_image.new_zeros(())
    x = fake_reward_image - fake_reward_image.mean()
    y = iou - iou.mean()
    denom = (x.pow(2).sum().sqrt() * y.pow(2).sum().sqrt()).clamp(min=1e-6)
    corr = (x * y).sum() / denom
    return 1.0 - corr


def iou_pair(mask: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    mask = (mask > 0.5).float()
    target = (target > 0.5).float()
    dims = tuple(range(1, mask.dim()))
    inter = (mask * target).sum(dim=dims)
    union = mask.sum(dim=dims) + target.sum(dim=dims) - inter
    return inter / (union + 1e-6)
