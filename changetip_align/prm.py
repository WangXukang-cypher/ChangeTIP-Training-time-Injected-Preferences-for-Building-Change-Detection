from typing import List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class StageRewardHead(nn.Module):
    """Cross-modal discriminator that scores (stage feature z, mask, diff).

    Key design: explicit cross-products `mask * z`, `(1 - mask) * z`, `mask * diff`
    are concatenated as input. This is essential — the discriminator must answer
    "do features inside the mask look like change?" and "do features outside the
    mask look like no-change?". Without explicit cross terms, a 2-layer conv has
    to learn these element-wise products through convolution weights, which is
    why the previous version stalled at gap ~0.15.

    Also widened (hidden=128) and deepened (4 conv blocks) with a dilated layer
    for a larger receptive field, since the deepest stages are at 8x8 spatial
    resolution and need to integrate global context.
    """

    def __init__(self, feat_channels: int, hidden: int = 128, ext_channels: int = 0):
        super().__init__()
        # Inputs: z, mask*z, (1-mask)*z, mask, diff, mask*diff [, e, mask*e]
        self.ext_channels = ext_channels
        in_ch = 3 * feat_channels + 3
        if ext_channels > 0:
            # Project external features to a small width before fusion to keep
            # the conv input manageable when ext_channels is large (e.g. 1920
            # for ResNet18 multi-scale concat).
            self.ext_proj = nn.Conv2d(ext_channels, 64, 1)
            in_ch += 2 * 64
        else:
            self.ext_proj = None
        groups = 8 if hidden % 8 == 0 else 1
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, hidden, 3, padding=1),
            nn.GroupNorm(groups, hidden),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.GroupNorm(groups, hidden),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, hidden, 3, padding=2, dilation=2),
            nn.GroupNorm(groups, hidden),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, hidden, 3, padding=4, dilation=4),
            nn.GroupNorm(groups, hidden),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, 1, 1),
        )

    def forward(
        self,
        z: torch.Tensor,
        mask: torch.Tensor,
        diff: torch.Tensor,
        ext_feat: torch.Tensor = None,
    ):
        if mask.shape[-2:] != z.shape[-2:]:
            mask = F.interpolate(mask, size=z.shape[-2:], mode="bilinear", align_corners=False)
        if diff.shape[-2:] != z.shape[-2:]:
            diff = F.interpolate(diff, size=z.shape[-2:], mode="bilinear", align_corners=False)
        z_in = z * mask
        z_out = z * (1.0 - mask)
        d_in = diff * mask
        parts = [z, z_in, z_out, mask, diff, d_in]
        if self.ext_proj is not None and ext_feat is not None:
            if ext_feat.shape[-2:] != z.shape[-2:]:
                ext_feat = F.interpolate(
                    ext_feat, size=z.shape[-2:], mode="bilinear", align_corners=False
                )
            e = self.ext_proj(ext_feat)
            parts.append(e)
            parts.append(e * mask)
        x = torch.cat(parts, dim=1)
        return self.net(x)


class MultiStageProcessReward(nn.Module):
    """Multi-Stage Auxiliary Discriminator (MSAD).

    Originally introduced as a Process Reward Model (PRM) in our staged
    GRPO setup, this module is functionally a multi-scale conditional
    discriminator. In the end-to-end training pipeline (``train_e2e.py``)
    it is jointly optimised with the decoder as an auxiliary network whose
    role is to:

      1. Discriminate ground-truth masks from perturbed/policy masks,
         providing a learned structural-correctness signal beyond
         pixel-wise BCE/Dice.
      2. Provide a smooth gradient for the decoder via a frozen-MSAD
         backward pass (the "direct" path in train_e2e.py).
      3. Distil group-relative preference among K stochastic candidate
         masks into both itself and (through the direct path) the decoder.

    Architecture: each spatial-head stage gets its own per-stage head; the
    final score is an upsample-and-mix of all stage logits to the input
    resolution. The deep stages dominate via a monotonically increasing
    weight schedule that is consistent with the monotonicity regulariser.

    Naming: we keep the original class name ``MultiStageProcessReward`` to
    preserve checkpoint compatibility. New code should refer to it as MSAD.
    """

    def __init__(
        self,
        stage_channels: Sequence[int],
        hidden: int = 128,
        weight_temperature: float = 1.0,
        ext_channels: int = 0,
    ):
        super().__init__()
        self.ext_channels = ext_channels
        self.heads = nn.ModuleList(
            [
                StageRewardHead(c, hidden=hidden, ext_channels=ext_channels)
                for c in stage_channels
            ]
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
        ext_feat: torch.Tensor = None,
        return_stage: bool = False,
    ):
        diff = (post - pre).abs().mean(dim=1, keepdim=True)
        target_size = mask.shape[-2:]
        weights = self.stage_weights()
        per_stage = []
        fused = 0.0
        for w, head, z in zip(weights, self.heads, stages):
            r = head(z, mask, diff, ext_feat=ext_feat)
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
        ext_feat: torch.Tensor = None,
    ) -> torch.Tensor:
        """Score K candidate masks. masks: [B, K, 1, H, W]; returns [B, K, H, W]."""
        b, k = masks.shape[:2]
        rewards = []
        for i in range(k):
            r = self.forward(stages, masks[:, i], pre, post, ext_feat=ext_feat)
            rewards.append(r.squeeze(1))
        return torch.stack(rewards, dim=1)


def perturb_mask(mask: torch.Tensor, p_flip: float = 0.20, mode: str = "mix"):
    """Generate a 'fake but distinguishable' mask.

    Modes:
        flip:  iid pixel flips with probability p_flip.
        erase: zero out a random rectangle (false-negative style fake).
        smear: dilate the GT mask with a max-pool to create false-positive style fake.
        shift: translate the mask by a few pixels.
        mix:   per-sample randomly choose one of the above.

    The previous version used p_flip=0.05 which left 95% of pixels identical to
    GT and flooded the discriminator with unsolvable label noise. Structured
    perturbations create masks that are coherent but clearly *wrong*, providing
    the strong negative signal the PRM needs.
    """
    b, _, h, w = mask.shape
    out = mask.clone()

    if mode == "flip" or mode == "mix":
        rand = torch.rand_like(mask)
        flipped = torch.where(rand < p_flip, 1.0 - mask, mask)
    if mode == "flip":
        return flipped

    if mode == "erase" or mode == "smear" or mode == "shift" or mode == "mix":
        for i in range(b):
            if mode == "mix":
                choice = int(torch.randint(0, 4, (1,)).item())
            else:
                choice = {"flip": 0, "erase": 1, "smear": 2, "shift": 3}[mode]
            if choice == 0:
                out[i] = flipped[i]
            elif choice == 1:
                rh = int(torch.randint(h // 6, h // 2, (1,)).item())
                rw = int(torch.randint(w // 6, w // 2, (1,)).item())
                ry = int(torch.randint(0, h - rh + 1, (1,)).item())
                rx = int(torch.randint(0, w - rw + 1, (1,)).item())
                out[i, 0, ry:ry + rh, rx:rx + rw] = 0.0
            elif choice == 2:
                k = int(torch.randint(5, 13, (1,)).item())
                if k % 2 == 0:
                    k += 1
                out[i:i + 1] = F.max_pool2d(mask[i:i + 1], kernel_size=k, stride=1, padding=k // 2)
            elif choice == 3:
                dy = int(torch.randint(-12, 13, (1,)).item())
                dx = int(torch.randint(-12, 13, (1,)).item())
                shifted = torch.roll(mask[i], shifts=(dy, dx), dims=(-2, -1))
                out[i] = shifted
        return out
    raise ValueError(f"unknown perturb mode: {mode}")


def prm_discriminator_loss(
    real_reward: torch.Tensor,
    fake_reward: torch.Tensor,
    label_smoothing: float = 0.05,
    bt_weight: float = 1.0,
    hinge_weight: float = 0.5,
    margin: float = 1.0,
):
    """Per-pixel BCE + image-level Bradley-Terry pairwise loss + a residual
    hinge.

    BCE anchors the absolute level (real around +2.94, fake around -2.94 with
    label_smoothing 0.05). BT `-log sigma(r_real - r_fake)` is the standard
    RLHF reward-model objective and provides a non-vanishing gradient on the
    image-level gap regardless of absolute position. The hinge keeps a strong
    linear gradient when the gap is below `margin`, which empirically helps the
    early-training phase where BT alone is sigmoid-soft (gradient ~0.5 at gap=0).
    """
    real_target = torch.full_like(real_reward, 1.0 - label_smoothing)
    fake_target = torch.full_like(fake_reward, label_smoothing)
    real_loss = F.binary_cross_entropy_with_logits(real_reward, real_target)
    fake_loss = F.binary_cross_entropy_with_logits(fake_reward, fake_target)
    bce = 0.5 * (real_loss + fake_loss)

    real_img = real_reward.mean(dim=(1, 2, 3))
    fake_img = fake_reward.mean(dim=(1, 2, 3))
    bt = F.softplus(-(real_img - fake_img)).mean()
    hinge = F.relu(margin - (real_img - fake_img)).mean()
    return bce + bt_weight * bt + hinge_weight * hinge


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


# Public alias so new code can use the descriptive name without breaking the
# checkpoints that store ``MultiStageProcessReward`` weights under that key.
MultiStageAuxDiscriminator = MultiStageProcessReward
