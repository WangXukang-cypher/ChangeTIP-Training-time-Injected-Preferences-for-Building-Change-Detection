import math

import torch
import torch.nn.functional as F

from .candidate import as_prob


def prob_to_logits(prob):
    prob = prob.clamp(1e-6, 1.0 - 1e-6)
    return torch.log(prob) - torch.log1p(-prob)


def bernoulli_logprob(logits, mask):
    log_p1 = -F.softplus(-logits)
    log_p0 = -logits + log_p1
    return mask * log_p1 + (1.0 - mask) * log_p0


def masked_mean(value, focus):
    return (value * focus).sum(dim=(1, 2, 3)) / (focus.sum(dim=(1, 2, 3)) + 1e-6)


def focus_region(chosen, rejected, target=None, dilate=5):
    focus = ((chosen - rejected).abs() > 0).float()
    if target is not None:
        focus = torch.maximum(focus, (target > 0.5).float())
    if dilate > 1:
        if dilate % 2 == 0:
            dilate += 1
        focus = F.max_pool2d(focus, kernel_size=dilate, stride=1, padding=dilate // 2)
    return focus


def dpo_loss(policy_prob, ref_prob, chosen, rejected, beta=0.2, focus=None):
    policy_logits = prob_to_logits(policy_prob)
    ref_logits = prob_to_logits(ref_prob)
    if focus is None:
        focus = focus_region(chosen, rejected)

    lp_chosen = masked_mean(bernoulli_logprob(policy_logits, chosen), focus)
    lp_rejected = masked_mean(bernoulli_logprob(policy_logits, rejected), focus)
    lr_chosen = masked_mean(bernoulli_logprob(ref_logits, chosen), focus)
    lr_rejected = masked_mean(bernoulli_logprob(ref_logits, rejected), focus)
    advantage = (lp_chosen - lp_rejected) - (lr_chosen - lr_rejected)
    return F.softplus(-beta * advantage).mean()


def kl_bernoulli(policy_prob, ref_prob, focus=None):
    p = as_prob(policy_prob)
    q = as_prob(ref_prob)
    kl = p * torch.log(p / q) + (1.0 - p) * torch.log((1.0 - p) / (1.0 - q))
    if focus is None:
        return kl.mean()
    return masked_mean(kl, focus).mean()


def supervised_loss(prob, target):
    prob = as_prob(prob)
    target = (target > 0.5).float()
    bce = F.binary_cross_entropy(prob, target)
    inter = (prob * target).sum(dim=(1, 2, 3))
    dice = 1.0 - ((2.0 * inter + 1.0) / (prob.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) + 1.0)).mean()
    return 0.5 * bce + 0.5 * dice


def boundary_loss(prob, target):
    sobel_x = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=prob.dtype, device=prob.device)
    sobel_x = sobel_x.view(1, 1, 3, 3)
    sobel_y = sobel_x.transpose(2, 3)
    tgt_edge = F.conv2d(target, sobel_x, padding=1).abs() + F.conv2d(target, sobel_y, padding=1).abs()
    tgt_edge = (tgt_edge > 0).float()
    pred_edge = F.conv2d(prob, sobel_x, padding=1).abs() + F.conv2d(prob, sobel_y, padding=1).abs()
    pred_edge = torch.sigmoid(pred_edge)
    return F.binary_cross_entropy(pred_edge, tgt_edge)


def false_positive_penalty(prob, target, weight=2.0):
    return (weight * prob * (1.0 - target)).mean()


def pixel_grpo_loss(
    logp_new: torch.Tensor,
    logp_old: torch.Tensor,
    rewards: torch.Tensor,
    focus: torch.Tensor = None,
    clip_eps: float = 0.2,
    std_eps: float = 1e-4,
    advantage_clip: float = 5.0,
):
    """Pixel-level Group Relative Policy Optimization loss.

    Shapes:
        logp_new: [B, K, H, W] log pi_theta(mask_i | x), with grad
        logp_old: [B, K, H, W] log pi_theta_old(mask_i | x), no grad
        rewards : [B, K, H, W] per-pixel reward (no grad)
        focus   : [B, 1, H, W] optional spatial weight, broadcast over K

    Returns: scalar loss.

    The advantage is normalized within the group of K candidates per spatial location,
    which (i) removes the need for a learned value baseline, (ii) reduces variance
    relative to pairwise DPO under heavy-tailed rewards, and (iii) automatically
    rescales credit assignment per pixel.
    """
    mean_r = rewards.mean(dim=1, keepdim=True)
    std_r = rewards.std(dim=1, keepdim=True).clamp(min=std_eps)
    advantage = ((rewards - mean_r) / std_r).clamp(-advantage_clip, advantage_clip).detach()

    log_ratio = logp_new - logp_old.detach()
    ratio = torch.exp(log_ratio.clamp(-10.0, 10.0))
    # Trust-region mask: 1 where the PPO clip is *inactive* (ratio inside the
    # trust region OR pushing in the direction we want), 0 where the clip stops
    # the gradient. At single-epoch on-policy update the mask is all-ones.
    in_trust = ~(((advantage > 0) & (ratio > 1.0 + clip_eps)) |
                 ((advantage < 0) & (ratio < 1.0 - clip_eps)))
    in_trust = in_trust.float()
    # Score-function loss with PPO mask. Gradient at ratio=1 is exactly the
    # standard PPO gradient -A * grad(logp_new); displayed scalar is non-zero
    # because cov(A, logp_new) over K is non-zero.
    pixel_loss = -in_trust * advantage * logp_new

    if focus is None:
        return pixel_loss.mean()
    focus = focus.expand_as(pixel_loss)
    return (pixel_loss * focus).sum() / (focus.sum() + 1e-6)


def prm_gated_kl(
    policy_logit: torch.Tensor,
    ref_logit: torch.Tensor,
    reward_map: torch.Tensor,
    temperature: float = 1.0,
):
    """KL(pi_theta || pi_ref) with a per-pixel gate produced by the PRM.

    policy_logit, ref_logit: [B, 1, H, W] logits of base+residual policies.
    reward_map: [B, 1, H, W] PRM logit on the *current* policy mask.

    The gate sigma(temperature * reward_map) localizes regularization to pixels the
    PRM marks as low-quality, which is where pi_theta should *not* drift away from
    pi_ref. On already-confident-correct pixels the gate is small and KL is light.
    """
    from .sampling import kl_bernoulli_per_pixel

    kl = kl_bernoulli_per_pixel(policy_logit, ref_logit)  # [B, 1, H, W]
    gate = torch.sigmoid(-temperature * reward_map)  # high where PRM is uncertain or negative
    return (kl * gate).sum() / (gate.sum() + 1e-6)


def cosine_lr(step, total_steps, base_lr, warmup_steps=500, min_lr_factor=0.2):
    if step <= warmup_steps:
        return base_lr * max(0.1, step / max(1, warmup_steps))
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    factor = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
    return base_lr * max(min_lr_factor, factor)
