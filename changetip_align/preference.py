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
    std_eps: float = 0.05,
    advantage_clip: float = 3.0,
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


def direct_prm_loss(
    prm,
    stages,
    policy_prob: torch.Tensor,
    pre: torch.Tensor,
    post: torch.Tensor,
    ext_feat: torch.Tensor = None,
    focus: torch.Tensor = None,
    squash: float = 0.0,
):
    """Direct, off-policy PRM-as-differentiable-loss.

    Bypasses GRPO's K-candidate sampling. We feed the policy's continuous
    probability map straight into the PRM and minimize -E[PRM(prob)]. The
    gradient flows back through the PRM (whose weights are frozen) into the
    policy, so any direction in mask-space the PRM doesn't like is pushed
    against. This is what we actually want when the PRM is well-trained but
    GRPO's K candidates are too similar to differentiate.

    squash > 0: tanh(score / squash) * squash before averaging, to bound
        outlier rewards (e.g. when PRM is over-confident on a few pixels and
        a single tile dominates the gradient). Recommended: squash=2.0 when
        the PRM gap is around 2.0–3.0.
    """
    score = prm(stages, policy_prob, pre, post, ext_feat=ext_feat)
    if squash > 0.0:
        score = torch.tanh(score / squash) * squash
    if focus is None:
        return -score.mean()
    return -(score * focus).sum() / (focus.sum() + 1e-6)


def prm_gated_kl(
    policy_logit: torch.Tensor,
    ref_logit: torch.Tensor,
    reward_map: torch.Tensor,
    temperature: float = 1.0,
    gate_mode: str = "uncertainty",
):
    """KL(pi_theta || pi_ref) with a per-pixel gate produced by the PRM.

    gate_mode='negative_reward' (legacy):
        gate = sigma(-T * R). KL is heavy on PRM-low-score pixels — pulls policy
        back to ref where the PRM thinks the current mask is bad. Risk: if ref
        is *also* wrong on those pixels, KL locks the policy in the wrong place.

    gate_mode='uncertainty' (new default):
        gate = exp(-(T * R)^2 / 2). KL is heavy where the PRM is *unsure*
        (|R| small) and zero where the PRM is decisive. This frees policy to
        follow the PRM whenever the PRM has a confident opinion (positive or
        negative) and only anchors to ref where signal is weak.
    """
    from .sampling import kl_bernoulli_per_pixel

    kl = kl_bernoulli_per_pixel(policy_logit, ref_logit)  # [B, 1, H, W]
    if gate_mode == "negative_reward":
        gate = torch.sigmoid(-temperature * reward_map)
    elif gate_mode == "uncertainty":
        gate = torch.exp(-(temperature * reward_map) ** 2 / 2.0)
    else:
        raise ValueError(f"Unknown gate_mode: {gate_mode}")
    return (kl * gate).sum() / (gate.sum() + 1e-6)


def unweighted_kl(policy_logit: torch.Tensor, ref_logit: torch.Tensor):
    """Plain KL(pi_theta || pi_ref) averaged over all pixels.
    Used by the w/o-PRM-gated-KL ablation."""
    from .sampling import kl_bernoulli_per_pixel

    return kl_bernoulli_per_pixel(policy_logit, ref_logit).mean()


def grpo_to_dpo_pair(masks: torch.Tensor, rewards: torch.Tensor):
    """Convert K candidates with per-pixel rewards into DPO chosen/rejected pair.

    masks:   [B, K, H, W] — the K sampled binary masks.
    rewards: [B, K, H, W] — per-pixel PRM rewards for each candidate.

    Returns:
        chosen, rejected: each [B, 1, H, W]. Chosen = candidate with the highest
        image-level mean reward; rejected = candidate with the lowest. This gives
        DPO a hard preference signal derived from the same PRM that GRPO uses,
        making the comparison apples-to-apples.
    """
    img_rewards = rewards.mean(dim=(2, 3))  # [B, K]
    best = img_rewards.argmax(dim=1)        # [B]
    worst = img_rewards.argmin(dim=1)       # [B]
    b = masks.shape[0]
    idx = torch.arange(b, device=masks.device)
    chosen = masks[idx, best].unsqueeze(1)
    rejected = masks[idx, worst].unsqueeze(1)
    return chosen, rejected


def cosine_lr(step, total_steps, base_lr, warmup_steps=500, min_lr_factor=0.2):
    if step <= warmup_steps:
        return base_lr * max(0.1, step / max(1, warmup_steps))
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    factor = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
    return base_lr * max(min_lr_factor, factor)
