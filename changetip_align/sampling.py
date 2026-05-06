import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _gaussian_kernel(sigma: float):
    ksize = max(3, int(2 * round(2 * sigma) + 1))
    if ksize % 2 == 0:
        ksize += 1
    coords = torch.arange(ksize).float() - (ksize - 1) / 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    kernel = (g[:, None] * g[None, :]).view(1, 1, ksize, ksize)
    return kernel, ksize // 2


class LowRankSpatialNoise(nn.Module):
    """Spatially correlated noise via Gaussian low-pass filtering of i.i.d. samples.

    A standard Bernoulli policy on H*W independent pixels has effective sample size
    H*W; this kills group-relative advantages because per-sample reward concentrates
    around its mean. Filtering iid Gaussian noise with a Gaussian kernel of stddev
    sigma reduces the effective rank to roughly (H*W) / sigma^2, producing samples
    whose log-prob differences carry useful signal.
    """

    def __init__(self, sigma: float = 4.0):
        super().__init__()
        self.sigma = float(sigma)
        if self.sigma > 0.0:
            kernel, padding = _gaussian_kernel(self.sigma)
            self.register_buffer("kernel", kernel, persistent=False)
            self.padding = padding
        else:
            self.kernel = None
            self.padding = 0

    def forward(self, batch: int, k: int, height: int, width: int, device, dtype):
        z = torch.randn(batch * k, 1, height, width, device=device, dtype=dtype)
        if self.sigma > 0.0:
            z = F.conv2d(z, self.kernel, padding=self.padding)
            # normalize each sample to roughly unit variance
            std = z.flatten(1).std(dim=1, keepdim=True).view(-1, 1, 1, 1).clamp(min=1e-6)
            z = z / std
        # sigma <= 0 → i.i.d. Gaussian (used by w/o-spatial-noise ablation).
        return z.view(batch, k, height, width)


def stochastic_logits(
    base_logit: torch.Tensor,
    delta_logit: torch.Tensor,
    residual_scale,
    noise_module: LowRankSpatialNoise,
    k: int,
    tau: float,
):
    """Build K perturbed pixel-Bernoulli logit maps.

    base_logit:  [B, 1, H, W] frozen reference logit (no grad expected).
    delta_logit: [B, 1, H, W] residual logit from spatial head (with grad).
    Returns logits of shape [B, K, H, W].
    """
    b, _, h, w = base_logit.shape
    base = base_logit.expand(b, k, h, w)
    delta = delta_logit.expand(b, k, h, w)
    if k > 1 and tau > 0.0:
        noise = noise_module(b, k, h, w, base.device, base.dtype) * tau
    else:
        noise = torch.zeros_like(base)
    return base + residual_scale * delta + noise


def bernoulli_logprob(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    log_p1 = -F.softplus(-logits)
    log_p0 = -logits + log_p1
    return mask * log_p1 + (1.0 - mask) * log_p0


@torch.no_grad()
def sample_masks(logits: torch.Tensor) -> torch.Tensor:
    """Stochastic Bernoulli sample with the Gumbel-trick for stability."""
    probs = torch.sigmoid(logits)
    return torch.bernoulli(probs)


def kl_bernoulli_per_pixel(p_logit: torch.Tensor, q_logit: torch.Tensor) -> torch.Tensor:
    """Per-pixel KL(p || q) for two Bernoulli policies parameterized by logits.

    KL = p * (log p - log q) + (1-p) * (log(1-p) - log(1-q))
       = p * (sp(-q) - sp(-p)) + (1-p) * (sp(q) - sp(p))
    where sp(x) = log(1+e^x). This avoids ever evaluating log(0).
    """
    p = torch.sigmoid(p_logit)
    sp_p_pos = F.softplus(p_logit)        # log(1+e^p)   = -log(1-sigmoid(p))
    sp_p_neg = F.softplus(-p_logit)       # log(1+e^-p)  = -log(sigmoid(p))
    sp_q_pos = F.softplus(q_logit)
    sp_q_neg = F.softplus(-q_logit)
    return p * (sp_q_neg - sp_p_neg) + (1.0 - p) * (sp_q_pos - sp_p_pos)
