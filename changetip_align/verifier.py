import torch
import torch.nn as nn
import torch.nn.functional as F


class ChangeVerifier(nn.Module):
    """A lightweight learned verifier for candidate change masks."""

    def __init__(self, in_channels=10, width=32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, width, 3, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
            nn.Conv2d(width, width, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
            nn.Conv2d(width, width * 2, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(width * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(width * 2, width * 4, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(width * 4),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(width * 4, width),
            nn.ReLU(inplace=True),
            nn.Linear(width, 1),
        )

    def forward(self, pre, post, mask):
        diff = (post - pre).abs()
        x = torch.cat([pre, post, diff, mask], dim=1)
        return self.head(self.encoder(x)).squeeze(-1)


def score_candidates(verifier, pre, post, masks, chunk_size=16):
    b, k = masks.shape[:2]
    scores = []
    for start in range(0, k, chunk_size):
        end = min(k, start + chunk_size)
        n = end - start
        pre_rep = pre[:, None].expand(-1, n, -1, -1, -1).reshape(b * n, *pre.shape[1:])
        post_rep = post[:, None].expand(-1, n, -1, -1, -1).reshape(b * n, *post.shape[1:])
        mask_flat = masks[:, start:end].reshape(b * n, *masks.shape[2:])
        scores.append(verifier(pre_rep, post_rep, mask_flat).view(b, n))
    return torch.cat(scores, dim=1)


def verifier_loss(pred_score, target_score, margin=0.05):
    mse = F.mse_loss(torch.sigmoid(pred_score), target_score)
    if pred_score.numel() < 2:
        return mse
    order = target_score[:, :, None] - target_score[:, None, :]
    pred_diff = pred_score[:, :, None] - pred_score[:, None, :]
    valid = order.abs() > margin
    if not valid.any():
        return mse
    rank_target = (order > 0).float()
    rank_loss = F.binary_cross_entropy_with_logits(pred_diff[valid], rank_target[valid])
    return mse + 0.2 * rank_loss
