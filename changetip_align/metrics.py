import torch
import torch.nn.functional as F


def mask_iou_f1(mask: torch.Tensor, target: torch.Tensor):
    mask = (mask > 0.5).float()
    target = (target > 0.5).float()
    dims = tuple(range(1, mask.dim()))
    inter = (mask * target).sum(dim=dims)
    pred_sum = mask.sum(dim=dims)
    target_sum = target.sum(dim=dims)
    union = pred_sum + target_sum - inter
    iou = inter / (union + 1e-6)
    precision = inter / (pred_sum + 1e-6)
    recall = inter / (target_sum + 1e-6)
    f1 = 2.0 * precision * recall / (precision + recall + 1e-6)
    return iou, f1, precision, recall


def boundary_map(mask: torch.Tensor):
    kernel = torch.ones((1, 1, 3, 3), dtype=mask.dtype, device=mask.device)
    dilated = F.conv2d(mask, kernel, padding=1)
    return ((dilated > 0) & (dilated < 9)).float()


def boundary_f1(mask: torch.Tensor, target: torch.Tensor):
    edge_p = boundary_map((mask > 0.5).float())
    edge_t = boundary_map((target > 0.5).float())
    _, f1, _, _ = mask_iou_f1(edge_p, edge_t)
    return f1


def temporal_evidence(pre: torch.Tensor, post: torch.Tensor, mask: torch.Tensor):
    """Score whether selected pixels have stronger temporal difference than background."""
    mask = (mask > 0.5).float()
    diff = (post - pre).abs().mean(dim=1, keepdim=True)
    inside = (diff * mask).sum(dim=(1, 2, 3)) / (mask.sum(dim=(1, 2, 3)) + 1e-6)
    bg = 1.0 - mask
    outside = (diff * bg).sum(dim=(1, 2, 3)) / (bg.sum(dim=(1, 2, 3)) + 1e-6)
    return torch.sigmoid(8.0 * (inside - outside))


def candidate_quality(pre, post, mask, target, w_iou=0.55, w_boundary=0.25, w_evidence=0.20):
    iou, f1, _, _ = mask_iou_f1(mask, target)
    bnd = boundary_f1(mask, target)
    evid = temporal_evidence(pre, post, mask)
    return w_iou * f1 + w_boundary * bnd + w_evidence * evid
