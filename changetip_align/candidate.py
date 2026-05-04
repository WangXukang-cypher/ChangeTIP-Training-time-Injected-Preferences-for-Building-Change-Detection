from dataclasses import dataclass
from typing import Iterable, List

import torch
import torch.nn.functional as F


@dataclass
class CandidateSet:
    probs: torch.Tensor
    masks: torch.Tensor
    names: List[str]


def as_prob(output: torch.Tensor) -> torch.Tensor:
    if output.detach().min() >= 0.0 and output.detach().max() <= 1.0:
        return output.clamp(1e-6, 1.0 - 1e-6)
    return torch.sigmoid(output).clamp(1e-6, 1.0 - 1e-6)


@torch.no_grad()
def predict_prob(model, pre, post):
    return as_prob(model.update_bcd(pre, post))


@torch.no_grad()
def collect_probability_views(
    model,
    pre: torch.Tensor,
    post: torch.Tensor,
    use_flips: bool = True,
    scales: Iterable[float] = (),
):
    views = [("base", predict_prob(model, pre, post))]

    if use_flips:
        ph = predict_prob(model, torch.flip(pre, [3]), torch.flip(post, [3]))
        views.append(("hflip", torch.flip(ph, [3])))
        pv = predict_prob(model, torch.flip(pre, [2]), torch.flip(post, [2]))
        views.append(("vflip", torch.flip(pv, [2])))
        phv = predict_prob(model, torch.flip(pre, [2, 3]), torch.flip(post, [2, 3]))
        views.append(("hvflip", torch.flip(phv, [2, 3])))

    h, w = pre.shape[-2:]
    for scale in scales:
        if abs(float(scale) - 1.0) < 1e-6:
            continue
        sh, sw = max(16, int(h * scale)), max(16, int(w * scale))
        pre_s = F.interpolate(pre, size=(sh, sw), mode="bilinear", align_corners=False)
        post_s = F.interpolate(post, size=(sh, sw), mode="bilinear", align_corners=False)
        prob_s = predict_prob(model, pre_s, post_s)
        prob_s = F.interpolate(prob_s, size=(h, w), mode="bilinear", align_corners=False)
        views.append((f"scale{scale:.2f}", prob_s.clamp(1e-6, 1.0 - 1e-6)))

    return views


@torch.no_grad()
def generate_candidates(
    model,
    pre: torch.Tensor,
    post: torch.Tensor,
    thresholds: Iterable[float],
    use_flips: bool = True,
    scales: Iterable[float] = (),
) -> CandidateSet:
    was_training = model.training
    model.eval()
    views = collect_probability_views(model, pre, post, use_flips=use_flips, scales=scales)
    if was_training:
        model.train()
    probs = []
    masks = []
    names = []
    for view_name, prob in views:
        for threshold in thresholds:
            probs.append(prob)
            masks.append((prob >= float(threshold)).float())
            names.append(f"{view_name}@{threshold:.2f}")
    return CandidateSet(
        probs=torch.stack(probs, dim=1),
        masks=torch.stack(masks, dim=1),
        names=names,
    )


def gather_candidate(tensor: torch.Tensor, index: torch.Tensor):
    b = tensor.shape[0]
    view = index.view(b, 1, 1, 1, 1).expand(-1, 1, *tensor.shape[2:])
    return tensor.gather(1, view).squeeze(1)
