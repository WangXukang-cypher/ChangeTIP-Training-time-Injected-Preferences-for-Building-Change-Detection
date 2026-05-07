"""Best-of-N inference with PRM-based candidate selection.

For each test image we:
  1. Forward the policy once to get (base_logit, delta_logit, stages).
  2. Sample K candidate logit maps using the same low-rank spatial noise used at
     training time (or a different sigma/tau if you want a wider search).
  3. Score each candidate with the frozen PRM.
  4. Combine the K candidates by one of three strategies:
        hard       — pick the K-th candidate with the highest image-level mean reward.
        soft       — softmax-weighted average of probabilities, weights = softmax(R).
        per_pixel  — at each pixel independently pick the K-th candidate with the
                     highest per-pixel reward (most aggressive; may produce
                     non-smooth masks).

The PRM never receives gradients, so reward hacking is impossible by
construction. K=1 reduces to the standard deterministic-policy evaluation.
"""
import argparse

import torch
import torch.nn.functional as F

from .data import build_loader, split_batch
from .external_prior import ExternalPrior
from .imports import add_change3d_root
from .models import build_change3d_model
from .prm import MultiStageProcessReward
from .sampling import LowRankSpatialNoise, sample_masks, stochastic_logits


def load_prm(path: str, device: str):
    ckpt = torch.load(path, map_location="cpu")
    if not isinstance(ckpt, dict) or "stage_channels" not in ckpt:
        raise ValueError(f"PRM checkpoint missing stage_channels: {path}")
    ext_channels = ckpt.get("ext_channels", 0)
    ext_backbone = ckpt.get("external_backbone", "")
    prm = MultiStageProcessReward(
        ckpt["stage_channels"], ext_channels=ext_channels,
    ).to(device)
    prm.load_state_dict(ckpt["state_dict"], strict=True)
    prm.eval()
    for param in prm.parameters():
        param.requires_grad = False
    ext_prior = None
    if ext_channels > 0 and ext_backbone:
        ext_prior = ExternalPrior(backbone=ext_backbone).to(device)
        print(f"[ExternalPrior] loaded {ext_backbone} (out_channels={ext_channels})")
    return prm, ext_prior


@torch.no_grad()
def select_bon(args, model, prm, ext_prior, noise_module, pre, post):
    """Run policy + PRM and return a single combined probability map [B, 1, H, W]."""
    out = model.update_bcd_full(pre, post)
    base_logit = out["base_logit"]
    delta_logit = out["delta_logit"]
    scale = out["residual_scale"]
    stages = out["stages"]

    deterministic_prob = torch.sigmoid(base_logit + scale * delta_logit)
    if args.bon_k <= 1:
        return deterministic_prob

    sampling_logits = stochastic_logits(
        base_logit, delta_logit, scale, noise_module,
        k=args.bon_k, tau=args.bon_tau,
    )  # [B, K, H, W]
    probs_k = torch.sigmoid(sampling_logits)  # [B, K, H, W]

    # PRM input: either continuous probs (soft) or hard binary samples.
    if args.bon_use_prob:
        candidates = probs_k.unsqueeze(2)  # [B, K, 1, H, W]
    else:
        candidates = sample_masks(sampling_logits).unsqueeze(2)  # [B, K, 1, H, W]

    ext_feat = ext_prior(pre, post) if ext_prior is not None else None
    rewards = prm.reward_for_candidates(
        stages, candidates, pre, post, ext_feat=ext_feat,
    )  # [B, K, H, W]

    if args.bon_include_deterministic:
        # Always include the deterministic policy as one of the candidates so BoN
        # never loses to plain inference.
        det_prob = deterministic_prob.unsqueeze(1)  # [B, 1, 1, H, W]
        det_mask = (det_prob > 0.5).float() if not args.bon_use_prob else det_prob
        det_reward = prm.reward_for_candidates(
            stages, det_mask, pre, post, ext_feat=ext_feat,
        )  # [B, 1, H, W]
        probs_k = torch.cat([probs_k, deterministic_prob], dim=1)  # [B, K+1, H, W]
        rewards = torch.cat([rewards, det_reward], dim=1)

    if args.bon_mode == "hard":
        img_rewards = rewards.mean(dim=(2, 3))           # [B, K]
        best_idx = img_rewards.argmax(dim=1)             # [B]
        b = pre.shape[0]
        idx = torch.arange(b, device=pre.device)
        return probs_k[idx, best_idx].unsqueeze(1)       # [B, 1, H, W]

    if args.bon_mode == "soft":
        img_rewards = rewards.mean(dim=(2, 3))           # [B, K]
        weights = F.softmax(img_rewards / args.bon_temperature, dim=1)
        weights = weights.view(-1, weights.shape[1], 1, 1)
        return (probs_k * weights).sum(dim=1, keepdim=True)

    if args.bon_mode == "per_pixel":
        best_idx = rewards.argmax(dim=1, keepdim=True)   # [B, 1, H, W]
        return probs_k.gather(dim=1, index=best_idx)     # [B, 1, H, W]

    raise ValueError(f"Unknown bon_mode: {args.bon_mode}")


@torch.no_grad()
def evaluate_bon(args, model, prm, ext_prior, noise_module, loader, thresholds):
    from utils.metric_tool import ConfuseMatrixMeter

    meters = {th: ConfuseMatrixMeter(n_class=2) for th in thresholds}
    model.eval()
    for batch in loader:
        pre, post, target = split_batch(batch, args.device)
        prob = select_bon(args, model, prm, ext_prior, noise_module, pre, post)
        if args.eval_tta:
            pre_h, post_h = torch.flip(pre, [3]), torch.flip(post, [3])
            prob_h = select_bon(args, model, prm, ext_prior, noise_module, pre_h, post_h)
            prob = 0.5 * prob + 0.5 * torch.flip(prob_h, [3])
        for th, meter in meters.items():
            pred = (prob > th).long().cpu().numpy()
            meter.update_cm(pr=pred, gt=target.cpu().numpy())
    return {th: meter.get_scores() for th, meter in meters.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--change3d_root", default="..")
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--ckpt", required=True, help="Policy checkpoint (baseline or post-trained).")
    parser.add_argument("--prm_ckpt", required=True, help="Frozen PRM used as the verifier.")
    parser.add_argument("--pretrained", default="")
    parser.add_argument("--dataset", default="WHU-CD")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--in_height", type=int, default=256)
    parser.add_argument("--in_width", type=int, default=256)
    parser.add_argument("--use_moe", type=int, default=0)
    parser.add_argument("--bcd_feature_mode", default="pre")
    parser.add_argument("--decoder_head", default="dpt_residual", choices=["dpt_residual"])
    parser.add_argument("--head_channels", type=int, default=64)
    parser.add_argument("--head_init_scale", type=float, default=0.1)
    parser.add_argument("--head_zero_init", type=int, default=1)
    parser.add_argument("--head_mode", default="residual", choices=["residual", "direct"])
    parser.add_argument("--thresholds", default="0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70")
    parser.add_argument("--eval_tta", type=int, default=0)
    # BoN knobs
    parser.add_argument("--bon_k", type=int, default=8,
                        help="Number of candidates. K=1 = standard inference.")
    parser.add_argument("--bon_tau", type=float, default=0.7,
                        help="Sampling temperature for noise. Lower = candidates closer to det.")
    parser.add_argument("--bon_sigma", type=float, default=4.0,
                        help="Spatial noise sigma; match PRM training value when in doubt.")
    parser.add_argument("--bon_mode", default="hard",
                        choices=["hard", "soft", "per_pixel"])
    parser.add_argument("--bon_temperature", type=float, default=1.0,
                        help="Softmax temperature for soft mode.")
    parser.add_argument("--bon_use_prob", type=int, default=0,
                        help="Score continuous probs instead of binary samples (1 = on).")
    parser.add_argument("--bon_include_deterministic", type=int, default=1,
                        help="Always include the deterministic policy as a candidate. "
                             "Guarantees BoN never loses to plain inference.")
    parser.add_argument("--min_change_ratio", type=float, default=0.0)
    parser.add_argument("--backbone", default="change3d",
                        choices=["change3d", "dinov2", "sam2"])
    parser.add_argument("--dino_arch", default="vits14",
                        choices=["vits14", "vitb14", "vitl14"])
    parser.add_argument("--sam2_cfg", default="sam2.1_hiera_t")
    parser.add_argument("--sam2_ckpt", default="")
    parser.add_argument("--dino_input_size", type=int, default=0)
    parser.add_argument("--sam2_input_size", type=int, default=512)
    parser.add_argument("--decoder_channels", type=int, default=128)
    args = parser.parse_args()
    thresholds = [float(x) for x in args.thresholds.split(",") if x.strip()]

    add_change3d_root(args.change3d_root)
    loader = build_loader(args, args.split, train=False)
    model = build_change3d_model(args, args.ckpt, train=False)
    model.eval()
    prm, ext_prior = load_prm(args.prm_ckpt, args.device)
    noise_module = LowRankSpatialNoise(sigma=args.bon_sigma).to(args.device)

    print(f"[BoN] K={args.bon_k} tau={args.bon_tau} sigma={args.bon_sigma} "
          f"mode={args.bon_mode} use_prob={args.bon_use_prob} "
          f"include_det={args.bon_include_deterministic}")

    if 0.5 not in thresholds:
        thresholds = sorted(set(thresholds) | {0.5})
    scores = evaluate_bon(args, model, prm, ext_prior, noise_module, loader, thresholds)
    best_th = max(scores, key=lambda th: scores[th]["F1"])
    for th in thresholds:
        sc = scores[th]
        print(f"th={th:.2f} F1={sc['F1']:.4f} IoU={sc['IoU']:.4f} "
              f"P={sc['precision']:.4f} R={sc['recall']:.4f}")
    primary = scores[0.5]
    best = scores[best_th]
    print(f"[primary K={args.bon_k} th=0.50] F1={primary['F1']:.4f} "
          f"IoU={primary['IoU']:.4f} P={primary['precision']:.4f} "
          f"R={primary['recall']:.4f} Kappa={primary['Kappa']:.4f}")
    print(f"[oracle K={args.bon_k} th={best_th:.2f}] F1={best['F1']:.4f} "
          f"IoU={best['IoU']:.4f}  (sanity only, not for paper)")


if __name__ == "__main__":
    main()
