import argparse

import numpy as np
import torch
import torch.nn.functional as F

from .candidate import as_prob
from .data import build_loader, split_batch
from .external_prior import ExternalPrior
from .imports import add_change3d_root
from .models import build_change3d_model
from .prm import MultiStageProcessReward


@torch.no_grad()
def eval_prob(args, model, pre, post):
    prob = as_prob(model.update_bcd(pre, post))
    if not args.eval_tta:
        return prob
    outputs = [prob]
    out_h = as_prob(model.update_bcd(torch.flip(pre, [3]), torch.flip(post, [3])))
    outputs.append(torch.flip(out_h, [3]))
    out_v = as_prob(model.update_bcd(torch.flip(pre, [2]), torch.flip(post, [2])))
    outputs.append(torch.flip(out_v, [2]))
    out_hv = as_prob(model.update_bcd(torch.flip(pre, [2, 3]), torch.flip(post, [2, 3])))
    outputs.append(torch.flip(out_hv, [2, 3]))
    return torch.stack(outputs, dim=0).mean(dim=0)


def load_msad_from_e2e_ckpt(ckpt_path: str, device: str):
    """Load the MSAD jointly trained by ``train_e2e.py`` from the same ckpt.

    Returns (msad, ext_prior) or (None, None) if the checkpoint is the legacy
    decoder-only state_dict (in which case self-verifier mode is unavailable).
    """
    blob = torch.load(ckpt_path, map_location="cpu")
    if not isinstance(blob, dict) or "msad_state" not in blob:
        return None, None
    stage_channels = blob.get("stage_channels", [64, 64, 64, 64, 64])
    ext_channels = blob.get("ext_channels", 0)
    ext_backbone = blob.get("external_backbone", "")
    msad = MultiStageProcessReward(
        stage_channels, ext_channels=ext_channels,
    ).to(device)
    msad.load_state_dict(blob["msad_state"], strict=True)
    msad.eval()
    for p in msad.parameters():
        p.requires_grad = False
    ext_prior = None
    if ext_channels > 0 and ext_backbone:
        ext_prior = ExternalPrior(backbone=ext_backbone).to(device)
        print(f"[ExternalPrior] loaded {ext_backbone} (out_channels={ext_channels})")
    print(f"[Self-Verifier] MSAD loaded from {ckpt_path} "
          f"(stage_channels={stage_channels}, ext_channels={ext_channels})")
    return msad, ext_prior


@torch.no_grad()
def eval_self_verifier(args, model, msad, ext_prior, loader, thresholds, sv_temperature: float = 1.0):
    """Self-Verifier-Guided Multi-Threshold Decoding.

    Standard evaluation thresholds the predicted probability at a fixed value
    (we report 0.5 as the Change3D-aligned primary metric). With a jointly
    trained MSAD we can do better at inference time *without* changing the
    decoder: at each pixel, we ask MSAD to score the binary mask produced by
    each candidate threshold, then take the per-pixel argmax of MSAD reward.

    This is the inference-time analogue of GCR (Group-Contrastive Refinement)
    but with the K candidates being the deterministic threshold-binarised
    predictions, not stochastic samples — which is critical: with strong
    foundation backbones, stochastic candidates are too uniform to differ.
    Threshold-binarised candidates differ exactly at uncertain pixels, and
    that's where MSAD's verdict matters.

    Returns: dict mapping a synthetic key 'self_verifier' to score dict.
    """
    from utils.metric_tool import ConfuseMatrixMeter

    meter = ConfuseMatrixMeter(n_class=2)
    model.eval()
    for batch in loader:
        pre, post, target = split_batch(batch, args.device)
        out = model.update_bcd_full(pre, post)
        prob = out["final_prob"]
        stages = out["stages"]
        ext_feat = ext_prior(pre, post) if ext_prior is not None else None

        # Binary masks at every threshold + their MSAD reward maps.
        cand_masks = []
        cand_rewards = []
        for th in thresholds:
            mask = (prob > th).float()
            r = msad(stages, mask, pre, post, ext_feat=ext_feat)  # [B, 1, H, W]
            cand_masks.append(mask)
            cand_rewards.append(r)
        cand_masks = torch.stack(cand_masks, dim=1)      # [B, T, 1, H, W]
        cand_rewards = torch.stack(cand_rewards, dim=1)  # [B, T, 1, H, W]

        # Per-pixel softmax over thresholds → weighted-vote final mask.
        weights = F.softmax(cand_rewards / sv_temperature, dim=1)  # [B, T, 1, H, W]
        soft_pred = (cand_masks * weights).sum(dim=1)  # [B, 1, H, W]
        # Re-binarise at 0.5 (since the weights sum to 1 and masks are 0/1,
        # 0.5 is the "majority-of-mass" threshold).
        pred = (soft_pred > 0.5).long().cpu().numpy()
        meter.update_cm(pr=pred, gt=target.cpu().numpy())
    return meter.get_scores()


@torch.no_grad()
def evaluate(args, model, loader, thresholds):
    from utils.metric_tool import ConfuseMatrixMeter

    meters = {th: ConfuseMatrixMeter(n_class=2) for th in thresholds}
    model.eval()
    for batch in loader:
        pre, post, target = split_batch(batch, args.device)
        prob = eval_prob(args, model, pre, post)
        for th, meter in meters.items():
            pred = (prob > th).long().cpu().numpy()
            meter.update_cm(pr=pred, gt=target.cpu().numpy())
    return {th: meter.get_scores() for th, meter in meters.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--change3d_root", default="..")
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--pretrained", default="")
    parser.add_argument("--dataset", default="WHU-CD")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--in_height", type=int, default=256)
    parser.add_argument("--in_width", type=int, default=256)
    parser.add_argument("--thresholds", default="0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70")
    parser.add_argument("--use_moe", type=int, default=0)
    parser.add_argument("--bcd_feature_mode", default="pre")
    parser.add_argument("--decoder_head", default="dpt_residual", choices=["baseline", "dpt_residual"])
    parser.add_argument("--head_channels", type=int, default=64)
    parser.add_argument("--head_init_scale", type=float, default=0.1)
    parser.add_argument("--head_zero_init", type=int, default=1,
                        help="Match the value used at training time.")
    parser.add_argument("--head_mode", default="residual", choices=["residual", "direct"],
                        help="Match the value used at training time.")
    parser.add_argument("--eval_tta", type=int, default=0)
    parser.add_argument("--backbone", default="change3d",
                        choices=["change3d", "dinov2", "dinov3", "sam2"])
    parser.add_argument("--dino_arch", default="vits14",
                        choices=["vits14", "vitb14", "vitl14"])
    parser.add_argument("--dinov3_arch", default="vits16plus",
                        choices=["vits16", "vits16plus", "vitb16", "vitl16", "vitl16_sat"])
    parser.add_argument("--sam2_cfg", default="sam2.1_hiera_t")
    parser.add_argument("--sam2_ckpt", default="")
    parser.add_argument("--dino_input_size", type=int, default=0)
    parser.add_argument("--dinov3_input_size", type=int, default=0)
    parser.add_argument("--sam2_input_size", type=int, default=512)
    parser.add_argument("--decoder_channels", type=int, default=128)
    parser.add_argument("--fusion_mode", default="concat", choices=["concat", "frm"])
    parser.add_argument("--use_self_verifier", type=int, default=0,
                        help="If 1 and the checkpoint contains a jointly trained "
                             "MSAD (from train_e2e.py), report a [self_verifier] "
                             "row that uses MSAD-weighted multi-threshold decoding.")
    parser.add_argument("--sv_thresholds", default="0.30,0.40,0.50,0.60,0.70",
                        help="Threshold candidates used by self-verifier decoding. "
                             "Should bracket 0.5; finer = better but slower.")
    parser.add_argument("--sv_temperature", type=float, default=1.0,
                        help="Softmax temperature over MSAD scores. Lower = "
                             "more aggressive winner-takes-all.")
    args = parser.parse_args()
    thresholds = [float(x) for x in args.thresholds.split(",") if x.strip()]

    add_change3d_root(args.change3d_root)
    loader = build_loader(args, args.split, train=False)
    model = build_change3d_model(args, args.ckpt, train=False)
    # Always include 0.50 — this is the Change3D-aligned reporting threshold.
    if 0.5 not in thresholds:
        thresholds = sorted(set(thresholds) | {0.5})
    scores = evaluate(args, model, loader, thresholds)
    best_th = max(scores, key=lambda th: scores[th]["F1"])
    for th in thresholds:
        sc = scores[th]
        print(f"th={th:.2f} F1={sc['F1']:.4f} IoU={sc['IoU']:.4f} P={sc['precision']:.4f} R={sc['recall']:.4f}")
    primary = scores[0.5]
    best = scores[best_th]
    # [primary] is the official metric for paper reporting (matches Change3D's
    # fixed-threshold protocol). [oracle] sweeps thresholds on the test set
    # itself and is reported only as a sanity bound — DO NOT cite in papers.
    print(f"[primary th=0.50] F1={primary['F1']:.4f} IoU={primary['IoU']:.4f} "
          f"P={primary['precision']:.4f} R={primary['recall']:.4f} "
          f"Kappa={primary['Kappa']:.4f} OA={primary['OA']:.4f}")
    print(f"[oracle th={best_th:.2f}] F1={best['F1']:.4f} IoU={best['IoU']:.4f} "
          f"(sanity only, not for paper)")

    # Optional: Self-Verifier-Guided Multi-Threshold Decoding.
    if args.use_self_verifier:
        msad, ext_prior = load_msad_from_e2e_ckpt(args.ckpt, args.device)
        if msad is None:
            print("[self_verifier] requested but no msad_state found in ckpt — "
                  "did you train with train_e2e.py? Skipping.")
        else:
            sv_thresholds = [float(x) for x in args.sv_thresholds.split(",") if x.strip()]
            sv_scores = eval_self_verifier(
                args, model, msad, ext_prior, loader, sv_thresholds,
                sv_temperature=args.sv_temperature,
            )
            print(f"[self_verifier] F1={sv_scores['F1']:.4f} "
                  f"IoU={sv_scores['IoU']:.4f} "
                  f"P={sv_scores['precision']:.4f} R={sv_scores['recall']:.4f} "
                  f"Kappa={sv_scores['Kappa']:.4f}")
            print(f"[self_verifier-vs-primary] ΔF1={sv_scores['F1'] - primary['F1']:+.4f}")


if __name__ == "__main__":
    main()
