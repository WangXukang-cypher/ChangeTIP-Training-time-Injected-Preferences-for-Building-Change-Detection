"""MSAD diagnostics: does MSAD actually rank masks by IoU?

For each test sample, generate K candidate masks by binarising the predicted
prob at multiple thresholds (the same setup SVGD uses at inference). For each
candidate, compute (a) MSAD scalar reward = mean reward map, (b) true IoU vs GT.
Aggregate (reward, iou) pairs across the test set and report:

  - Spearman / Pearson correlation (per-sample, then averaged; and pooled).
  - Top-1 agreement: fraction of samples where argmax-by-reward == argmax-by-iou.
  - Per-threshold mean reward and mean IoU (sanity).

Run on a CLCD Full ckpt (must contain msad_state):

    python scripts/msad_diagnostics.py \
        --ckpt T2/A5.pth --dataset CLCD --data_root /fast/Wang/BCD_datasets/CLCD \
        --change3d_root /fast/Wang/Change3D --device cuda \
        --thresholds 0.30,0.40,0.50,0.60,0.70 \
        --backbone dinov3 --dinov3_arch vitl16_sat --dinov3_input_size 512
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch

from changetip_align.data import build_loader, split_batch
from changetip_align.evaluate import load_msad_from_e2e_ckpt
from changetip_align.imports import add_change3d_root
from changetip_align.models import build_change3d_model


def spearman(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if len(x) < 2:
        return float("nan")
    rx = np.argsort(np.argsort(x))
    ry = np.argsort(np.argsort(y))
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denom = np.sqrt((rx ** 2).sum() * (ry ** 2).sum())
    return float((rx * ry).sum() / denom) if denom > 0 else float("nan")


def pearson(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if len(x) < 2:
        return float("nan")
    x = x - x.mean()
    y = y - y.mean()
    denom = np.sqrt((x ** 2).sum() * (y ** 2).sum())
    return float((x * y).sum() / denom) if denom > 0 else float("nan")


def iou_single(pred, gt):
    pred = (pred > 0.5).float()
    gt = (gt > 0.5).float()
    inter = (pred * gt).sum().item()
    union = (pred + gt - pred * gt).sum().item()
    return inter / (union + 1e-6)


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--change3d_root", default="..")
    p.add_argument("--data_root", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--dataset", default="CLCD")
    p.add_argument("--split", default="test")
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--in_height", type=int, default=256)
    p.add_argument("--in_width", type=int, default=256)
    p.add_argument("--thresholds", default="0.30,0.40,0.50,0.60,0.70")
    p.add_argument("--backbone", default="dinov3")
    p.add_argument("--dino_arch", default="vits14")
    p.add_argument("--dinov3_arch", default="vitl16_sat")
    p.add_argument("--sam2_cfg", default="sam2.1_hiera_t")
    p.add_argument("--sam2_ckpt", default="")
    p.add_argument("--dino_input_size", type=int, default=0)
    p.add_argument("--dinov3_input_size", type=int, default=0)
    p.add_argument("--sam2_input_size", type=int, default=512)
    p.add_argument("--decoder_head", default="dpt_residual")
    p.add_argument("--head_channels", type=int, default=64)
    p.add_argument("--head_init_scale", type=float, default=0.1)
    p.add_argument("--head_zero_init", type=int, default=1)
    p.add_argument("--head_mode", default="residual")
    p.add_argument("--decoder_channels", type=int, default=128)
    p.add_argument("--fusion_mode", default="concat")
    p.add_argument("--bcd_feature_mode", default="pre")
    p.add_argument("--use_moe", type=int, default=0)
    p.add_argument("--pretrained", default="")
    p.add_argument("--out_csv", default="", help="optional path to dump per-pair (sample,th,iou,reward) csv")
    args = p.parse_args()

    add_change3d_root(args.change3d_root)
    thresholds = [float(x) for x in args.thresholds.split(",") if x.strip()]
    K = len(thresholds)

    loader = build_loader(args, args.split, train=False)
    model = build_change3d_model(args, args.ckpt, train=False)
    model.eval()
    msad, ext_prior = load_msad_from_e2e_ckpt(args.ckpt, args.device)
    if msad is None:
        raise SystemExit(f"[FATAL] {args.ckpt} has no msad_state — cannot diagnose. "
                         "Use a ckpt trained with MSAD active (lambda_disc>0).")

    # Per-sample collectors.
    sample_spearman = []
    sample_pearson = []
    top1_agree = 0
    n_samples = 0
    pooled_iou = []
    pooled_reward = []
    per_th_iou = [[] for _ in thresholds]
    per_th_reward = [[] for _ in thresholds]
    rows = []

    for batch in loader:
        pre, post, target = split_batch(batch, args.device)
        out = model.update_bcd_full(pre, post)
        prob = out["final_prob"]
        stages = out["stages"]
        ext_feat = ext_prior(pre, post) if ext_prior is not None else None

        B = pre.shape[0]
        # [B, K] reward and iou.
        rewards_bk = torch.zeros(B, K, device=args.device)
        ious_bk = torch.zeros(B, K, device=args.device)
        for ki, th in enumerate(thresholds):
            mask = (prob > th).float()
            r_map = msad(stages, mask, pre, post, ext_feat=ext_feat)  # [B,1,H,W]
            rewards_bk[:, ki] = r_map.flatten(1).mean(dim=1)
            # per-sample IoU
            for bi in range(B):
                ious_bk[bi, ki] = iou_single(mask[bi], target[bi])

        rewards_np = rewards_bk.cpu().numpy()
        ious_np = ious_bk.cpu().numpy()

        for bi in range(B):
            r = rewards_np[bi]; i = ious_np[bi]
            sample_spearman.append(spearman(r, i))
            sample_pearson.append(pearson(r, i))
            top1_agree += int(np.argmax(r) == np.argmax(i))
            n_samples += 1
            pooled_iou.extend(i.tolist())
            pooled_reward.extend(r.tolist())
            for ki in range(K):
                per_th_iou[ki].append(float(i[ki]))
                per_th_reward[ki].append(float(r[ki]))
                rows.append((n_samples - 1, thresholds[ki], float(i[ki]), float(r[ki])))

    sp_per = np.nanmean(sample_spearman)
    pe_per = np.nanmean(sample_pearson)
    sp_pool = spearman(pooled_reward, pooled_iou)
    pe_pool = pearson(pooled_reward, pooled_iou)
    agree = top1_agree / max(1, n_samples)

    print("=" * 80)
    print(f"[MSAD Diagnostics] ckpt={args.ckpt}  N={n_samples}  K={K}  thresholds={thresholds}")
    print("=" * 80)
    print(f"  per-sample Spearman ρ (mean): {sp_per:+.4f}")
    print(f"  per-sample Pearson  r (mean): {pe_per:+.4f}")
    print(f"  pooled    Spearman ρ        : {sp_pool:+.4f}")
    print(f"  pooled    Pearson  r        : {pe_pool:+.4f}")
    print(f"  top-1 agreement (argmax_r == argmax_iou): {agree:.4f}")
    print(f"  random-chance top-1                    : {1.0/K:.4f}")
    print("-" * 80)
    print(f"  {'th':>6}  {'mean_iou':>10}  {'mean_reward':>12}")
    for ki, th in enumerate(thresholds):
        print(f"  {th:>6.2f}  {np.mean(per_th_iou[ki]):>10.4f}  {np.mean(per_th_reward[ki]):>+12.4f}")
    print("=" * 80)
    print("[interp] ρ > 0.5 = MSAD ranks candidates well; ρ near 0 = MSAD is noise.")
    print("[interp] If pooled ρ >> per-sample ρ, MSAD only learns global scale, not "
          "per-sample discrimination — which is what GCR is supposed to fix.")

    if args.out_csv:
        with open(args.out_csv, "w") as f:
            f.write("sample,threshold,iou,reward\n")
            for row in rows:
                f.write(f"{row[0]},{row[1]},{row[2]:.6f},{row[3]:.6f}\n")
        print(f"[saved] per-pair csv → {args.out_csv}")


if __name__ == "__main__":
    main()
