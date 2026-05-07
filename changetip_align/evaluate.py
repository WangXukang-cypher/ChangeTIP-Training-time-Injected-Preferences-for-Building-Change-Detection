import argparse

import numpy as np
import torch

from .candidate import as_prob
from .data import build_loader, split_batch
from .imports import add_change3d_root
from .models import build_change3d_model


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


if __name__ == "__main__":
    main()
