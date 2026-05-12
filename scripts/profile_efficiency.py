"""Profile FLOPs / Params / FPS for ChangeTIP backbone variants.

Run per-config and print a table row. Aggregate manually into T5.

Example:
    python scripts/profile_efficiency.py \
        --backbone dinov3 --dinov3_arch vitl16_sat --dinov3_input_size 512 \
        --change3d_root /fast/Wang/Change3D --device cuda --n_warmup 5 --n_iter 50
"""
import argparse
import os
import sys
import time
from contextlib import contextmanager

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch

from changetip_align.imports import add_change3d_root
from changetip_align.models import build_change3d_model


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def try_flops(model, pre, post):
    """Try fvcore -> thop -> ptflops -> None."""
    try:
        from fvcore.nn import FlopCountAnalysis
        wrapper = torch.nn.Module()
        wrapper.forward = lambda x, y: model.update_bcd(x, y)
        flops = FlopCountAnalysis(wrapper, (pre, post))
        flops.unsupported_ops_warnings(False)
        flops.uncalled_modules_warnings(False)
        return float(flops.total()), "fvcore"
    except Exception as e:
        print(f"[flops] fvcore failed: {e}", flush=True)
    try:
        from thop import profile
        with torch.no_grad():
            f, _ = profile(model, inputs=(pre, post), verbose=False)
        return float(f), "thop"
    except Exception as e:
        print(f"[flops] thop failed: {e}", flush=True)
    return None, "none"


@torch.no_grad()
def measure_fps(model, pre, post, n_warmup, n_iter, device):
    model.eval()
    for _ in range(n_warmup):
        _ = model.update_bcd(pre, post)
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iter):
        _ = model.update_bcd(pre, post)
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    fps = (n_iter * pre.shape[0]) / dt
    ms_per_img = 1000.0 / fps
    return fps, ms_per_img


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--change3d_root", default="..")
    p.add_argument("--dataset", default="CLCD")
    p.add_argument("--data_root", default="")
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--in_height", type=int, default=256)
    p.add_argument("--in_width", type=int, default=256)
    p.add_argument("--backbone", default="dinov3",
                   choices=["change3d", "dinov2", "dinov3", "sam2"])
    p.add_argument("--dino_arch", default="vits14")
    p.add_argument("--dinov3_arch", default="vitl16_sat")
    p.add_argument("--sam2_cfg", default="sam2.1_hiera_t")
    p.add_argument("--sam2_ckpt", default="")
    p.add_argument("--dino_input_size", type=int, default=0)
    p.add_argument("--dinov3_input_size", type=int, default=0)
    p.add_argument("--sam2_input_size", type=int, default=512)
    p.add_argument("--decoder_head", default="dpt_residual",
                   choices=["baseline", "dpt_residual"])
    p.add_argument("--head_channels", type=int, default=64)
    p.add_argument("--head_init_scale", type=float, default=0.1)
    p.add_argument("--head_zero_init", type=int, default=1)
    p.add_argument("--head_mode", default="residual", choices=["residual", "direct"])
    p.add_argument("--decoder_channels", type=int, default=128)
    p.add_argument("--fusion_mode", default="concat", choices=["concat", "frm"])
    p.add_argument("--bcd_feature_mode", default="pre")
    p.add_argument("--use_moe", type=int, default=0)
    p.add_argument("--pretrained", default="")
    p.add_argument("--n_warmup", type=int, default=5)
    p.add_argument("--n_iter", type=int, default=50)
    p.add_argument("--tag", default="", help="label printed in the summary row")
    args = p.parse_args()

    add_change3d_root(args.change3d_root)
    model = build_change3d_model(args, ckpt_path=None, train=False)
    model.eval()

    H, W = args.in_height, args.in_width
    pre = torch.randn(args.batch_size, 3, H, W, device=args.device)
    post = torch.randn(args.batch_size, 3, H, W, device=args.device)

    total, trainable = count_params(model)
    flops, flops_src = try_flops(model, pre[:1], post[:1])
    fps, ms = measure_fps(model, pre, post, args.n_warmup, args.n_iter, args.device)

    if torch.cuda.is_available() and args.device.startswith("cuda"):
        peak_mem = torch.cuda.max_memory_allocated() / (1024 ** 3)
    else:
        peak_mem = 0.0

    tag = args.tag or f"{args.backbone}-{args.dino_arch if args.backbone=='dinov2' else args.dinov3_arch if args.backbone=='dinov3' else args.sam2_cfg}"
    print("=" * 80)
    print(f"[CONFIG] tag={tag} backbone={args.backbone} input={H}x{W} bs={args.batch_size} fusion={args.fusion_mode}")
    print(f"[PARAMS] total={total/1e6:.2f}M  trainable={trainable/1e6:.2f}M")
    if flops is not None:
        print(f"[FLOPS]  {flops/1e9:.2f} GFLOPs (per image, bs=1, via {flops_src})")
    else:
        print(f"[FLOPS]  N/A (install fvcore or thop)")
    print(f"[SPEED]  {fps:.2f} img/s  ({ms:.2f} ms/img)  peak_mem={peak_mem:.2f} GB")
    print("=" * 80)


if __name__ == "__main__":
    main()
