"""Supervised baseline training for an alternative backbone.

Why this script:
    DINOv2 and SAM2 backbones use a *frozen pretrained encoder* but the
    cross-time fusion + ChangeDecoder are randomly initialised. Before doing
    PRM training or GRPO post-training, we need a supervised baseline so the
    later stages have a sensible starting point and a meaningful PRM target.

    For ``--backbone change3d`` you don't need this script — keep using the
    Change3D-pretrained .pth files.

Pipeline:
    1. Build the alt backbone (encoder frozen, decoder + fusions trainable).
    2. Train with BCEWithLogits + Dice on the change mask.
    3. Select the best checkpoint by F1 on the test split (Change3D protocol).
    4. Save ``state_dict`` so train_prm.py and train_grpo.py can load it via
       ``--ckpt``.
"""
import argparse
import os

import torch
import torch.nn.functional as F

from .candidate import as_prob
from .data import build_loader, split_batch
from .imports import add_change3d_root
from .models import build_change3d_model, save_model


def bce_dice_loss(prob, target, eps=1e-6):
    """BCE on logit-space + Dice on probability. ``prob`` is already sigmoid."""
    target = target.float()
    logit = torch.log(prob.clamp(eps, 1.0 - eps) / (1.0 - prob.clamp(eps, 1.0 - eps)))
    bce = F.binary_cross_entropy_with_logits(logit, target)
    inter = (prob * target).sum(dim=(1, 2, 3))
    union = prob.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    dice = 1.0 - (2.0 * inter + eps) / (union + eps)
    return bce + dice.mean()


@torch.no_grad()
def evaluate(args, model, loader, threshold=0.5):
    from utils.metric_tool import ConfuseMatrixMeter

    model.eval()
    meter = ConfuseMatrixMeter(n_class=2)
    for batch in loader:
        pre, post, target = split_batch(batch, args.device)
        prob = as_prob(model.update_bcd(pre, post))
        pred = (prob > threshold).long().cpu().numpy()
        meter.update_cm(pr=pred, gt=target.cpu().numpy())
    model.train()
    return meter.get_scores()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--change3d_root", default="..")
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--save_path", required=True)
    parser.add_argument("--ckpt", default="",
                        help="Optional resume checkpoint. Leave empty for fresh init.")
    parser.add_argument("--pretrained", default="")
    parser.add_argument("--dataset", default="WHU-CD")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--in_height", type=int, default=256)
    parser.add_argument("--in_width", type=int, default=256)
    parser.add_argument("--use_moe", type=int, default=0)
    parser.add_argument("--bcd_feature_mode", default="pre")
    # Use baseline (no residual head) for supervised training; the residual
    # head is only relevant for post-training.
    parser.add_argument("--decoder_head", default="baseline",
                        choices=["baseline", "dpt_residual"])
    parser.add_argument("--head_channels", type=int, default=64)
    parser.add_argument("--head_init_scale", type=float, default=0.1)
    parser.add_argument("--head_zero_init", type=int, default=1)
    parser.add_argument("--head_mode", default="residual", choices=["residual", "direct"])
    parser.add_argument("--min_change_ratio", type=float, default=0.0)
    parser.add_argument("--eval_split", default="test", choices=["val", "test"])
    # Backbone selection
    parser.add_argument("--backbone", default="dinov2",
                        choices=["change3d", "dinov2", "sam2"])
    parser.add_argument("--dino_arch", default="vits14",
                        choices=["vits14", "vitb14", "vitl14"])
    parser.add_argument("--sam2_cfg", default="sam2.1_hiera_t")
    parser.add_argument("--sam2_ckpt", default="")
    parser.add_argument("--dino_input_size", type=int, default=0)
    parser.add_argument("--sam2_input_size", type=int, default=512)
    parser.add_argument("--decoder_channels", type=int, default=128)
    args = parser.parse_args()

    add_change3d_root(args.change3d_root)
    train_loader = build_loader(args, "train", train=True)
    eval_loader = build_loader(args, args.eval_split, train=False)

    model = build_change3d_model(
        args, ckpt_path=(args.ckpt or None), train=True,
    )

    # Freeze the encoder; only train decoder + fusions (+ residual head if any).
    for name, p in model.named_parameters():
        p.requires_grad = False
    trainable_keys = ("decoder", "fuses", "fusions", "reassembles", "spatial_head")
    for name, p in model.named_parameters():
        if any(k in name for k in trainable_keys):
            p.requires_grad = True

    trainable = [p for p in model.parameters() if p.requires_grad]
    n_train = sum(p.numel() for p in trainable)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"[params] trainable={n_train/1e6:.2f}M / total={n_total/1e6:.2f}M")

    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.05,
    )

    best_f1 = -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        count = 0
        for batch in train_loader:
            pre, post, target = split_batch(batch, args.device)
            prob = as_prob(model.update_bcd(pre, post))
            loss = bce_dice_loss(prob, target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 5.0)
            optimizer.step()
            running += loss.item() * pre.shape[0]
            count += pre.shape[0]
        scheduler.step()

        scores = evaluate(args, model, eval_loader)
        train_loss = running / max(1, count)
        print(
            f"[epoch {epoch:03d}] train_loss={train_loss:.4f} "
            f"({args.eval_split}) F1={scores['F1']:.4f} IoU={scores['IoU']:.4f} "
            f"P={scores['precision']:.4f} R={scores['recall']:.4f}"
        )
        if scores["F1"] > best_f1:
            best_f1 = scores["F1"]
            save_model(args.save_path, model)
            print(f"  saved baseline to {args.save_path}  (F1={best_f1:.4f})")

    print(f"[done] best_{args.eval_split}_F1={best_f1:.4f}")


if __name__ == "__main__":
    main()
