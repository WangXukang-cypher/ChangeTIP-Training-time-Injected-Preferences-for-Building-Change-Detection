import argparse
import os

import torch

from .candidate import generate_candidates
from .data import build_loader, split_batch
from .imports import add_change3d_root
from .metrics import candidate_quality
from .models import build_change3d_model
from .verifier import ChangeVerifier, score_candidates, verifier_loss


def parse_floats(text):
    if not text:
        return []
    return [float(item.strip()) for item in text.split(",") if item.strip()]


def compute_quality(pre, post, masks, target):
    qualities = []
    for idx in range(masks.shape[1]):
        qualities.append(candidate_quality(pre, post, masks[:, idx], target))
    return torch.stack(qualities, dim=1).clamp(0.0, 1.0)


@torch.no_grad()
def validate(args, model, verifier, loader):
    verifier.eval()
    total_loss = 0.0
    total_count = 0
    top1_quality = 0.0
    oracle_quality = 0.0
    for batch in loader:
        pre, post, target = split_batch(batch, args.device)
        candidates = generate_candidates(
            model,
            pre,
            post,
            thresholds=args.thresholds,
            use_flips=args.use_flips,
            scales=args.scales,
        )
        target_score = compute_quality(pre, post, candidates.masks, target)
        pred_score = score_candidates(verifier, pre, post, candidates.masks)
        loss = verifier_loss(pred_score, target_score)
        top_idx = pred_score.argmax(dim=1)
        oracle_idx = target_score.argmax(dim=1)
        batch_idx = torch.arange(pre.shape[0], device=pre.device)
        top1_quality += target_score[batch_idx, top_idx].sum().item()
        oracle_quality += target_score[batch_idx, oracle_idx].sum().item()
        total_loss += loss.item() * pre.shape[0]
        total_count += pre.shape[0]
    verifier.train()
    return {
        "loss": total_loss / max(1, total_count),
        "top1_quality": top1_quality / max(1, total_count),
        "oracle_quality": oracle_quality / max(1, total_count),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--change3d_root", default="..")
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--pretrained", default="")
    parser.add_argument("--save_path", required=True)
    parser.add_argument("--dataset", default="WHU-CD")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--in_height", type=int, default=256)
    parser.add_argument("--in_width", type=int, default=256)
    parser.add_argument("--thresholds", default="0.30,0.40,0.50,0.60,0.70")
    parser.add_argument("--scales", default="")
    parser.add_argument("--use_flips", type=int, default=1)
    parser.add_argument("--use_moe", type=int, default=0)
    parser.add_argument("--bcd_feature_mode", default="pre")
    parser.add_argument("--decoder_head", default="baseline", choices=["baseline", "dpt_residual"])
    parser.add_argument("--head_channels", type=int, default=64)
    parser.add_argument("--head_init_scale", type=float, default=0.1)
    args = parser.parse_args()
    args.thresholds = parse_floats(args.thresholds)
    args.scales = parse_floats(args.scales)
    args.use_flips = bool(args.use_flips)

    add_change3d_root(args.change3d_root)
    train_loader = build_loader(args, "train", train=True)
    val_loader = build_loader(args, "val", train=False)

    generator = build_change3d_model(args, args.ckpt, train=False)
    for param in generator.parameters():
        param.requires_grad = False

    verifier = ChangeVerifier().to(args.device)
    optimizer = torch.optim.AdamW(verifier.parameters(), lr=args.lr, weight_decay=1e-4)

    best = -1.0
    for epoch in range(1, args.epochs + 1):
        verifier.train()
        running = 0.0
        count = 0
        for batch in train_loader:
            pre, post, target = split_batch(batch, args.device)
            with torch.no_grad():
                candidates = generate_candidates(
                    generator,
                    pre,
                    post,
                    thresholds=args.thresholds,
                    use_flips=args.use_flips,
                    scales=args.scales,
                )
                target_score = compute_quality(pre, post, candidates.masks, target)

            pred_score = score_candidates(verifier, pre, post, candidates.masks)
            loss = verifier_loss(pred_score, target_score)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(verifier.parameters(), 5.0)
            optimizer.step()

            running += loss.item() * pre.shape[0]
            count += pre.shape[0]

        val = validate(args, generator, verifier, val_loader)
        train_loss = running / max(1, count)
        print(
            f"[epoch {epoch:03d}] train_loss={train_loss:.4f} "
            f"val_loss={val['loss']:.4f} top1={val['top1_quality']:.4f} "
            f"oracle={val['oracle_quality']:.4f}"
        )
        if val["top1_quality"] > best:
            best = val["top1_quality"]
            directory = os.path.dirname(args.save_path)
            if directory:
                os.makedirs(directory, exist_ok=True)
            torch.save({"state_dict": verifier.state_dict(), "args": vars(args)}, args.save_path)
            print(f"  saved verifier to {args.save_path}")


if __name__ == "__main__":
    main()
