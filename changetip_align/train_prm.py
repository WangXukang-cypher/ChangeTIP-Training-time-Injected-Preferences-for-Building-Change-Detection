import argparse
import os

import torch
import torch.nn.functional as F

from .candidate import as_prob
from .data import build_loader, split_batch
from .imports import add_change3d_root
from .models import build_change3d_model
from .prm import (
    MultiStageProcessReward,
    iou_pair,
    perturb_mask,
    prm_discriminator_loss,
    prm_iou_calibration_loss,
    prm_monotonicity_loss,
)
from .sampling import LowRankSpatialNoise, sample_masks, stochastic_logits


def parse_floats(text):
    if not text:
        return []
    return [float(item.strip()) for item in text.split(",") if item.strip()]


@torch.no_grad()
def validate(args, model, prm, loader, noise_module):
    prm.eval()
    real_correct = 0.0
    fake_correct = 0.0
    total = 0
    iou_total = 0.0
    real_mean_sum = 0.0
    fake_mean_sum = 0.0
    for batch in loader:
        pre, post, target = split_batch(batch, args.device)
        out = model.update_bcd_full(pre, post)
        stages = out["stages"]
        r_real = prm(stages, target, pre, post)
        logits = stochastic_logits(
            out["base_logit"], out["delta_logit"], out["residual_scale"],
            noise_module, k=1, tau=args.sampling_tau,
        )
        policy_mask = sample_masks(logits)
        gt_perturbed = perturb_mask(target, p_flip=args.p_flip, mode="mix")
        b = pre.shape[0]
        use_policy = torch.rand(b, device=pre.device) < 0.5
        use_policy = use_policy.view(-1, 1, 1, 1).float()
        fake_mask = use_policy * policy_mask + (1.0 - use_policy) * gt_perturbed
        r_fake = prm(stages, fake_mask, pre, post)
        real_correct += (r_real > 0).float().mean().item() * b
        fake_correct += (r_fake < 0).float().mean().item() * b
        iou_total += iou_pair(fake_mask, target).mean().item() * b
        real_mean_sum += r_real.mean().item() * b
        fake_mean_sum += r_fake.mean().item() * b
        total += b
    prm.train()
    return {
        "real_acc": real_correct / max(1, total),
        "fake_acc": fake_correct / max(1, total),
        "fake_iou": iou_total / max(1, total),
        "real_logit": real_mean_sum / max(1, total),
        "fake_logit": fake_mean_sum / max(1, total),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--change3d_root", default="..")
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--ckpt", required=True, help="Change3D / ChangeTIP-Align checkpoint")
    parser.add_argument("--pretrained", default="")
    parser.add_argument("--save_path", required=True)
    parser.add_argument("--dataset", default="WHU-CD")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--in_height", type=int, default=256)
    parser.add_argument("--in_width", type=int, default=256)
    parser.add_argument("--use_moe", type=int, default=0)
    parser.add_argument("--bcd_feature_mode", default="pre")
    parser.add_argument("--decoder_head", default="dpt_residual", choices=["dpt_residual"])
    parser.add_argument("--head_channels", type=int, default=64)
    parser.add_argument("--head_init_scale", type=float, default=0.1)
    parser.add_argument("--prm_hidden", type=int, default=128)
    parser.add_argument("--sampling_sigma", type=float, default=4.0)
    parser.add_argument("--sampling_tau", type=float, default=1.0)
    parser.add_argument("--p_flip", type=float, default=0.05)
    parser.add_argument("--lambda_disc", type=float, default=1.0)
    parser.add_argument("--lambda_mono", type=float, default=0.2)
    parser.add_argument("--lambda_calib", type=float, default=0.5)
    args = parser.parse_args()

    add_change3d_root(args.change3d_root)
    train_loader = build_loader(args, "train", train=True)
    val_loader = build_loader(args, "val", train=False)

    model = build_change3d_model(args, args.ckpt, train=False)
    for param in model.parameters():
        param.requires_grad = False

    stage_channels = model.spatial_head.stage_channels
    prm = MultiStageProcessReward(stage_channels, hidden=args.prm_hidden).to(args.device)
    noise_module = LowRankSpatialNoise(sigma=args.sampling_sigma).to(args.device)

    optimizer = torch.optim.AdamW(prm.parameters(), lr=args.lr, weight_decay=1e-4)

    best = -1.0
    for epoch in range(1, args.epochs + 1):
        prm.train()
        running = 0.0
        count = 0
        for batch in train_loader:
            pre, post, target = split_batch(batch, args.device)
            with torch.no_grad():
                out = model.update_bcd_full(pre, post)
                stages = out["stages"]
                # build a "fake" mask: mix policy-sampled and GT-perturbed
                logits = stochastic_logits(
                    out["base_logit"], out["delta_logit"], out["residual_scale"],
                    noise_module, k=1, tau=args.sampling_tau,
                )
                policy_mask = sample_masks(logits)
                gt_perturbed = perturb_mask(target, p_flip=args.p_flip, mode="mix")
                # half/half within batch — both branches now produce IoU<0.7 fakes,
                # so the discriminator gets a clean negative signal everywhere.
                b = pre.shape[0]
                use_policy = torch.rand(b, device=pre.device) < 0.5
                use_policy = use_policy.view(-1, 1, 1, 1).float()
                fake_mask = use_policy * policy_mask + (1.0 - use_policy) * gt_perturbed
                fake_iou = iou_pair(fake_mask, target)

            r_real, per_stage_real = prm(stages, target, pre, post, return_stage=True)
            r_fake = prm(stages, fake_mask, pre, post)

            disc = prm_discriminator_loss(r_real, r_fake)
            mono = prm_monotonicity_loss(per_stage_real)
            calib = prm_iou_calibration_loss(r_fake.mean(dim=(1, 2, 3)), fake_iou)
            loss = (
                args.lambda_disc * disc
                + args.lambda_mono * mono
                + args.lambda_calib * calib
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(prm.parameters(), 5.0)
            optimizer.step()

            running += loss.item() * pre.shape[0]
            count += pre.shape[0]

        val = validate(args, model, prm, val_loader, noise_module)
        train_loss = running / max(1, count)
        score = 0.5 * (val["real_acc"] + val["fake_acc"])
        print(
            f"[epoch {epoch:03d}] train_loss={train_loss:.4f} "
            f"real_acc={val['real_acc']:.4f} fake_acc={val['fake_acc']:.4f} "
            f"fake_iou={val['fake_iou']:.4f} "
            f"r_real={val['real_logit']:+.3f} r_fake={val['fake_logit']:+.3f} "
            f"gap={val['real_logit'] - val['fake_logit']:+.3f}"
        )
        if score > best:
            best = score
            directory = os.path.dirname(args.save_path)
            if directory:
                os.makedirs(directory, exist_ok=True)
            torch.save(
                {
                    "state_dict": prm.state_dict(),
                    "stage_channels": list(stage_channels),
                    "args": vars(args),
                },
                args.save_path,
            )
            print(f"  saved PRM to {args.save_path}")


if __name__ == "__main__":
    main()
