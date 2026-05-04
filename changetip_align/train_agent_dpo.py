import argparse
import os

import torch

from .candidate import as_prob, gather_candidate, generate_candidates
from .data import build_loader, split_batch
from .imports import add_change3d_root
from .models import ModelEMA, build_change3d_model, save_model, set_trainable_policy
from .preference import (
    boundary_loss,
    cosine_lr,
    dpo_loss,
    false_positive_penalty,
    focus_region,
    kl_bernoulli,
    supervised_loss,
)
from .verifier import ChangeVerifier, score_candidates


def parse_floats(text):
    if not text:
        return []
    return [float(item.strip()) for item in text.split(",") if item.strip()]


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
def evaluate(args, model, loader, threshold=0.5):
    from utils.metric_tool import ConfuseMatrixMeter

    model.eval()
    meter = ConfuseMatrixMeter(n_class=2)
    for batch in loader:
        pre, post, target = split_batch(batch, args.device)
        prob = eval_prob(args, model, pre, post)
        pred = (prob > threshold).long().cpu().numpy()
        meter.update_cm(pr=pred, gt=target.cpu().numpy())
    model.train()
    return meter.get_scores()


def load_verifier(path, device):
    ckpt = torch.load(path, map_location="cpu")
    verifier = ChangeVerifier().to(device)
    state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    verifier.load_state_dict(state, strict=True)
    verifier.eval()
    for param in verifier.parameters():
        param.requires_grad = False
    return verifier


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--change3d_root", default="..")
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--verifier_ckpt", required=True)
    parser.add_argument("--pretrained", default="")
    parser.add_argument("--save_path", required=True)
    parser.add_argument("--dataset", default="WHU-CD")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--min_lr_factor", type=float, default=0.2)
    parser.add_argument("--ema_decay", type=float, default=0.999)
    parser.add_argument("--in_height", type=int, default=256)
    parser.add_argument("--in_width", type=int, default=256)
    parser.add_argument("--thresholds", default="0.30,0.40,0.50,0.60,0.70")
    parser.add_argument("--scales", default="")
    parser.add_argument("--use_flips", type=int, default=1)
    parser.add_argument("--use_moe", type=int, default=0)
    parser.add_argument("--bcd_feature_mode", default="pre")
    parser.add_argument("--decoder_head", default="dpt_residual", choices=["baseline", "dpt_residual"])
    parser.add_argument("--head_channels", type=int, default=64)
    parser.add_argument("--head_init_scale", type=float, default=0.1)
    parser.add_argument("--train_backbone_last", type=int, default=0)
    parser.add_argument("--train_base_decoder", type=int, default=0)
    parser.add_argument("--lambda_sup", type=float, default=1.0)
    parser.add_argument("--lambda_dpo", type=float, default=1.0)
    parser.add_argument("--lambda_kl", type=float, default=3e-3)
    parser.add_argument("--lambda_boundary", type=float, default=0.3)
    parser.add_argument("--lambda_fp", type=float, default=0.3)
    parser.add_argument("--beta_dpo", type=float, default=0.2)
    parser.add_argument("--focus_dilate", type=int, default=5)
    parser.add_argument("--eval_tta", type=int, default=0)
    parser.add_argument("--eval_split", default="val", choices=["val", "test"])
    args = parser.parse_args()
    args.thresholds = parse_floats(args.thresholds)
    args.scales = parse_floats(args.scales)
    args.use_flips = bool(args.use_flips)

    add_change3d_root(args.change3d_root)
    train_loader = build_loader(args, "train", train=True)
    val_loader = build_loader(args, "val", train=False)
    test_loader = build_loader(args, "test", train=False)
    eval_loader = val_loader if args.eval_split == "val" else test_loader

    policy = build_change3d_model(args, args.ckpt, train=True)
    ref = build_change3d_model(args, args.ckpt, train=False)
    for param in ref.parameters():
        param.requires_grad = False
    set_trainable_policy(
        policy,
        train_backbone_last=bool(args.train_backbone_last),
        train_base_decoder=bool(args.train_base_decoder),
    )
    verifier = load_verifier(args.verifier_ckpt, args.device)

    optimizer = torch.optim.AdamW(
        [p for p in policy.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=1e-4,
    )
    ema = ModelEMA(policy, decay=args.ema_decay)
    total_steps = max(1, len(train_loader) * args.epochs)
    global_step = 0
    best_f1 = -1.0

    before = evaluate(args, policy, eval_loader)
    print(f"[before] {args.eval_split} F1={before['F1']:.4f} IoU={before['IoU']:.4f}")

    for epoch in range(1, args.epochs + 1):
        policy.train()
        for batch in train_loader:
            global_step += 1
            lr = cosine_lr(
                global_step,
                total_steps,
                args.lr,
                warmup_steps=args.warmup_steps,
                min_lr_factor=args.min_lr_factor,
            )
            for group in optimizer.param_groups:
                group["lr"] = lr

            pre, post, target = split_batch(batch, args.device)
            with torch.no_grad():
                candidates = generate_candidates(
                    policy,
                    pre,
                    post,
                    thresholds=args.thresholds,
                    use_flips=args.use_flips,
                    scales=args.scales,
                )
                cand_score = score_candidates(verifier, pre, post, candidates.masks)
                chosen_idx = cand_score.argmax(dim=1)
                rejected_idx = cand_score.argmin(dim=1)
                chosen = gather_candidate(candidates.masks, chosen_idx)
                rejected = gather_candidate(candidates.masks, rejected_idx)

                ref_prob = as_prob(ref.update_bcd(pre, post))

            policy_prob = as_prob(policy.update_bcd(pre, post))
            focus = focus_region(chosen, rejected, target=target, dilate=args.focus_dilate)
            sup = supervised_loss(policy_prob, target)
            pref = dpo_loss(policy_prob, ref_prob, chosen, rejected, beta=args.beta_dpo, focus=focus)
            kl = kl_bernoulli(policy_prob, ref_prob, focus=focus)
            bnd = boundary_loss(policy_prob, target)
            fp = false_positive_penalty(policy_prob, target)
            loss = (
                args.lambda_sup * sup
                + args.lambda_dpo * pref
                + args.lambda_kl * kl
                + args.lambda_boundary * bnd
                + args.lambda_fp * fp
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 5.0)
            optimizer.step()
            ema.update(policy)

        ema.apply_shadow(policy)
        val = evaluate(args, policy, eval_loader)
        ema.restore(policy)
        print(
            f"[epoch {epoch:03d}] {args.eval_split} F1={val['F1']:.4f} "
            f"IoU={val['IoU']:.4f} P={val['precision']:.4f} R={val['recall']:.4f}"
        )
        if val["F1"] > best_f1:
            best_f1 = val["F1"]
            ema.apply_shadow(policy)
            save_model(args.save_path, policy)
            ema.restore(policy)
            print(f"  saved policy to {args.save_path}")

    policy.load_state_dict(torch.load(args.save_path, map_location=args.device), strict=True)
    test = evaluate(args, policy, test_loader)
    print(f"[test] F1={test['F1']:.4f} IoU={test['IoU']:.4f} P={test['precision']:.4f} R={test['recall']:.4f}")


if __name__ == "__main__":
    main()
