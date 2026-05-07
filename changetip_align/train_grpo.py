import argparse
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from .candidate import as_prob
from .data import build_loader, split_batch
from .external_prior import ExternalPrior
from .imports import add_change3d_root
from .models import ModelEMA, build_change3d_model, save_model, set_trainable_policy
from .preference import (
    boundary_loss,
    cosine_lr,
    direct_prm_loss,
    dpo_loss,
    false_positive_penalty,
    grpo_to_dpo_pair,
    pixel_grpo_loss,
    prm_gated_kl,
    supervised_loss,
    unweighted_kl,
)
from .prm import MultiStageProcessReward
from .sampling import (
    LowRankSpatialNoise,
    bernoulli_logprob,
    sample_masks,
    stochastic_logits,
)


def parse_floats(text):
    if not text:
        return []
    return [float(item.strip()) for item in text.split(",") if item.strip()]


def apply_variant(args):
    """Override individual flags based on the chosen ablation variant.

    The variant is the single source of truth: if you set --variant no_boundary
    the script forces lambda_boundary=0 even if you also passed a non-zero value
    on the command line. This keeps experiment configs reproducible from a
    single token in the run name.
    """
    v = args.variant
    if v == "full":
        return
    if v == "no_residual_head":
        # Head replaces base decoder rather than adding a residual on top of it.
        # Zero-init no longer makes sense in this regime → also disable it.
        args.head_mode = "direct"
        args.head_zero_init = 0
    elif v == "no_zero_init":
        args.head_zero_init = 0
    elif v == "use_dpo":
        args.lambda_grpo = 0.0
        args.lambda_dpo = 1.0
    elif v == "no_prm_gated_kl":
        args.use_prm_gated_kl = 0
    elif v == "no_boundary":
        args.lambda_boundary = 0.0
    elif v == "no_spatial_noise":
        args.sampling_sigma = 0.0
    else:
        raise ValueError(f"Unknown variant: {v}")
    print(f"[variant] {v} applied:"
          f" head_mode={args.head_mode} head_zero_init={args.head_zero_init}"
          f" lambda_grpo={args.lambda_grpo} lambda_dpo={args.lambda_dpo}"
          f" use_prm_gated_kl={args.use_prm_gated_kl}"
          f" lambda_boundary={args.lambda_boundary}"
          f" sampling_sigma={args.sampling_sigma}")


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--change3d_root", default="..")
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--prm_ckpt", required=True)
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
    parser.add_argument("--use_moe", type=int, default=0)
    parser.add_argument("--bcd_feature_mode", default="pre")
    parser.add_argument("--decoder_head", default="dpt_residual", choices=["dpt_residual"])
    parser.add_argument("--head_channels", type=int, default=64)
    parser.add_argument("--head_init_scale", type=float, default=0.1)
    parser.add_argument("--train_backbone_last", type=int, default=0)
    parser.add_argument("--train_base_decoder", type=int, default=0)
    # GRPO knobs
    parser.add_argument("--grpo_k", type=int, default=8, help="group size (K candidates per image)")
    parser.add_argument("--sampling_sigma", type=float, default=4.0)
    parser.add_argument("--sampling_tau_init", type=float, default=1.0)
    parser.add_argument("--sampling_tau_final", type=float, default=0.2)
    parser.add_argument("--clip_eps", type=float, default=0.2)
    parser.add_argument("--advantage_clip", type=float, default=3.0,
                        help="Per-pixel advantage clip. Lower = more stable.")
    parser.add_argument("--grpo_std_eps", type=float, default=0.05,
                        help="Floor for the per-pixel std used in group normalization. "
                             "1e-4 (old default) amplifies noise into spurious advantages "
                             "when K candidates are too similar; 0.05 is a safer floor.")
    parser.add_argument("--reward_squash", type=float, default=2.0,
                        help="If > 0, squash PRM rewards by tanh(R/s)*s before GRPO and "
                             "direct-PRM losses, to bound outlier signal. 0 disables.")
    parser.add_argument("--kl_gate_mode", default="uncertainty",
                        choices=["uncertainty", "negative_reward"],
                        help="'uncertainty' = anchor where PRM is unsure (recommended); "
                             "'negative_reward' = legacy (anchor where PRM disagrees).")
    parser.add_argument("--prm_direct_squash", type=float, default=2.0,
                        help="Tanh-squash for direct PRM loss. 0 disables.")
    # Loss weights
    parser.add_argument("--lambda_sup", type=float, default=1.0)
    parser.add_argument("--lambda_grpo", type=float, default=1.0)
    parser.add_argument("--lambda_prm_direct", type=float, default=0.0,
                        help="Direct PRM-as-loss term. Set to >0 to enable; "
                             "GRPO and direct can run together if both lambdas are >0.")
    parser.add_argument("--lambda_kl", type=float, default=3e-3)
    parser.add_argument("--lambda_boundary", type=float, default=0.3)
    parser.add_argument("--lambda_fp", type=float, default=0.3)
    parser.add_argument("--kl_temperature", type=float, default=1.0)
    # focus dilation derived from sampled masks (for GRPO loss)
    parser.add_argument("--grpo_focus_dilate", type=int, default=5)
    parser.add_argument("--eval_tta", type=int, default=0)
    parser.add_argument("--eval_split", default="test", choices=["val", "test"],
                        help="Split used for per-epoch best-ckpt selection. Default 'test' "
                             "matches the Change3D evaluation protocol so reported F1 is "
                             "directly comparable to that baseline.")
    parser.add_argument("--min_change_ratio", type=float, default=0.0,
                        help="Filter train samples whose change-mask ratio is below this "
                             "threshold (e.g. 0.02 = at least 2%% positive pixels). "
                             "0 disables filtering. Only applied to the train split.")
    # Ablation switches
    parser.add_argument("--variant", default="full",
                        choices=["full", "no_residual_head", "no_zero_init",
                                 "use_dpo", "no_prm_gated_kl", "no_boundary",
                                 "no_spatial_noise"],
                        help="Component-ablation variant. 'full' = ChangeTIP-Align baseline.")
    parser.add_argument("--head_zero_init", type=int, default=1,
                        help="Zero-init the residual head output conv (preserves base at init).")
    parser.add_argument("--head_mode", default="residual", choices=["residual", "direct"],
                        help="'residual' adds delta to base_logit; 'direct' replaces base.")
    parser.add_argument("--lambda_dpo", type=float, default=0.0,
                        help="Weight on the DPO surrogate (used by --variant use_dpo).")
    parser.add_argument("--dpo_beta", type=float, default=0.2)
    parser.add_argument("--use_prm_gated_kl", type=int, default=1,
                        help="If 0, replace prm_gated_kl with unweighted KL.")
    # Backbone selection
    parser.add_argument("--backbone", default="change3d",
                        choices=["change3d", "dinov2", "sam2"],
                        help="Which encoder to use as the frozen prior. "
                             "change3d = X3D-based (default), dinov2 = ViT-S/14, "
                             "sam2 = Hiera image encoder.")
    parser.add_argument("--dino_arch", default="vits14",
                        choices=["vits14", "vitb14", "vitl14"],
                        help="DINOv2 size when --backbone dinov2.")
    parser.add_argument("--sam2_cfg", default="sam2.1_hiera_t",
                        help="SAM2 config name when --backbone sam2.")
    parser.add_argument("--sam2_ckpt", default="",
                        help="Path to SAM2 .pt checkpoint when --backbone sam2.")
    parser.add_argument("--dino_input_size", type=int, default=0,
                        help="Override DINOv2 internal forward size. Must be a "
                             "multiple of 14. 0 = auto (largest multiple of 14 "
                             "that fits in input; e.g. 252 for 256-pixel input).")
    parser.add_argument("--sam2_input_size", type=int, default=512,
                        help="SAM2 internal forward size. Default 512 is a good "
                             "tradeoff for 256-pixel CD inputs; 1024 = native.")
    parser.add_argument("--decoder_channels", type=int, default=128,
                        help="Hidden channels for the alt-backbone ChangeDecoder.")
    args = parser.parse_args()
    apply_variant(args)

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

    prm, ext_prior = load_prm(args.prm_ckpt, args.device)
    noise_module = LowRankSpatialNoise(sigma=args.sampling_sigma).to(args.device)

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

            progress = min(1.0, global_step / total_steps)
            tau = args.sampling_tau_init + progress * (args.sampling_tau_final - args.sampling_tau_init)

            pre, post, target = split_batch(batch, args.device)

            # --- ROLLOUT (no grad): sample K candidates, score with PRM ---
            with torch.no_grad():
                out_old = policy.update_bcd_full(pre, post)
                base_logit_old = out_old["base_logit"]
                delta_logit_old = out_old["delta_logit"]
                scale_old = out_old["residual_scale"]
                stages_old = out_old["stages"]
                policy_logit_old = base_logit_old + scale_old * delta_logit_old  # [B,1,H,W]

                # K stochastic logits (used only for sampling); policy logp is computed
                # against the deterministic policy_logit (action-noise factorization).
                sampling_logits = stochastic_logits(
                    base_logit_old, delta_logit_old, scale_old,
                    noise_module, k=args.grpo_k, tau=tau,
                )  # [B, K, H, W]
                masks = sample_masks(sampling_logits)  # [B, K, H, W] in {0,1}

                # logp_old under deterministic policy
                pol_old_expand = policy_logit_old.expand(-1, args.grpo_k, -1, -1)
                logp_old = bernoulli_logprob(pol_old_expand, masks)

                # External prior features (frozen) — gives PRM access to
                # information independent of the change-detection training set.
                ext_feat = ext_prior(pre, post) if ext_prior is not None else None

                # PRM rewards per-pixel for each candidate
                masks_for_prm = masks.unsqueeze(2)  # [B, K, 1, H, W]
                rewards = prm.reward_for_candidates(
                    stages_old, masks_for_prm, pre, post, ext_feat=ext_feat,
                )  # [B, K, H, W]
                # Bound outlier rewards before whitening, otherwise a few
                # extreme pixels dominate the per-pixel std and kill signal
                # everywhere else.
                if args.reward_squash > 0.0:
                    s = float(args.reward_squash)
                    rewards = torch.tanh(rewards / s) * s

                # Spatial focus: pixels that disagree across the group
                disagreement = masks.std(dim=1, keepdim=True)  # [B, 1, H, W]
                gt_focus = (target > 0.5).float()
                focus = ((disagreement > 0).float() + gt_focus).clamp(max=1.0)
                if args.grpo_focus_dilate > 1:
                    d = args.grpo_focus_dilate + (1 - args.grpo_focus_dilate % 2)
                    focus = F.max_pool2d(focus, kernel_size=d, stride=1, padding=d // 2)

                # Reference policy logits (PRM-gated KL target)
                out_ref = ref.update_bcd_full(pre, post)
                ref_logit = (
                    out_ref["base_logit"]
                    + out_ref["residual_scale"] * out_ref["delta_logit"]
                )
                # PRM reward on the *current* deterministic policy mask (for KL gate)
                pol_mask_old = (torch.sigmoid(policy_logit_old) > 0.5).float()
                gate_reward = prm(stages_old, pol_mask_old, pre, post, ext_feat=ext_feat)

            # --- POLICY UPDATE (with grad) ---
            out = policy.update_bcd_full(pre, post)
            base_logit = out["base_logit"]
            delta_logit = out["delta_logit"]
            scale = out["residual_scale"]
            policy_logit = base_logit + scale * delta_logit  # [B, 1, H, W]
            policy_logit_k = policy_logit.expand(-1, args.grpo_k, -1, -1)
            logp_new = bernoulli_logprob(policy_logit_k, masks)

            if args.lambda_grpo > 0.0:
                grpo = pixel_grpo_loss(
                    logp_new, logp_old, rewards,
                    focus=focus, clip_eps=args.clip_eps,
                    std_eps=args.grpo_std_eps,
                    advantage_clip=args.advantage_clip,
                )
            else:
                grpo = policy_logit.new_zeros(())

            policy_prob = torch.sigmoid(policy_logit)
            sup = supervised_loss(policy_prob, target)
            if args.use_prm_gated_kl:
                kl = prm_gated_kl(
                    policy_logit, ref_logit, gate_reward,
                    temperature=args.kl_temperature,
                    gate_mode=args.kl_gate_mode,
                )
            else:
                kl = unweighted_kl(policy_logit, ref_logit)
            bnd = boundary_loss(policy_prob, target) if args.lambda_boundary > 0.0 \
                else policy_prob.new_zeros(())
            fp = false_positive_penalty(policy_prob, target)

            # DPO surrogate (used by --variant use_dpo). Picks best/worst from
            # the same K candidates GRPO uses, ranked by image-level PRM reward.
            if args.lambda_dpo > 0.0:
                with torch.no_grad():
                    chosen, rejected = grpo_to_dpo_pair(masks, rewards)
                    ref_prob = torch.sigmoid(ref_logit)
                dpo = dpo_loss(policy_prob, ref_prob, chosen, rejected, beta=args.dpo_beta)
            else:
                dpo = policy_prob.new_zeros(())

            # Direct PRM-as-loss term. Gradient flows through PRM (frozen
            # weights) into the policy. This bypasses GRPO's K-candidate
            # sampling and uses the PRM's full discriminative power directly.
            if args.lambda_prm_direct > 0.0:
                prm_direct = direct_prm_loss(
                    prm, out["stages"], policy_prob, pre, post,
                    ext_feat=ext_feat, focus=focus,
                    squash=args.prm_direct_squash,
                )
            else:
                prm_direct = policy_prob.new_zeros(())

            loss = (
                args.lambda_sup * sup
                + args.lambda_grpo * grpo
                + args.lambda_dpo * dpo
                + args.lambda_prm_direct * prm_direct
                + args.lambda_kl * kl
                + args.lambda_boundary * bnd
                + args.lambda_fp * fp
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 5.0)
            optimizer.step()
            ema.update(policy)

            if global_step % 50 == 0:
                with torch.no_grad():
                    r_mean = rewards.mean().item()
                    r_std_global = rewards.std().item()
                    # std across K candidates per-pixel — if this is tiny, GRPO
                    # has no signal and direct PRM should carry the load.
                    r_std_perpix = rewards.std(dim=1).mean().item()
                    r_max = rewards.max().item()
                    r_min = rewards.min().item()
                print(
                    f"[step {global_step:06d}] lr={lr:.2e} tau={tau:.2f} "
                    f"sup={sup.item():.4f} grpo={grpo.item():.4f} "
                    f"dpo={dpo.item():.4f} prm_d={prm_direct.item():+.4f} "
                    f"kl={kl.item():.4f} bnd={bnd.item():.4f} fp={fp.item():.4f} | "
                    f"R[mean={r_mean:+.2f} std_global={r_std_global:.2f} "
                    f"std_perpix={r_std_perpix:.3f} range=[{r_min:+.2f},{r_max:+.2f}]]"
                )

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
    print(
        f"[final/test] F1={test['F1']:.4f} IoU={test['IoU']:.4f} "
        f"P={test['precision']:.4f} R={test['recall']:.4f} "
        f"(best epoch was selected on '{args.eval_split}' split, "
        f"best_{args.eval_split}_F1={best_f1:.4f})"
    )


if __name__ == "__main__":
    main()
