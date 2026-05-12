"""End-to-end training of ChangeTIP from scratch with reward-inspired components.

This is the *only* training script in the project — there is no separate
``train_prm.py`` / ``train_grpo.py`` post-training stage. The script starts
from a frozen pretrained backbone (e.g. DINOv3-SAT) and trains the decoder
plus auxiliary components in a single optimisation:

  * Supervised loss (BCE + Dice) on the final probability map.
  * Multi-Stage Auxiliary Discriminator (MSAD), jointly trained — the same
    network architecture as the previous PRM, repurposed as an auxiliary
    discriminator that scores (real_target vs perturbed_target) at every
    decoder stage. Provides a learned structural-correctness signal.
  * Direct MSAD path: at every step, MSAD is momentarily frozen and the
    decoder receives a smooth gradient through MSAD's reward of the policy
    output. This distils MSAD's preference into the decoder weights.
  * Pixel-GRPO: at every step, K stochastic-noise candidate masks are
    sampled from the policy logits; MSAD (frozen here) scores each candidate
    pixel-wise; advantages are normalised group-relatively across the K
    candidates per pixel; a PPO-style score-function term updates the
    decoder. This is the LLM-side GRPO objective transposed to dense
    prediction, fused into end-to-end training rather than run as a
    separate post-training stage.

All losses are summed and back-propagated together. Frozen backbone is
default; ``--train_backbone_last`` / ``--train_base_decoder`` can selectively
unfreeze parts. Set any ``--lambda_*`` to 0 to ablate that component.

Two-phase curriculum (controlled by ``--reward_warmup_ratio``, default 0.7):

  * **Phase 1 — Warmup (first 70% of epochs)**: The training is *identical*
    to a plain supervised baseline. ``spatial_head`` is frozen; MSAD is not
    trained; only the base decoder receives gradient via BCE+Dice. This
    lets DINOv3 + base decoder converge to a strong supervised solution
    before any reward signal is introduced.

  * **Phase 2 — Reward-active (final 30% of epochs)**: ``spatial_head`` is
    unfrozen, MSAD starts training (disc + mono + calib), and the decoder
    receives additional gradient via the Direct path and Pixel-GRPO. This
    matches the "post-training on a strong base" paradigm without the
    multi-stage training pipeline.
"""
import argparse
import os
from contextlib import contextmanager

import torch
import torch.nn.functional as F

from .candidate import as_prob
from .data import build_loader, split_batch
from .external_prior import ExternalPrior
from .imports import add_change3d_root
from .models import (
    ModelEMA,
    build_change3d_model,
    save_model,
    set_trainable_policy,
)
from .preference import (
    boundary_loss,
    cosine_lr,
    false_positive_penalty,
    pixel_grpo_loss,
    supervised_loss,
)
from .prm import (
    MultiStageProcessReward,
    perturb_mask,
    prm_discriminator_loss,
    prm_iou_calibration_loss,
    prm_monotonicity_loss,
)
from .sampling import (
    LowRankSpatialNoise,
    bernoulli_logprob,
    sample_masks,
    stochastic_logits,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def bce_dice_loss(prob, target, eps=1e-6, pos_weight: float = 1.0):
    """BCE on logit-space + Dice on probability. ``prob`` is already sigmoid."""
    target = target.float()
    logit = torch.log(prob.clamp(eps, 1.0 - eps) / (1.0 - prob.clamp(eps, 1.0 - eps)))
    if pos_weight != 1.0:
        pw = logit.new_tensor(pos_weight)
        bce = F.binary_cross_entropy_with_logits(logit, target, pos_weight=pw)
    else:
        bce = F.binary_cross_entropy_with_logits(logit, target)
    inter = (prob * target).sum(dim=(1, 2, 3))
    union = prob.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    dice = 1.0 - (2.0 * inter + eps) / (union + eps)
    return bce + dice.mean()


@contextmanager
def freeze_module(module):
    """Temporarily set requires_grad=False on every parameter in ``module``."""
    saved = [p.requires_grad for p in module.parameters()]
    for p in module.parameters():
        p.requires_grad_(False)
    try:
        yield
    finally:
        for p, s in zip(module.parameters(), saved):
            p.requires_grad_(s)


def iou_per_candidate(masks_k: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Compute IoU per (B, K) candidate vs target.

    masks_k: [B, K, 1, H, W] continuous-valued; thresholded at 0.5 internally.
    target:  [B, 1, H, W].
    Returns: [B, K] IoU.
    """
    bin_pred = (masks_k > 0.5).float()
    bin_targ = (target.unsqueeze(1) > 0.5).float()
    inter = (bin_pred * bin_targ).sum(dim=(2, 3, 4))
    union = bin_pred.sum(dim=(2, 3, 4)) + bin_targ.sum(dim=(2, 3, 4)) - inter
    return inter / (union + 1e-6)


@torch.no_grad()
def evaluate(args, model, loader, threshold: float = 0.5):
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


# ---------------------------------------------------------------------------
# CLI + main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="End-to-end training of ChangeTIP. Single-stage, frozen "
                    "foundation backbone, with reward-inspired decoder components."
    )
    # I/O
    parser.add_argument("--change3d_root", default="..")
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--save_path", required=True)
    parser.add_argument("--ckpt", default="",
                        help="Optional resume checkpoint. Leave empty for fresh init.")
    parser.add_argument("--pretrained", default="")
    parser.add_argument("--dataset", default="WHU-CD")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--min_lr_factor", type=float, default=0.2)
    parser.add_argument("--ema_decay", type=float, default=0.999)
    parser.add_argument("--in_height", type=int, default=256)
    parser.add_argument("--in_width", type=int, default=256)
    parser.add_argument("--use_moe", type=int, default=0)
    parser.add_argument("--bcd_feature_mode", default="pre")
    # The default is now dpt_residual because MSAD's direct path needs the
    # full forward (base_logit + delta_logit + stages). Setting to "baseline"
    # disables the residual head and forces lambda_direct/lambda_gcr to be
    # interpreted on the base decoder output instead — see code below.
    parser.add_argument("--decoder_head", default="dpt_residual",
                        choices=["baseline", "dpt_residual"])
    parser.add_argument("--head_channels", type=int, default=64)
    parser.add_argument("--head_init_scale", type=float, default=0.1)
    parser.add_argument("--head_zero_init", type=int, default=1)
    parser.add_argument("--head_mode", default="residual", choices=["residual", "direct"])
    parser.add_argument("--train_backbone_last", type=int, default=0)
    parser.add_argument("--train_base_decoder", type=int, default=1)
    parser.add_argument("--min_change_ratio", type=float, default=0.0)
    parser.add_argument("--eval_split", default="test", choices=["val", "test"])
    # Backbone selection
    parser.add_argument("--backbone", default="dinov3",
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
    # Temporal fusion module ('concat' = CrossTimeFusion, 'frm' = FRMFusion)
    parser.add_argument("--fusion_mode", default="concat", choices=["concat", "frm"],
                        help="Cross-temporal fusion module. 'concat' (default) uses the "
                             "simple concat+1×1+ResUnit baseline. 'frm' uses the Feature "
                             "Rectify Module with learned cross-temporal channel+spatial "
                             "attention (higher capacity, ~10%% more params in decoder).")
    # Class-imbalance weighting for BCE (pos_weight > 1 upweights change pixels)
    parser.add_argument("--pos_weight", type=float, default=1.0,
                        help="BCE positive-class weight. 1.0 = equal (default). "
                             "WHU-CD: 3.0 recommended (change pixels are rare). "
                             "LEVIR: 1.0 is fine.")
    # MSAD (Multi-Stage Auxiliary Discriminator — formerly PRM)
    parser.add_argument("--msad_hidden", type=int, default=128)
    parser.add_argument("--use_external_prior", type=int, default=1,
                        help="External (frozen) prior for MSAD. Strongly recommended.")
    parser.add_argument("--external_backbone", default="dinov2_vits14",
                        choices=["resnet18", "dinov2_vits14"])
    parser.add_argument("--p_flip", type=float, default=0.05)
    # Pixel-GRPO (K stochastic candidates, MSAD-as-reward)
    parser.add_argument("--grpo_k", type=int, default=4,
                        help="Number of stochastic candidates per step. K=1 disables GRPO.")
    parser.add_argument("--grpo_sigma", type=float, default=2.0,
                        help="Spatial-noise sigma for candidate generation.")
    parser.add_argument("--grpo_tau", type=float, default=0.7,
                        help="Sampling temperature multiplier on the noise.")
    parser.add_argument("--grpo_clip_eps", type=float, default=0.2,
                        help="PPO trust-region clip. Inactive at single-epoch on-policy.")
    parser.add_argument("--grpo_std_eps", type=float, default=0.05,
                        help="Floor for the per-pixel std used in advantage normalisation.")
    parser.add_argument("--grpo_advantage_clip", type=float, default=3.0,
                        help="Clip the normalised advantage to ±A. Reduces variance.")
    parser.add_argument("--reward_squash", type=float, default=2.0,
                        help="If >0, squash MSAD rewards by tanh(R/s)*s before GRPO.")
    # Loss weights — set any to 0 to ablate that component
    parser.add_argument("--lambda_sup", type=float, default=1.0)
    parser.add_argument("--lambda_disc", type=float, default=1.0,
                        help="MSAD discriminator loss (real-vs-perturbed). 0 disables MSAD entirely.")
    parser.add_argument("--lambda_mono", type=float, default=0.2)
    parser.add_argument("--lambda_calib", type=float, default=0.5,
                        help="MSAD-IoU calibration over the K GRPO candidates.")
    parser.add_argument("--lambda_direct", type=float, default=0.1,
                        help="Direct MSAD-to-decoder gradient (frozen MSAD).")
    parser.add_argument("--lambda_grpo", type=float, default=0.5,
                        help="Pixel-GRPO score-function loss across K candidates.")
    parser.add_argument("--lambda_boundary", type=float, default=0.0)
    parser.add_argument("--lambda_fp", type=float, default=0.0)
    parser.add_argument("--direct_squash", type=float, default=2.0)
    # Reward-warmup: don't apply Direct + GRPO until the model has trained
    # supervised + MSAD-discriminator long enough to be useful. Cold-starting
    # GRPO is destructive — MSAD gives noise, K candidates are nearly
    # identical, and the score-function gradient just adds variance to a
    # randomly initialised decoder. Default kicks reward losses in at 70%
    # of total epochs, when supervised has converged and MSAD has gap > 1.
    parser.add_argument("--reward_warmup_ratio", type=float, default=0.7,
                        help="Fraction of total epochs to train with supervised "
                             "+ MSAD-discriminator only. After this fraction, "
                             "Direct and GRPO losses are added. Default 0.7. "
                             "Set 0 to apply rewards from epoch 1 (legacy). "
                             "Set 1 to disable rewards entirely.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Global random seed for reproducibility across runs.")
    args = parser.parse_args()

    import random
    random.seed(args.seed)
    import numpy as np
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    msad_enabled = args.lambda_disc > 0 or args.lambda_direct > 0 or args.lambda_grpo > 0
    using_residual = (args.decoder_head == "dpt_residual")
    if msad_enabled and not using_residual:
        raise ValueError(
            "MSAD components require --decoder_head dpt_residual (need access "
            "to base_logit + delta_logit + stages). Either set "
            "--decoder_head dpt_residual or set lambda_disc=lambda_direct="
            "lambda_grpo=0 to disable MSAD entirely."
        )

    # ----- Data + model -----
    # Two train loaders: ``warmup`` uses the full set (no min_change_ratio
    # filtering, so the supervised stage matches a vanilla CD baseline) and
    # ``reward`` uses the filtered set (samples with mask ratio >= threshold,
    # so GRPO's K candidates have meaningful diversity). Phase switch happens
    # at the warmup→reward boundary.
    add_change3d_root(args.change3d_root)
    train_loader_warmup = build_loader(args, "train", train=True, min_ratio_override=0.0)
    has_filter = float(getattr(args, "min_change_ratio", 0.0)) > 0.0
    if has_filter:
        train_loader_reward = build_loader(args, "train", train=True)
    else:
        train_loader_reward = train_loader_warmup
    val_loader = build_loader(args, "val", train=False)
    test_loader = build_loader(args, "test", train=False)
    eval_loader = val_loader if args.eval_split == "val" else test_loader

    model = build_change3d_model(
        args, ckpt_path=(args.ckpt or None), train=True,
    )
    set_trainable_policy(
        model,
        train_backbone_last=bool(args.train_backbone_last),
        train_base_decoder=bool(args.train_base_decoder),
    )
    # Always include decoder + fusion + spatial_head in the trainable set.
    for name, p in model.named_parameters():
        if any(k in name for k in ("decoder", "fuses", "fusions", "reassembles", "spatial_head")):
            p.requires_grad = True

    decoder_params = [p for p in model.parameters() if p.requires_grad]
    n_dec = sum(p.numel() for p in decoder_params)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"[params] decoder/head trainable={n_dec/1e6:.2f}M / total={n_total/1e6:.2f}M")

    # MSAD (jointly trained, only built when needed)
    msad = None
    ext_prior = None
    ext_channels = 0
    msad_params = []
    if msad_enabled:
        stage_channels = model.spatial_head.stage_channels
        if args.use_external_prior:
            ext_prior = ExternalPrior(backbone=args.external_backbone).to(args.device)
            ext_channels = ext_prior.out_channels
            print(f"[ExternalPrior] backbone={args.external_backbone} "
                  f"out_channels={ext_channels}")
        msad = MultiStageProcessReward(
            stage_channels, hidden=args.msad_hidden, ext_channels=ext_channels,
        ).to(args.device)
        msad_params = list(msad.parameters())
        print(f"[params] msad={sum(p.numel() for p in msad_params)/1e6:.2f}M  "
              f"stage_channels={list(stage_channels)}")

    noise_module = LowRankSpatialNoise(sigma=args.grpo_sigma).to(args.device) \
        if (msad_enabled and args.grpo_k > 1) else None

    optimizer = torch.optim.AdamW(
        [
            {"params": decoder_params, "lr": args.lr},
        ] + (
            # MSAD trains 5x faster — it's a fresh-init aux network, the
            # decoder starts from a strong frozen-backbone init.
            [{"params": msad_params, "lr": args.lr * 5.0}] if msad_enabled else []
        ),
        weight_decay=1e-4,
    )

    # NOTE on EMA: we deliberately defer EMA construction to the start of the
    # reward-active phase. Constructing EMA at epoch 1 with a randomly-init'd
    # decoder makes the shadow track a near-random starting point — and the
    # eval/save paths apply that shadow, so for many epochs you'd see F1≈0
    # even though the actual model is learning. By instantiating EMA only
    # *after* the supervised warmup converges, the shadow starts from a
    # strong base and EMA does what it's supposed to do (variance reduction
    # in the noisier reward-active phase).
    ema = None
    # Approximate total steps using the warmup loader length × epochs. The
    # reward-active loader is usually a strict subset, so this overestimates
    # by a few %; cosine_lr's behaviour at min_lr_factor floor handles that.
    total_steps = max(1, len(train_loader_warmup) * args.epochs)
    global_step = 0
    best_f1 = -1.0

    # Reward-warmup: epoch index at which Direct + GRPO start contributing.
    # epoch numbering is 1..epochs, so reward_start_epoch = int(epochs * ratio) + 1
    # means rewards activate AT epoch (warmup_until + 1).
    reward_warmup_until = int(args.epochs * args.reward_warmup_ratio)
    print(f"[warmup] supervised + MSAD-disc only for epochs 1..{reward_warmup_until}; "
          f"Direct + GRPO active from epoch {reward_warmup_until + 1} onwards "
          f"(reward_warmup_ratio={args.reward_warmup_ratio})")

    # Initial eval (before any training).
    before = evaluate(args, model, eval_loader)
    print(f"[before] {args.eval_split} F1={before['F1']:.4f} IoU={before['IoU']:.4f}")

    # Snapshot the spatial_head parameter list once so we can freeze/unfreeze it
    # cleanly each epoch. During warmup we want spatial_head ENTIRELY frozen so
    # the optimisation is identical to a plain supervised run on the base
    # decoder — no residual-head random-init drift, no MSAD signal, nothing.
    head_params = list(model.spatial_head.parameters()) if hasattr(model, "spatial_head") else []

    for epoch in range(1, args.epochs + 1):
        model.train()
        rewards_active = epoch > reward_warmup_until

        # Freeze/unfreeze spatial_head based on warmup state.
        for p in head_params:
            p.requires_grad = rewards_active

        if epoch == reward_warmup_until + 1:
            print(f"[reward-on] unfreezing spatial_head and activating "
                  f"MSAD-disc + Direct + GRPO from epoch {epoch}")
            # Reload the BEST warmup checkpoint as the starting point for the
            # reward-active phase. Without this, we'd inherit whatever state
            # the model happened to be in at the last warmup epoch, which is
            # rarely the best (F1 oscillates during long warmup).
            if os.path.exists(args.save_path) and best_f1 > 0:
                blob = torch.load(args.save_path, map_location=args.device)
                model.load_state_dict(blob["model_state"], strict=True)
                warmup_best_epoch = blob.get("epoch", "?")
                print(f"[warmup-best] reloaded best warmup state "
                      f"(epoch={warmup_best_epoch}, "
                      f"{args.eval_split}_F1={best_f1:.4f}) "
                      f"— reward-active phase starts from here")
                # Re-freeze spatial_head momentarily before unfreezing — the
                # reload also restores spatial_head params which were frozen
                # during warmup; we explicitly unfreeze again now.
                for p in head_params:
                    p.requires_grad = True
            else:
                print(f"[warmup-best] no improving warmup ckpt found "
                      f"(best_f1={best_f1}); proceeding from current state")
            # Initialise EMA from this best state (not random init, not last
            # epoch state). EMA shadow now starts from a strong base.
            ema = ModelEMA(model, decay=args.ema_decay)
            print(f"[ema-init] initialised EMA shadow from best warmup state "
                  f"(decay={args.ema_decay})")
        # Pick the right train loader for this epoch:
        #   warmup → full unfiltered set
        #   reward-active → filtered set (mask ratio >= min_change_ratio)
        # Print a one-line note when the loader switches.
        active_loader = train_loader_reward if rewards_active else train_loader_warmup
        if epoch == 1:
            n_w = len(train_loader_warmup.dataset)
            print(f"[loader] warmup uses {n_w} samples (full, no min_change_ratio filter)")
        if epoch == reward_warmup_until + 1 and has_filter:
            n_r = len(train_loader_reward.dataset)
            print(f"[loader] reward-active uses {n_r} samples "
                  f"(filtered at min_change_ratio={args.min_change_ratio})")

        running = {"sup": 0.0, "disc": 0.0, "direct": 0.0, "grpo": 0.0, "n": 0}
        last_real = None  # most-recent r_real.mean() within this epoch
        last_fake = None
        for batch in active_loader:
            global_step += 1
            lr = cosine_lr(
                global_step, total_steps,
                base_lr=args.lr, warmup_steps=args.warmup_steps,
                min_lr_factor=args.min_lr_factor,
            )
            for i, pg in enumerate(optimizer.param_groups):
                pg["lr"] = lr * (5.0 if i == 1 else 1.0)

            pre, post, target = split_batch(batch, args.device)
            B = pre.shape[0]

            # Decoder forward. During warmup we only need the final probability
            # — spatial_head is frozen so its delta is zero (or near-zero from
            # zero-init), and we don't compute any MSAD-side signal.
            if msad_enabled and rewards_active:
                out = model.update_bcd_full(pre, post)
                final_prob = out["final_prob"]
                base_logit = out["base_logit"]
                delta_logit = out["delta_logit"]
                stages = out["stages"]
                scale = out["residual_scale"]
            else:
                final_prob = as_prob(model.update_bcd(pre, post))

            # --- Loss 1: supervised (always on) ---
            sup = bce_dice_loss(final_prob, target, pos_weight=args.pos_weight)

            disc = sup.new_zeros(())
            mono = sup.new_zeros(())
            calib = sup.new_zeros(())
            direct = sup.new_zeros(())
            grpo = sup.new_zeros(())

            # During warmup: skip ALL post-training losses. Pure supervised on
            # the base decoder, identical to a vanilla CD baseline.
            if msad_enabled and rewards_active:
                ext_feat = ext_prior(pre, post) if ext_prior is not None else None
                stages_det = [s.detach() for s in stages]
                ext_feat_det = ext_feat.detach() if ext_feat is not None else None

                # --- Loss 2: MSAD discriminator (real GT vs perturbed GT) ---
                if args.lambda_disc > 0:
                    with torch.no_grad():
                        fake_mask = perturb_mask(target, p_flip=args.p_flip, mode="mix")
                    r_real, per_stage_real = msad(
                        stages_det, target, pre, post, ext_feat=ext_feat_det,
                        return_stage=True,
                    )
                    r_fake = msad(stages_det, fake_mask, pre, post, ext_feat=ext_feat_det)
                    disc = prm_discriminator_loss(r_real, r_fake)
                    mono = prm_monotonicity_loss(per_stage_real)

                # --- Loss 3: direct MSAD signal into decoder (MSAD frozen) ---
                if args.lambda_direct > 0:
                    with freeze_module(msad):
                        r_policy = msad(stages, final_prob, pre, post, ext_feat=ext_feat)
                    if args.direct_squash > 0:
                        r_squashed = torch.tanh(r_policy / args.direct_squash) * args.direct_squash
                    else:
                        r_squashed = r_policy
                    direct = -r_squashed.mean()

                # --- Loss 4: Pixel-GRPO over K stochastic candidates ---
                # Standard GRPO objective transposed to dense prediction:
                #   1. Sample K logit perturbations from a low-rank Gaussian noise.
                #   2. Sample K binary masks (Bernoulli over those logits).
                #   3. MSAD (frozen here) gives per-pixel rewards for each candidate.
                #   4. Per-pixel advantage = (R - mean_K) / std_K, clipped.
                #   5. Score-function loss: -A · log_prob(mask | logits), with PPO
                #      trust-region clip (effectively inactive on-policy).
                # MSAD also gets a calibration signal: its candidate-image-level
                # mean rewards should correlate with each candidate's true IoU.
                if args.lambda_grpo > 0 and args.grpo_k > 1:
                    K = args.grpo_k
                    # K_logits has gradient through base_logit / delta_logit / scale.
                    K_logits = stochastic_logits(
                        base_logit, delta_logit, scale, noise_module,
                        k=K, tau=args.grpo_tau,
                    )  # [B, K, H, W]
                    with torch.no_grad():
                        K_masks = sample_masks(K_logits)  # [B, K, H, W] in {0, 1}
                        K_masks_5d = K_masks.unsqueeze(2)  # [B, K, 1, H, W]
                        # MSAD frozen → reward signal carries no gradient.
                        K_rewards = msad.reward_for_candidates(
                            stages_det, K_masks_5d, pre, post, ext_feat=ext_feat_det,
                        )  # [B, K, H, W]
                        if args.reward_squash > 0:
                            K_rewards = torch.tanh(K_rewards / args.reward_squash) * args.reward_squash
                    # Log prob with gradient flowing back to the decoder via K_logits.
                    logp_new = bernoulli_logprob(K_logits, K_masks)  # [B, K, H, W]
                    logp_old = logp_new.detach()  # single-epoch on-policy
                    # Focus on the union of GT and any predicted positive — same
                    # convention as the legacy script.
                    with torch.no_grad():
                        focus = (target > 0.5).float()
                        focus = focus + (final_prob > 0.5).float()
                        focus = (focus > 0).float()
                    grpo = pixel_grpo_loss(
                        logp_new, logp_old, K_rewards,
                        focus=focus,
                        clip_eps=args.grpo_clip_eps,
                        std_eps=args.grpo_std_eps,
                        advantage_clip=args.grpo_advantage_clip,
                    )

                    # --- MSAD calibration on the same K candidates ---
                    if args.lambda_calib > 0:
                        with torch.no_grad():
                            ious = iou_per_candidate(K_masks_5d, target)  # [B, K]
                        r_all_img = []
                        for k in range(K):
                            r_k = msad(stages_det, K_masks_5d[:, k], pre, post,
                                       ext_feat=ext_feat_det)
                            r_all_img.append(r_k.mean(dim=(1, 2, 3)))
                        r_all_img = torch.stack(r_all_img, dim=1)  # [B, K]
                        calib = prm_iou_calibration_loss(
                            r_all_img.flatten(), ious.flatten(),
                        )

            # --- Optional regularisers ---
            bnd = boundary_loss(final_prob, target) if args.lambda_boundary > 0 else sup.new_zeros(())
            fp = false_positive_penalty(final_prob, target) if args.lambda_fp > 0 else sup.new_zeros(())

            loss = (
                args.lambda_sup * sup
                + args.lambda_disc * disc
                + args.lambda_mono * mono
                + args.lambda_calib * calib
                + args.lambda_direct * direct
                + args.lambda_grpo * grpo
                + args.lambda_boundary * bnd
                + args.lambda_fp * fp
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(decoder_params + msad_params, 5.0)
            optimizer.step()
            if ema is not None:
                ema.update(model)

            running["sup"] += sup.item() * B
            running["disc"] += disc.item() * B
            running["direct"] += direct.item() * B
            running["grpo"] += grpo.item() * B
            running["n"] += B
            # Track the most recent MSAD gap for the epoch summary (reward
            # phase only). Per-step prints removed — keep only epoch summaries.
            if msad_enabled and rewards_active and args.lambda_disc > 0:
                with torch.no_grad():
                    last_real = r_real.mean().item()
                    last_fake = r_fake.mean().item()

        # ---- Epoch summary + best-ckpt selection ----
        n = max(1, running["n"])
        # Eval directly on the model during warmup (no EMA, identical to old
        # supervised baseline). EMA is only instantiated after warmup ends.
        if ema is not None:
            ema.apply_shadow(model)
        scores = evaluate(args, model, eval_loader)
        if ema is not None:
            ema.restore(model)
        phase = "reward-active" if rewards_active else "warmup"
        if rewards_active:
            gap_str = (f" | gap={last_real - last_fake:+.3f}"
                       if last_real is not None and last_fake is not None else "")
            print(
                f"[epoch {epoch:03d} | {phase}] sup={running['sup']/n:.4f} "
                f"disc={running['disc']/n:.4f} direct={running['direct']/n:+.4f} "
                f"grpo={running['grpo']/n:.4f} "
                f"| ({args.eval_split}) F1={scores['F1']:.4f} "
                f"IoU={scores['IoU']:.4f} P={scores['precision']:.4f} "
                f"R={scores['recall']:.4f}{gap_str}"
            )
        else:
            # Warmup: only sup is meaningful — print just that + the eval row.
            print(
                f"[epoch {epoch:03d} | {phase}] sup={running['sup']/n:.4f} "
                f"| ({args.eval_split}) F1={scores['F1']:.4f} "
                f"IoU={scores['IoU']:.4f} P={scores['precision']:.4f} "
                f"R={scores['recall']:.4f}"
            )
        if scores["F1"] > best_f1:
            best_f1 = scores["F1"]
            if ema is not None:
                ema.apply_shadow(model)
            directory = os.path.dirname(args.save_path)
            if directory:
                os.makedirs(directory, exist_ok=True)
            blob = {
                "model_state": model.state_dict(),
                "args": vars(args),
                "best_f1": best_f1,
                "epoch": epoch,
            }
            if msad is not None:
                blob.update({
                    "msad_state": msad.state_dict(),
                    "stage_channels": list(model.spatial_head.stage_channels),
                    "ext_channels": ext_channels,
                    "external_backbone": args.external_backbone if args.use_external_prior else "",
                    "use_external_prior": int(args.use_external_prior),
                })
            torch.save(blob, args.save_path)
            if ema is not None:
                ema.restore(model)
            print(f"  saved ckpt to {args.save_path}  ({args.eval_split}_F1={best_f1:.4f})")

    # Final test
    blob = torch.load(args.save_path, map_location=args.device)
    model.load_state_dict(blob["model_state"], strict=True)
    test_scores = evaluate(args, model, test_loader)
    print(
        f"[final/test] F1={test_scores['F1']:.4f} IoU={test_scores['IoU']:.4f} "
        f"P={test_scores['precision']:.4f} R={test_scores['recall']:.4f} "
        f"(best on '{args.eval_split}', best_F1={best_f1:.4f})"
    )


if __name__ == "__main__":
    main()
