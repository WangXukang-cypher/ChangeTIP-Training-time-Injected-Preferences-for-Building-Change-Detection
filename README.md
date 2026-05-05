# ChangeTIP-Align

TIPS-inspired spatial preference alignment for binary remote-sensing change detection.

This project is a clean successor to the earlier verifier-guided DPO prototype. It keeps the
candidate/verifier/DPO training logic, but adds a stronger architecture component: a
TIPS/DPT-style dense residual head adapted to Change3D's CNN feature pyramid.

## Core Idea

```text
pre/post image pair
    -> Change3D encoder
    -> original Change3D decoder -> base probability
    -> DPT-style spatial residual head -> delta logit
    -> sigmoid(logit(base probability) + alpha * delta logit)
    -> verifier-guided DPO + supervised/KL/boundary constraints
```

The residual head is zero-initialized at the output layer, so loading an old Change3D checkpoint
starts from the same prediction as the original model. Post-training then learns a spatial
refinement head that is explicitly aligned with verifier preferences.

## Why This Folder Exists

The old `ChangeAgent-DPO` name described the training procedure but not the project direction.
`ChangeTIP-Align` is meant to describe the actual method:

- `Change`: binary change detection.
- `TIP`: TIPS-inspired spatial dense prediction head.
- `Align`: preference alignment through verifier-guided DPO.

The original `ChangeAgent-DPO` directory is not modified.

## Main Files

- `changetip_align/heads.py`: TIPS/DPT-style residual dense head.
- `changetip_align/models.py`: Change3D builder plus residual-head wrapper.
- `changetip_align/train_verifier.py`: learned verifier for candidate masks.
- `changetip_align/train_agent_dpo.py`: verifier-guided DPO training for the residual head.
- `changetip_align/evaluate.py`: threshold sweep evaluation.
- `inf_batch.py`: batch inference/export.

## Train Verifier

The verifier can be trained from the original Change3D checkpoint. By default it uses the baseline
decoder as the candidate generator.

```bash
python scripts/train_verifier.py \
  --change3d_root /fast/Wang/Change3D \
  --data_root /fast/Wang/Chaofen/RLCD \
  --ckpt /fast/Wang/Change3D/S128exp/WHU-CD_iter_200000_lr_0.0002/best_model.pth \
  --save_path outputs/mashiki/agentdpo.pt \
  --batch_size 4 \
  --epochs 12 \
  --thresholds 0.30,0.40,0.50,0.60,0.70 \
  --pretrained /fast/Wang/Change3D/X3D_L.pyth
```

## Train ChangeTIP-Align

The DPO stage defaults to `--decoder_head dpt_residual`, loads the old checkpoint into the wrapped
model, freezes the base model, and trains only the residual spatial head unless
`--train_base_decoder 1` is set.

```bash
cd /fast/Wang/ChangeTIP-Align
CUDA_VISIBLE_DEVICES=0 python scripts/train_agent_dpo.py \
  --change3d_root /fast/Wang/Change3D \
  --data_root /fast/Wang/Chaofen/RLCD \
  --ckpt /fast/Wang/Change3D/S128exp/WHU-CD_iter_200000_lr_0.0002/best_model.pth \
  --verifier_ckpt outputs/mashiki/agentdpo.pt \
  --save_path outputs/rlcd_changetip_align.pth \
  --batch_size 4 \
  --eval_split test \
  --eval_tta 1 \
  --epochs 30 \
  --lr 2e-5 \
  --pretrained /fast/Wang/Change3D/X3D_L.pyth \
  --thresholds 0.30,0.40,0.50,0.60,0.70
```

## Evaluate

`evaluate.py` defaults to the residual wrapper, so it can load either an original Change3D
checkpoint or a trained ChangeTIP-Align checkpoint.

```bash
python scripts/evaluate.py \
  --change3d_root /fast/Wang/Change3D \
  --data_root /fast/Wang/Chaofen/RLCD \
  --ckpt outputs/rlcd_changetip_align.pth \
  --split test \
  --pretrained /fast/Wang/Change3D/X3D_L.pyth
```

## Recommended Ablations

- Change3D baseline.
- Baseline decoder + verifier-guided DPO.
- DPT residual head + supervised/KL only.
- DPT residual head + verifier-guided DPO.
- DPT residual head with/without `--train_base_decoder 1`.
- Candidate generation with/without flips/scales.

---

## Pixel-GRPO + Process Reward (new pipeline)

This pipeline replaces the verifier-guided DPO stage with three coupled
components:

- **Pixel-GRPO** — `changetip_align/preference.py::pixel_grpo_loss`.
  Group-relative, critic-free, per-pixel advantage estimation with PPO-clip.
- **Multi-Stage Process Reward Model (PRM)** — `changetip_align/prm.py`.
  Discriminator decomposed across the spatial-head fusion stages, trained
  with image-level IoU calibration and stage-monotonicity regularization.
- **Stochastic Spatial Sampling** — `changetip_align/sampling.py`.
  Low-rank Gaussian noise injected on the residual logit so that K candidate
  masks have meaningful log-prob differences.

See `THEORY.md` for the variance-reduction and concentration analysis.

### Stage 1 — Train the PRM (Linux)

```bash
cd /fast/Wang/ChangeTIP-Align
CUDA_VISIBLE_DEVICES=0 python scripts/train_prm.py \
  --change3d_root /fast/Wang/Change3D \
  --data_root /fast/Wang/Chaofen/RLCD \
  --ckpt /fast/Wang/Change3D/S128exp/WHU-CD_iter_200000_lr_0.0002/best_model.pth \
  --pretrained /fast/Wang/Change3D/X3D_L.pyth \
  --save_path outputs/rlcd_prm.pt \
  --batch_size 8 \
  --epochs 12 \
  --lr 1e-3 \
  --sampling_sigma 4.0 \
  --sampling_tau 1.0 \
  --p_flip 0.05 \
  --lambda_disc 1.0 \
  --lambda_mono 0.2 \
  --lambda_calib 0.5
```

### Stage 2 — Train the policy with Pixel-GRPO (Linux)

```bash
cd /fast/Wang/ChangeTIP-Align
CUDA_VISIBLE_DEVICES=0 python scripts/train_grpo.py \
  --change3d_root /fast/Wang/Change3D \
  --data_root /fast/Wang/Chaofen/RLCD \
  --ckpt /fast/Wang/Change3D/S128exp/WHU-CD_iter_200000_lr_0.0002/best_model.pth \
  --prm_ckpt outputs/rlcd_prm.pt \
  --pretrained /fast/Wang/Change3D/X3D_L.pyth \
  --save_path outputs/rlcd_changetip_grpo.pth \
  --batch_size 4 \
  --epochs 30 \
  --lr 2e-5 \
  --grpo_k 8 \
  --sampling_sigma 4.0 \
  --sampling_tau_init 1.0 \
  --sampling_tau_final 0.2 \
  --clip_eps 0.2 \
  --lambda_sup 1.0 \
  --lambda_grpo 1.0 \
  --lambda_kl 3e-3 \
  --lambda_boundary 0.3 \
  --lambda_fp 0.3 \
  --eval_split test \
  --eval_tta 1
```

### Stage 3 — Evaluate

```bash
python scripts/evaluate.py \
  --change3d_root /fast/Wang/Change3D \
  --data_root /fast/Wang/Chaofen/RLCD \
  --ckpt outputs/rlcd_changetip_grpo.pth \
  --split test \
  --pretrained /fast/Wang/Change3D/X3D_L.pyth \
  --eval_tta 1
```

### Suggested Ablations for the New Pipeline

- **DPO baseline** (existing): `scripts/train_agent_dpo.py` — main competitor in the paper.
- **Pixel-GRPO without PRM** (oracle reward): replace `--prm_ckpt` rewards with
  per-pixel IoU vs GT to isolate the PRM contribution.
- **Group size sweep**: `--grpo_k {2, 4, 8, 16}` — verifies Theorem 1.
- **No spatial noise**: `--sampling_tau_init 0 --sampling_tau_final 0` — should
  collapse to near-zero GRPO loss, verifies Theorem 4.
- **Clip eps sweep**: `--clip_eps {0.1, 0.2, 0.4}` — Proposition 1.
- **Image-level KL vs PRM-gated KL**: keep `--lambda_kl` but replace
  `prm_gated_kl` call with `kl_bernoulli` in `train_grpo.py`.
- **PRM stage weights**: edit `MultiStageProcessReward(weight_temperature=...)`
  to test uniform vs deep-favoring weighting.

