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
