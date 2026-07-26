# ChangeTIP

DINO-based binary remote-sensing change detection with training-time injected preferences.

This repository is **not** the old Change3D + verifier-guided DPO pipeline described by the previous README. The current code implements a single end-to-end training script that combines:

- a frozen foundation backbone, default `DINOv3`;
- a DPT-style reassemble decoder and spatial residual head;
- `MSAD` (Multi-Stage Auxiliary Discriminator), formerly named PRM in code for checkpoint compatibility;
- direct MSAD-to-decoder preference gradients;
- Pixel-GRPO over stochastic candidate masks.

In short, the default method is closer to **DINOv3 + MSAD + Pixel-GRPO** than to the older DPO-only Change3D wrapper.

## Current Pipeline

```text
pre/post image pair
    -> frozen backbone (default: DINOv3; optional: DINOv2, SAM2, Change3D)
    -> DPT reassemble + temporal fusion
    -> ChangeDecoder -> base probability
    -> DPT-style spatial residual head
    -> final probability
    -> supervised loss + MSAD discriminator + direct MSAD loss + Pixel-GRPO
```

Training uses a two-phase curriculum in `changetip_align/train_baseline.py`:

1. Warmup phase: supervised BCE + Dice only. The residual head is frozen so the model first learns a strong base decoder.
2. Reward-active phase: the residual head is unfrozen, MSAD is trained, and direct MSAD / Pixel-GRPO losses are added.

## Main Files

- `changetip_align/train_baseline.py`: main and only active training entry point.
- `changetip_align/models.py`: backbone dispatch plus residual-head wrapper.
- `changetip_align/backbones/dinov3_cd.py`: DINOv3 ViT encoder for BCD.
- `changetip_align/backbones/dinov2_cd.py`: DINOv2 ViT encoder for BCD.
- `changetip_align/backbones/sam2_cd.py`: SAM2 image encoder for BCD.
- `changetip_align/backbones/decoder.py`: DPT reassemble, temporal fusion, and change decoder blocks.
- `changetip_align/heads.py`: spatial residual change head.
- `changetip_align/prm.py`: MSAD implementation. The class name `MultiStageProcessReward` is kept for compatibility.
- `changetip_align/preference.py`: supervised, boundary, direct preference, KL, DPO helper, and Pixel-GRPO losses.
- `changetip_align/sampling.py`: low-rank spatial noise and mask sampling.
- `changetip_align/evaluate.py`: threshold sweep evaluation and optional MSAD self-verifier decoding.
- `scripts/evaluate.py`: wrapper for `changetip_align.evaluate`.
- `scripts/msad_diagnostics.py`: checks whether MSAD scores rank candidate masks by IoU.
- `scripts/profile_efficiency.py`: profiles backbone variants.

Historical compiled caches may reference removed scripts such as `train_prm.py`, `train_grpo.py`, or `train_agent_dpo.py`, but those source files are not part of the current repository state.

## Backbones

Supported values for `--backbone`:

- `dinov3`: default training backbone. Architecture is selected by `--dinov3_arch`.
- `dinov2`: DINOv2 backbone. Architecture is selected by `--dino_arch`.
- `sam2`: SAM2 Hiera image encoder. Requires `--sam2_ckpt`.
- `change3d`: original Change3D trainer path. Requires a local Change3D checkout via `--change3d_root`.

Important default:

- `train_baseline.py` defaults to `--backbone dinov3`.
- `evaluate.py` defaults to `--backbone change3d`, so pass `--backbone dinov3` when evaluating DINOv3 checkpoints.
- MSAD's external prior defaults to `--external_backbone dinov2_vits14` when `--use_external_prior 1`.

## Install

The provided `requirements.txt` targets the newer DINOv3/SAM2 environment. Follow the install notes in that file, especially the CUDA-specific PyTorch install order.

Typical setup:

```bash
conda create -n ChangeTIP-newbb python=3.11 -y
conda activate ChangeTIP-newbb

pip install torch==2.5.1+cu118 torchvision==0.20.1+cu118 \
  --index-url https://download.pytorch.org/whl/cu118

pip install -r requirements.txt

# Optional, only for --backbone sam2
pip install --no-deps git+https://github.com/facebookresearch/sam2.git
```

## Train

Run the current end-to-end trainer as a module:

```bash
cd /fast/Wang/ChangeTIP-Align
CUDA_VISIBLE_DEVICES=0 python -m changetip_align.train_baseline \
  --change3d_root /fast/Wang/Change3D \
  --data_root /fast/Wang/Chaofen/RLCD \
  --save_path outputs/rlcd_dinov3_msad.pth \
  --backbone dinov3 \
  --dinov3_arch vitl16_sat \
  --dinov3_input_size 512 \
  --decoder_head dpt_residual \
  --batch_size 4 \
  --epochs 80 \
  --lr 3e-4 \
  --reward_warmup_ratio 0.7 \
  --grpo_k 4 \
  --lambda_sup 1.0 \
  --lambda_disc 1.0 \
  --lambda_mono 0.2 \
  --lambda_calib 0.5 \
  --lambda_direct 0.1 \
  --lambda_grpo 0.5 \
  --eval_split test
```

To train a plain supervised baseline with the same backbone/decoder, disable the MSAD and GRPO terms:

```bash
python -m changetip_align.train_baseline \
  --change3d_root /fast/Wang/Change3D \
  --data_root /fast/Wang/Chaofen/RLCD \
  --save_path outputs/rlcd_dinov3_sup.pth \
  --backbone dinov3 \
  --dinov3_arch vitl16_sat \
  --decoder_head dpt_residual \
  --lambda_disc 0 \
  --lambda_direct 0 \
  --lambda_grpo 0 \
  --lambda_mono 0 \
  --lambda_calib 0
```

## Evaluate

Always match evaluation arguments to the checkpoint's training backbone and head:

```bash
python scripts/evaluate.py \
  --change3d_root /fast/Wang/Change3D \
  --data_root /fast/Wang/Chaofen/RLCD \
  --ckpt outputs/rlcd_dinov3_msad.pth \
  --backbone dinov3 \
  --dinov3_arch vitl16_sat \
  --dinov3_input_size 512 \
  --decoder_head dpt_residual \
  --split test \
  --eval_tta 1
```

If the checkpoint contains `msad_state`, optional MSAD self-verifier decoding can be enabled:

```bash
python scripts/evaluate.py \
  --change3d_root /fast/Wang/Change3D \
  --data_root /fast/Wang/Chaofen/RLCD \
  --ckpt outputs/rlcd_dinov3_msad.pth \
  --backbone dinov3 \
  --dinov3_arch vitl16_sat \
  --dinov3_input_size 512 \
  --decoder_head dpt_residual \
  --split test \
  --eval_tta 1 \
  --use_self_verifier 1 \
  --sv_thresholds 0.30,0.40,0.50,0.60,0.70
```

## Diagnostics And Profiling

Check whether MSAD ranks thresholded masks consistently with true IoU:

```bash
python scripts/msad_diagnostics.py \
  --change3d_root /fast/Wang/Change3D \
  --data_root /fast/Wang/Chaofen/RLCD \
  --ckpt outputs/rlcd_dinov3_msad.pth \
  --backbone dinov3 \
  --dinov3_arch vitl16_sat \
  --dinov3_input_size 512
```

Profile backbone variants:

```bash
python scripts/profile_efficiency.py \
  --change3d_root /fast/Wang/Change3D \
  --backbone dinov3 \
  --dinov3_arch vitl16_sat \
  --dinov3_input_size 512 \
  --device cuda
```

## Suggested Ablations

- Backbone: `dinov3` vs `dinov2` vs `sam2` vs `change3d`.
- DINOv3 architecture: `vits16plus` vs `vitl16` vs `vitl16_sat`.
- MSAD disabled: set `lambda_disc=lambda_direct=lambda_grpo=0`.
- Direct MSAD only: enable `lambda_direct`, disable `lambda_grpo`.
- Pixel-GRPO only after MSAD discriminator: enable `lambda_grpo`, vary `grpo_k`.
- External prior: `--use_external_prior 1/0`, or `--external_backbone resnet18/dinov2_vits14`.
- Fusion mode: `--fusion_mode concat` vs `--fusion_mode frm`.
- Reward warmup: sweep `--reward_warmup_ratio`.
