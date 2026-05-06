# =============================================================================
# Component-ablation study for ChangeTIP-Align.
#
# Trains a single PRM, then runs 7 variants of the post-training stage and
# evaluates each on the test set. Each variant changes exactly one component
# relative to the "full" baseline.
#
# Variants
#   full                — all components on (ChangeTIP-Align)
#   no_residual_head    — head replaces base decoder (head_mode=direct, zero_init off)
#   no_zero_init        — head out-conv kaiming-init instead of zero
#   use_dpo             — replace pixel-GRPO with DPO on best/worst PRM pair
#   no_prm_gated_kl     — replace PRM-gated KL with unweighted KL
#   no_boundary         — lambda_boundary = 0
#   no_spatial_noise    — sampling_sigma = 0  (i.i.d. Bernoulli noise)
#
# Usage:
#   .\scripts\run_ablation.ps1 -DataRoot <path> -BaseCkpt <path> -OutDir <path>
# =============================================================================

param(
    [Parameter(Mandatory=$true)] [string]$DataRoot,
    [Parameter(Mandatory=$true)] [string]$BaseCkpt,
    [Parameter(Mandatory=$true)] [string]$OutDir,
    [string]$Change3dRoot = "..\S-change3d",
    [string]$Dataset      = "WHU-CD",
    [string]$Device       = "cuda",
    [int]$BatchSize       = 4,
    [int]$NumWorkers      = 4,
    [int]$Epochs          = 24,
    [int]$PrmEpochs       = 12,
    [int]$PrmBatchSize    = 8,
    [double]$MinChangeRatio = 0.02,
    [int]$UseExternalPrior  = 1,
    [string]$ExternalBackbone = "resnet18",
    [string[]]$Variants = @(
        "full",
        "no_residual_head",
        "no_zero_init",
        "use_dpo",
        "no_prm_gated_kl",
        "no_boundary",
        "no_spatial_noise"
    )
)

if (-not (Test-Path $OutDir)) {
    New-Item -ItemType Directory -Path $OutDir | Out-Null
}
$LogDir = Join-Path $OutDir "logs"
if (-not (Test-Path $LogDir)) {
    New-Item -ItemType Directory -Path $LogDir | Out-Null
}

# -----------------------------------------------------------------------------
# Step 1: Train PRM once. The PRM is shared across variants — it is trained
# against the full residual head's stage features, and every variant either
# uses that head as-is (residual mode) or replaces only the head-output
# semantics (direct mode), which leaves stage_channels unchanged.
# -----------------------------------------------------------------------------
$PrmCkpt = Join-Path $OutDir "prm.pth"
if (-not (Test-Path $PrmCkpt)) {
    Write-Host "==================== Training PRM ====================" -ForegroundColor Cyan
    python -m changetip_align.train_prm `
        --change3d_root $Change3dRoot `
        --data_root $DataRoot `
        --dataset $Dataset `
        --ckpt $BaseCkpt `
        --save_path $PrmCkpt `
        --device $Device `
        --batch_size $PrmBatchSize `
        --num_workers $NumWorkers `
        --epochs $PrmEpochs `
        --lr 1e-3 `
        --in_height 256 --in_width 256 `
        --decoder_head dpt_residual `
        --head_channels 64 --head_init_scale 0.1 `
        --prm_hidden 128 `
        --sampling_sigma 4.0 --sampling_tau 1.0 `
        --p_flip 0.05 `
        --lambda_disc 1.0 --lambda_mono 0.2 --lambda_calib 0.5 `
        --use_external_prior $UseExternalPrior `
        --external_backbone $ExternalBackbone `
        --min_change_ratio $MinChangeRatio `
        2>&1 | Tee-Object -FilePath (Join-Path $LogDir "prm.log")
    if ($LASTEXITCODE -ne 0) { Write-Error "PRM training failed."; exit 1 }
} else {
    Write-Host "[skip] PRM checkpoint already exists at $PrmCkpt" -ForegroundColor Yellow
}

# -----------------------------------------------------------------------------
# Step 2: Run each variant. Each saves to its own checkpoint and log file so
# the runs are independent.
# -----------------------------------------------------------------------------
foreach ($v in $Variants) {
    $Save = Join-Path $OutDir ("grpo_" + $v + ".pth")
    $Log  = Join-Path $LogDir ($v + ".log")

    Write-Host "==================== Variant: $v ====================" -ForegroundColor Cyan
    python -m changetip_align.train_grpo `
        --change3d_root $Change3dRoot `
        --data_root $DataRoot `
        --dataset $Dataset `
        --ckpt $BaseCkpt `
        --prm_ckpt $PrmCkpt `
        --save_path $Save `
        --device $Device `
        --batch_size $BatchSize `
        --num_workers $NumWorkers `
        --epochs $Epochs `
        --lr 2e-5 `
        --warmup_steps 500 --min_lr_factor 0.2 `
        --ema_decay 0.999 `
        --in_height 256 --in_width 256 `
        --decoder_head dpt_residual `
        --head_channels 64 --head_init_scale 0.1 `
        --train_backbone_last 0 --train_base_decoder 0 `
        --grpo_k 8 `
        --sampling_sigma 4.0 `
        --sampling_tau_init 1.0 --sampling_tau_final 0.2 `
        --clip_eps 0.2 --advantage_clip 5.0 `
        --lambda_sup 1.0 `
        --lambda_grpo 1.0 `
        --lambda_dpo 0.0 `
        --lambda_prm_direct 0.0 `
        --lambda_kl 3e-3 `
        --lambda_boundary 0.3 --lambda_fp 0.3 `
        --kl_temperature 1.0 `
        --grpo_focus_dilate 5 `
        --eval_split test `
        --eval_tta 0 `
        --min_change_ratio $MinChangeRatio `
        --variant $v `
        2>&1 | Tee-Object -FilePath $Log
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Variant $v failed (see $Log)."
        continue
    }
}

# -----------------------------------------------------------------------------
# Step 3: Test-set evaluation for every variant. Tabulate F1/IoU.
# -----------------------------------------------------------------------------
Write-Host "==================== Test-Set Evaluation ====================" -ForegroundColor Cyan
$Summary = Join-Path $OutDir "ablation_summary.txt"
"variant`tF1`tIoU`tP`tR" | Out-File -FilePath $Summary -Encoding utf8

foreach ($v in $Variants) {
    $Save = Join-Path $OutDir ("grpo_" + $v + ".pth")
    if (-not (Test-Path $Save)) {
        Write-Warning "Missing checkpoint for $v at $Save — skipping eval."
        continue
    }
    $EvalLog = Join-Path $LogDir ("eval_" + $v + ".log")
    # Match decoder_head/head_mode/zero_init to the checkpoint by re-applying variant.
    $HeadMode    = "residual"
    $HeadZeroInit = 1
    if ($v -eq "no_residual_head") { $HeadMode = "direct"; $HeadZeroInit = 0 }
    if ($v -eq "no_zero_init")     { $HeadZeroInit = 0 }

    python -m changetip_align.evaluate `
        --change3d_root $Change3dRoot `
        --data_root $DataRoot `
        --dataset $Dataset `
        --ckpt $Save `
        --split test `
        --device $Device `
        --batch_size 8 `
        --num_workers $NumWorkers `
        --in_height 256 --in_width 256 `
        --decoder_head dpt_residual `
        --head_channels 64 --head_init_scale 0.1 `
        --head_mode $HeadMode `
        --head_zero_init $HeadZeroInit `
        --thresholds "0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70" `
        --eval_tta 0 `
        2>&1 | Tee-Object -FilePath $EvalLog | Out-Null

    $bestLine = Select-String -Path $EvalLog -Pattern "^\[best\]" | Select-Object -Last 1
    if ($bestLine) {
        $line = $bestLine.Line
        if ($line -match "F1=([\d\.]+) IoU=([\d\.]+)") {
            $f1  = $matches[1]
            $iou = $matches[2]
            Write-Host ("{0,-22} F1={1} IoU={2}" -f $v, $f1, $iou) -ForegroundColor Green
            ("{0}`t{1}`t{2}" -f $v, $f1, $iou) | Out-File -FilePath $Summary -Append -Encoding utf8
        }
    }
}

Write-Host "`nAblation summary written to: $Summary" -ForegroundColor Cyan
Get-Content $Summary
