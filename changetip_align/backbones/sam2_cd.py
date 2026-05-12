"""SAM2 Hiera image-encoder + cross-temporal fusion for BCD.

Why SAM2:
    SAM2's Hiera image encoder is hierarchical (4 stages at strides
    [4, 8, 16, 32]) and was pretrained for promptable segmentation, giving it
    strong boundary-aware priors that align well with our PRM's reward signal
    (especially around building edges).

Pipeline:
    pre, post  ─►  Hiera (frozen)  ─► 4 multi-scale features
                                   │
                                   ▼
                          CrossTimeFusion (per stage)
                                   │
                                   ▼
                            ChangeDecoder ────► base_prob

Dependency:
    The official SAM2 codebase. Install with::

        pip install git+https://github.com/facebookresearch/sam2.git

    and download a Hiera checkpoint (e.g. ``sam2.1_hiera_tiny.pt``). Pass the
    path via ``--sam2_ckpt`` and the matching config name (``sam2.1_hiera_t``,
    ``sam2.1_hiera_s``, etc.) via ``--sam2_cfg``.

    If the SAM2 package is unavailable, instantiation will fail with a clear
    message — switch to the DINOv2 backbone in that case.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .decoder import ChangeDecoder, build_fusion


# Default channel widths for Hiera variants used in SAM2.
# (Source: SAM2 Hiera config — these are the post-projection trunk channels at
# strides 4 / 8 / 16 / 32.)
_HIERA_CHANNELS = {
    "sam2.1_hiera_t": [96, 192, 384, 768],
    "sam2.1_hiera_s": [96, 192, 384, 768],
    "sam2.1_hiera_b+": [112, 224, 448, 896],
    "sam2.1_hiera_l": [144, 288, 576, 1152],
    "sam2_hiera_t": [96, 192, 384, 768],
    "sam2_hiera_s": [96, 192, 384, 768],
    "sam2_hiera_b+": [112, 224, 448, 896],
    "sam2_hiera_l": [144, 288, 576, 1152],
}

# SAM2 uses Hydra; ``build_sam2`` expects a config name relative to the
# ``sam2`` package's config search path (which is ``sam2/configs``). Map our
# short names to the actual yaml file paths.
_HIERA_CONFIG_PATHS = {
    "sam2.1_hiera_t":  "configs/sam2.1/sam2.1_hiera_t.yaml",
    "sam2.1_hiera_s":  "configs/sam2.1/sam2.1_hiera_s.yaml",
    "sam2.1_hiera_b+": "configs/sam2.1/sam2.1_hiera_b+.yaml",
    "sam2.1_hiera_l":  "configs/sam2.1/sam2.1_hiera_l.yaml",
    "sam2_hiera_t":    "configs/sam2/sam2_hiera_t.yaml",
    "sam2_hiera_s":    "configs/sam2/sam2_hiera_s.yaml",
    "sam2_hiera_b+":   "configs/sam2/sam2_hiera_b+.yaml",
    "sam2_hiera_l":    "configs/sam2/sam2_hiera_l.yaml",
}


def _build_sam2_image_encoder(cfg_name, ckpt_path):
    """Instantiate the SAM2 image_encoder (Hiera) from a config + checkpoint."""
    try:
        from sam2.build_sam import build_sam2
    except ImportError as e:
        raise ImportError(
            "SAM2 backbone requested but the `sam2` package is not installed. "
            "Run: pip install --no-deps git+https://github.com/facebookresearch/sam2.git"
        ) from e
    # Accept either our short name ("sam2.1_hiera_t") or the full yaml path.
    config_path = _HIERA_CONFIG_PATHS.get(cfg_name, cfg_name)
    sam2 = build_sam2(config_path, ckpt_path)
    return sam2.image_encoder


class SAM2Encoder(nn.Module):
    """Frozen SAM2 image encoder + cross-temporal fusion per stage."""

    def __init__(self, args):
        super().__init__()
        self.args = args
        cfg = getattr(args, "sam2_cfg", "sam2.1_hiera_t")
        ckpt = getattr(args, "sam2_ckpt", "")
        if not ckpt:
            raise ValueError(
                "SAM2 backbone requires --sam2_ckpt pointing to the checkpoint file."
            )
        self.cfg = cfg
        self.image_encoder = _build_sam2_image_encoder(cfg, ckpt)
        for p in self.image_encoder.parameters():
            p.requires_grad = False
        self.image_encoder.eval()

        embed_dims = _HIERA_CHANNELS.get(cfg)
        if embed_dims is None:
            raise ValueError(
                f"Unknown SAM2 config '{cfg}'. Add its channel widths to _HIERA_CHANNELS."
            )
        self.embed_dims = list(embed_dims)
        fusion_mode = getattr(args, "fusion_mode", "concat")
        self.fuses = nn.ModuleList([build_fusion(c, fusion_mode) for c in embed_dims])

        # SAM2 was trained at 1024x1024, but the Hiera image_encoder supports
        # arbitrary square inputs via pos_embed interpolation. For 256-pixel
        # CD patches, 1024 is wasteful (16x bilinear upsample). Default to
        # 512: a 2x upsample that keeps GPU cost reasonable while preserving
        # most of SAM2's pretraining benefit. Pass --sam2_input_size 1024 for
        # the highest-fidelity setting if you have the GPU budget.
        self.input_size = int(getattr(args, "sam2_input_size", 512))
        if self.input_size % 32 != 0:
            raise ValueError(
                f"--sam2_input_size={self.input_size} must be a multiple of 32 "
                f"(stride at the deepest stage)."
            )
        print(f"[SAM2Encoder] cfg={cfg} input_size={self.input_size} "
              f"embed_dims={self.embed_dims}")

    @torch.no_grad()
    def _features(self, x):
        """Run image_encoder.trunk to get the 4 hierarchical stage maps.

        SAM2's image_encoder = (trunk: Hiera, neck: FpnNeck). We bypass the
        FPN neck and consume raw trunk features so per-stage channels stay
        large enough for the PRM. Trunk returns a list of [B, C, H, W] feats.
        """
        x = F.interpolate(x, size=(self.input_size, self.input_size),
                          mode="bilinear", align_corners=False)
        # Trunk output shape varies by SAM2 version. Most variants return:
        #   list[Tensor] of length 4, fine→coarse, channels = self.embed_dims
        feats = self.image_encoder.trunk(x)
        if not isinstance(feats, (list, tuple)):
            raise RuntimeError(
                f"Unexpected SAM2 trunk output type: {type(feats)}. "
                f"Expected list of 4 stage tensors."
            )
        if len(feats) != 4:
            raise RuntimeError(
                f"Expected 4 stages from SAM2 trunk, got {len(feats)}."
            )
        return feats

    def forward(self, x, y, output_final=False):
        h_in, w_in = x.shape[-2:]
        pre_feats = self._features(x)
        post_feats = self._features(y)

        out = []
        for i, fu in enumerate(self.fuses):
            change = fu(pre_feats[i], post_feats[i])
            target_h = max(1, h_in // (4 * (2 ** i)))
            target_w = max(1, w_in // (4 * (2 ** i)))
            if change.shape[-2:] != (target_h, target_w):
                change = F.interpolate(
                    change, size=(target_h, target_w),
                    mode="bilinear", align_corners=False,
                )
            out.append([change])
        return out


class SAM2BCDTrainer(nn.Module):
    """SAM2-based BCD module with the same interface as Change3D's Trainer."""

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.encoder = SAM2Encoder(args)
        self.embed_dims = self.encoder.embed_dims
        self.decoder = ChangeDecoder(
            in_dim=self.embed_dims,
            channels=getattr(args, "decoder_channels", 128),
            has_sigmoid=True,
        )

    def update_bcd(self, x, y):
        features = self.encoder(x, y)
        change_features = [item[0] for item in features]
        return self.decoder(change_features)
