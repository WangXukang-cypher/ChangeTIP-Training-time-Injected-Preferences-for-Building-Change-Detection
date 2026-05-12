"""DINOv3 ViT encoder + DPT-reassemble + cross-temporal fusion for BCD.

Why DINOv3 over DINOv2 for this task:
    - 12x more pretraining data (1.7B vs 142M images).
    - Gram-matrix regularisation specifically targets long-range patch
      consistency, which is exactly what fails when DINOv2 underdetects
      changes at the corners of 256-pixel patches.
    - Patch size = 16 → 256/16 = 16 patches cleanly, no resize hack.
    - SAT-493M variant exists, pretrained on 493M satellite images, which is
      the closest thing to "domain-aligned" pretraining for remote-sensing
      change detection.

Dependency:
    pip install --upgrade "transformers>=4.46"

License:
    DINOv3 weights ship under a research-only license — fine for academic
    papers (CVPR/ICCV/etc.) but not for commercial deployment without a
    separate Meta license. Cite Simeoni et al., DINOv3, 2025.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .decoder import ChangeDecoder, build_fusion
from .dinov2_cd import DPTReassemble  # reuse the same reassemble block


# arch -> (HF model ID, embed_dim, layer_taps, depth)
# Layer taps follow the DPT convention: roughly 1/4, 1/2, 3/4, last block.
_DINO3_SPECS = {
    # General-purpose (LVD-1689M)
    "vits16":      ("facebook/dinov3-vits16-pretrain-lvd1689m",      384, [2, 5, 8, 11],   12),
    "vits16plus":  ("facebook/dinov3-vits16plus-pretrain-lvd1689m",  384, [2, 5, 8, 11],   12),
    "vitb16":      ("facebook/dinov3-vitb16-pretrain-lvd1689m",      768, [2, 5, 8, 11],   12),
    "vitl16":      ("facebook/dinov3-vitl16-pretrain-lvd1689m",     1024, [4, 11, 17, 23], 24),
    # Satellite (SAT-493M) — domain-matched for remote sensing
    "vitl16_sat":  ("facebook/dinov3-vitl16-pretrain-sat493m",      1024, [4, 11, 17, 23], 24),
}

_STAGE_INPUT_DIVISORS = (4, 8, 16, 32)


class DinoV3Encoder(nn.Module):
    """Frozen DINOv3 + per-stage DPT reassemble + cross-temporal fusion."""

    PATCH = 16

    def __init__(self, args):
        super().__init__()
        self.args = args
        arch = getattr(args, "dinov3_arch", "vits16plus")
        if arch not in _DINO3_SPECS:
            raise ValueError(
                f"Unknown dinov3_arch={arch}; choose from {list(_DINO3_SPECS)}"
            )
        model_id, dim, taps, _depth = _DINO3_SPECS[arch]
        self.dim = dim
        self.taps = taps
        self.model_id = model_id

        try:
            from transformers import AutoModel
        except ImportError as e:
            raise ImportError(
                "DINOv3 requires the `transformers` package. Install with: "
                "pip install --upgrade 'transformers>=4.46'"
            ) from e
        try:
            self.dino = AutoModel.from_pretrained(model_id)
        except Exception as e:
            raise RuntimeError(
                f"Failed to load DINOv3 '{model_id}'. Make sure you have "
                f"transformers>=4.46 and Hugging Face auth (some DINOv3 weights "
                f"are gated — accept the license at https://huggingface.co/{model_id}). "
                f"Original error: {e}"
            )
        for p in self.dino.parameters():
            p.requires_grad = False
        self.dino.eval()

        actual_depth = getattr(self.dino.config, "num_hidden_layers", _depth)
        if max(self.taps) >= actual_depth:
            new_taps = [max(0, int(round((i + 1) * actual_depth / 4)) - 1) for i in range(4)]
            print(f"[DinoV3Encoder] tap auto-correct: declared depth={_depth} taps={self.taps} "
                  f"-> actual depth={actual_depth} taps={new_taps}")
            self.taps = new_taps

        # Pick internal_size = largest multiple of 16 that fits inside the
        # input (or the user's override). For 256-pixel inputs this lands at
        # 256 exactly → 16x16 patch grid with no resize.
        in_h = getattr(args, "in_height", 256)
        in_w = getattr(args, "in_width", 256)
        side = min(in_h, in_w)
        forced = int(getattr(args, "dinov3_input_size", 0) or 0)
        if forced > 0:
            if forced % self.PATCH != 0:
                raise ValueError(
                    f"--dinov3_input_size={forced} must be a multiple of {self.PATCH}."
                )
            self.internal_size = forced
        else:
            self.internal_size = max(self.PATCH, (side // self.PATCH) * self.PATCH)
        self.grid = self.internal_size // self.PATCH
        print(f"[DinoV3Encoder] arch={arch} model_id={model_id} "
              f"internal_size={self.internal_size} patch_grid={self.grid}x{self.grid}")

        embed_dims = list(getattr(args, "embed_dims", [96, 192, 384, 768]))
        try:
            args.embed_dims = embed_dims
        except Exception:
            pass
        self.embed_dims = embed_dims

        scales = [4, 2, 1, 0.5]
        self.reassembles = nn.ModuleList([
            DPTReassemble(self.dim, embed_dims[i], scales[i]) for i in range(4)
        ])
        fusion_mode = getattr(args, "fusion_mode", "concat")
        self.fuses = nn.ModuleList([build_fusion(embed_dims[i], fusion_mode) for i in range(4)])

    @torch.no_grad()
    def _tokens(self, x):
        """Return tuple of [B, N, D] patch tokens at each tap (CLS+registers dropped)."""
        if x.shape[-2:] != (self.internal_size, self.internal_size):
            x = F.interpolate(x, size=(self.internal_size, self.internal_size),
                              mode="bilinear", align_corners=False)
        outputs = self.dino(pixel_values=x, output_hidden_states=True, return_dict=True)
        hidden_states = outputs.hidden_states  # tuple length = depth + 1
        # DINOv3 prepends 1 CLS + 4 register tokens; patch tokens are the
        # trailing N entries. Slicing the last `grid*grid` tokens is robust
        # to any prefix layout the HF impl decides to use.
        n_patch = self.grid * self.grid
        outs = []
        for tap in self.taps:
            h = hidden_states[tap + 1]  # +1: index 0 is post-embedding output
            h = h[:, -n_patch:, :]
            outs.append(h)
        return tuple(outs)

    def forward(self, x, y, output_final=False):
        h_in, w_in = x.shape[-2:]
        hp = wp = self.grid

        pre_tokens = self._tokens(x)
        post_tokens = self._tokens(y)

        out = []
        for i, (re, fu) in enumerate(zip(self.reassembles, self.fuses)):
            pre_feat = re(pre_tokens[i], hp, wp)
            post_feat = re(post_tokens[i], hp, wp)
            change = fu(pre_feat, post_feat)
            target_h = max(1, h_in // _STAGE_INPUT_DIVISORS[i])
            target_w = max(1, w_in // _STAGE_INPUT_DIVISORS[i])
            if change.shape[-2:] != (target_h, target_w):
                change = F.interpolate(
                    change, size=(target_h, target_w),
                    mode="bilinear", align_corners=False,
                )
            out.append([change])
        return out


class DinoV3BCDTrainer(nn.Module):
    """DINOv3-based BCD module with the same interface as Change3D's Trainer."""

    def __init__(self, args):
        super().__init__()
        self.args = args
        embed_dims = list(getattr(args, "embed_dims", [96, 192, 384, 768]))
        try:
            args.embed_dims = embed_dims
        except Exception:
            pass
        self.embed_dims = embed_dims

        self.encoder = DinoV3Encoder(args)
        self.decoder = ChangeDecoder(
            in_dim=self.embed_dims,
            channels=getattr(args, "decoder_channels", 128),
            has_sigmoid=True,
        )

    def update_bcd(self, x, y):
        features = self.encoder(x, y)
        change_features = [item[0] for item in features]
        return self.decoder(change_features)
