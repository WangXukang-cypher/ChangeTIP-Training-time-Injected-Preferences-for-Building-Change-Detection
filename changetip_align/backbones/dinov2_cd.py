"""DINOv2 ViT encoder + DPT-reassemble + cross-temporal fusion for BCD.

Why DINOv2:
    DINOv2 features are dense, semantic, and patch-level — exactly the regime
    where post-training (PRM + GRPO) has room to push the residual head into
    the "change-vs-non-change" boundary. The encoder is kept frozen by default
    (pretrained features serve as the prior) and only the reassemble blocks +
    cross-time fusion + decoder + residual head are trained.

Pipeline:
    pre, post  ─ resize 256→224 ─►  ViT-S/14  (frozen)
                                   │
                                   ├─ tap layers [2, 5, 8, 11]
                                   │
                                   ▼
                    DPT Reassemble (per-stage upsample)
                                   │
                                   ▼  4 stages at [1/4, 1/8, 1/16, 1/32]
                          CrossTimeFusion (per stage)
                                   │
                                   ▼
                            ChangeDecoder ────► base_prob

Output of ``encoder(x, y)`` is a list-of-list to mimic Change3D's interface.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .decoder import ChangeDecoder, ConvBNReLU, CrossTimeFusion, ResUnit


_DINO_SPECS = {
    # arch_name: (model_id, embed_dim, taps, # blocks)
    "vits14": ("dinov2_vits14", 384, [2, 5, 8, 11], 12),
    "vitb14": ("dinov2_vitb14", 768, [2, 5, 8, 11], 12),
    "vitl14": ("dinov2_vitl14", 1024, [4, 11, 17, 23], 24),
}

# Per-stage spatial scale factors with respect to the patch grid (which is at
# input/14 resolution). To map to a feature pyramid at [1/4, 1/8, 1/16, 1/32]
# of the *original input*, we go via an internal 224x224 forward (16x16 grid)
# and bilinear-upsample at the stage outputs to match the desired input scale.
_STAGE_INPUT_DIVISORS = (4, 8, 16, 32)


class DPTReassemble(nn.Module):
    """Convert ViT [B, N, D] tokens at a fixed [Hp, Wp] grid into [B, C, Ht, Wt].

    DPT does this by reshape → 1x1 project → ConvTranspose2d (for fine stages)
    or stride conv (for coarse stages). We use bilinear resize at the end to
    reach the requested target resolution exactly, which is more flexible when
    the input H/W is not an integer multiple of the stage divisor.
    """

    def __init__(self, in_dim, out_dim, scale_factor):
        super().__init__()
        self.proj = nn.Conv2d(in_dim, out_dim, 1)
        if scale_factor > 1:
            self.resize = nn.ConvTranspose2d(
                out_dim, out_dim,
                kernel_size=int(scale_factor), stride=int(scale_factor),
            )
        elif scale_factor < 1:
            inv = int(round(1.0 / scale_factor))
            self.resize = nn.Conv2d(out_dim, out_dim, kernel_size=inv, stride=inv)
        else:
            self.resize = nn.Identity()

    def forward(self, tokens, hp, wp):
        # tokens: [B, N, D]   N = hp * wp (CLS already dropped)
        b, n, d = tokens.shape
        assert n == hp * wp, f"token count {n} != hp*wp {hp * wp}"
        x = tokens.transpose(1, 2).reshape(b, d, hp, wp).contiguous()
        x = self.proj(x)
        x = self.resize(x)
        return x


class DinoV2Encoder(nn.Module):
    """Frozen DINOv2 + per-stage reassemble + cross-temporal fusion.

    The internal forward size is chosen as the largest multiple of 14 that
    fits inside the input resolution (default 252 for 256-pixel inputs →
    18x18 patch grid). DINOv2 natively supports arbitrary multiples of 14
    via positional-embedding interpolation, so accuracy is preserved.
    """

    PATCH = 14

    def __init__(self, args):
        super().__init__()
        self.args = args
        arch = getattr(args, "dino_arch", "vits14")
        if arch not in _DINO_SPECS:
            raise ValueError(f"Unknown dino_arch={arch}; choose from {list(_DINO_SPECS)}")
        model_id, dim, taps, _depth = _DINO_SPECS[arch]
        self.dim = dim
        self.taps = taps

        # Pick the largest multiple of PATCH that fits inside the user's input.
        # Override with --dino_input_size to fix a specific value (must be
        # divisible by 14). For 256-pixel inputs this lands at 252 = 18*14,
        # giving a 18x18 token grid (vs 16x16 at 224, ~27% more pixels).
        in_h = getattr(args, "in_height", 256)
        in_w = getattr(args, "in_width", 256)
        side = min(in_h, in_w)
        forced = int(getattr(args, "dino_input_size", 0) or 0)
        if forced > 0:
            if forced % self.PATCH != 0:
                raise ValueError(
                    f"--dino_input_size={forced} is not a multiple of {self.PATCH}."
                )
            self.internal_size = forced
        else:
            self.internal_size = max(self.PATCH, (side // self.PATCH) * self.PATCH)
        self.grid = self.internal_size // self.PATCH
        print(f"[DinoV2Encoder] arch={arch} internal_size={self.internal_size} "
              f"patch_grid={self.grid}x{self.grid}")

        try:
            self.dino = torch.hub.load(
                "facebookresearch/dinov2", model_id, pretrained=True, trust_repo=True,
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to load DINOv2 '{model_id}' via torch.hub. "
                f"Make sure the cache is populated or you have internet access. "
                f"Original error: {e}"
            )
        for p in self.dino.parameters():
            p.requires_grad = False
        self.dino.eval()

        # Reassemble: at internal 16x16 grid, scale factors [4,2,1,0.5] give
        # spatial sizes [64,32,16,8]. We then bilinear-resize to the target
        # input-relative scale.
        scales = [4, 2, 1, 0.5]
        embed_dims = list(args.embed_dims) if hasattr(args, "embed_dims") else None
        if embed_dims is None:
            embed_dims = [96, 192, 384, 768]
        self.embed_dims = embed_dims

        self.reassembles = nn.ModuleList([
            DPTReassemble(self.dim, embed_dims[i], scales[i]) for i in range(4)
        ])
        self.fuses = nn.ModuleList([
            CrossTimeFusion(embed_dims[i]) for i in range(4)
        ])

    @torch.no_grad()
    def _tokens(self, x):
        """Return tuple of [B, N, D] tokens at each tap (CLS dropped)."""
        if x.shape[-2:] != (self.internal_size, self.internal_size):
            x = F.interpolate(x, size=(self.internal_size, self.internal_size),
                              mode="bilinear", align_corners=False)
        # get_intermediate_layers returns features in order of self.taps when
        # ``n`` is a list; ``return_class_token=False`` strips the CLS.
        outs = self.dino.get_intermediate_layers(
            x, n=self.taps, return_class_token=False, norm=True,
        )
        return outs

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
            out.append([change])  # list-of-list to mimic Change3D
        return out


class DinoV2BCDTrainer(nn.Module):
    """DINOv2-based BCD module with the same interface as Change3D's Trainer."""

    def __init__(self, args):
        super().__init__()
        self.args = args
        embed_dims = getattr(args, "embed_dims", None)
        if embed_dims is None:
            embed_dims = [96, 192, 384, 768]
        # Inject so encoder picks up the same dims.
        if not hasattr(args, "embed_dims"):
            try:
                args.embed_dims = embed_dims
            except Exception:
                pass
        self.embed_dims = embed_dims

        self.encoder = DinoV2Encoder(args)
        self.decoder = ChangeDecoder(
            in_dim=self.embed_dims,
            channels=getattr(args, "decoder_channels", 128),
            has_sigmoid=True,
        )

    def update_bcd(self, x, y):
        features = self.encoder(x, y)
        change_features = [item[0] for item in features]
        return self.decoder(change_features)

    # The Change3D Trainer also exposes update_scd / update_bda / update_cc.
    # We do not need these for BCD post-training, so skip.
