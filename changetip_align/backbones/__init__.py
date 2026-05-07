"""Alternative encoders for ChangeTIP-Align (DINOv2, SAM2).

Each backbone exposes a ``Trainer``-compatible module with:
  - ``embed_dims``: list[int] of stage channels (must match SpatialResidualChangeHead)
  - ``encoder(x, y)``: returns ``[[stage0_change], ..., [stage3_change]]``
  - ``decoder(features)``: returns base probability ``[B, 1, H, W]``

The list-of-list nesting mirrors Change3D's ``Trainer`` interface so that
``ChangeTIPAlignModel`` continues to work without modification: it consumes
``features = encoder(x, y)`` and ``change_features = [item[0] for item in features]``.
"""

from .dinov2_cd import DinoV2BCDTrainer
from .sam2_cd import SAM2BCDTrainer


def build_alt_backbone(name, args):
    name = name.lower()
    if name in ("dinov2", "dino", "dinov2_vits14", "dinov2_vitb14", "dinov2_vitl14"):
        return DinoV2BCDTrainer(args)
    if name in ("sam2", "sam", "sam2_hiera_t", "sam2_hiera_s", "sam2_hiera_b"):
        return SAM2BCDTrainer(args)
    raise ValueError(f"Unknown backbone: {name}")


__all__ = ["DinoV2BCDTrainer", "SAM2BCDTrainer", "build_alt_backbone"]
