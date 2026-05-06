import os
import torch
import torch.nn as nn

from .heads import SpatialResidualChangeHead
from .imports import make_change3d_args


class ChangeTIPAlignModel(nn.Module):
    """Wrap Change3D with a TIPS-inspired residual dense head for BCD."""

    def __init__(
        self,
        base_model,
        head_channels=64,
        head_init_scale=0.1,
        head_zero_init: bool = True,
        head_mode: str = "residual",
    ):
        super().__init__()
        self.base_model = base_model
        self.spatial_head = SpatialResidualChangeHead(
            in_channels=tuple(base_model.embed_dims),
            channels=head_channels,
            init_scale=head_init_scale,
            zero_init=head_zero_init,
            mode=head_mode,
        )

    def update_bcd(self, x, y):
        features = self.base_model.encoder(x, y)
        change_features = [item[0] for item in features]
        base_prob = self.base_model.decoder(change_features)
        return self.spatial_head(change_features, base_prob)

    def update_bcd_full(self, x, y):
        """Forward pass exposing intermediate signals required by GRPO/PRM.

        Returns a dict with:
          base_prob:    [B, 1, H, W] frozen Change3D output
          base_logit:   [B, 1, H, W] logit(base_prob)
          delta_logit:  [B, 1, H, W] residual logit (with grad if head trainable)
          final_prob:   [B, 1, H, W] sigmoid(base_logit + scale * delta_logit)
          stages:       list of [B, C, H_l, W_l] from the spatial head
          residual_scale: scalar parameter (alpha)
        """
        features = self.base_model.encoder(x, y)
        change_features = [item[0] for item in features]
        base_prob = self.base_model.decoder(change_features)
        final_prob, base_logit, delta_logit, stages = self.spatial_head(
            change_features, base_prob, return_stages=True
        )
        return {
            "base_prob": base_prob,
            "base_logit": base_logit,
            "delta_logit": delta_logit,
            "final_prob": final_prob,
            "stages": stages,
            "residual_scale": self.spatial_head.residual_scale,
        }

    def update_scd(self, x, y):
        return self.base_model.update_scd(x, y)

    def update_bda(self, x, y):
        return self.base_model.update_bda(x, y)

    def update_cc(self, x, y):
        return self.base_model.update_cc(x, y)

    def forward(self, x, y):
        return self.update_bcd(x, y)

    @property
    def args(self):
        return self.base_model.args

    @property
    def embed_dims(self):
        return self.base_model.embed_dims


def _has_wrapper_keys(state):
    return any(key.startswith("base_model.") or key.startswith("spatial_head.") for key in state)


def load_state_flexible(model, ckpt_path: str):
    state = torch.load(ckpt_path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if isinstance(model, ChangeTIPAlignModel) and not _has_wrapper_keys(state):
        state = {f"base_model.{key}": value for key, value in state.items()}
        missing, unexpected = model.load_state_dict(state, strict=False)
        missing = [key for key in missing if not key.startswith("spatial_head.")]
        if missing or unexpected:
            raise RuntimeError(f"Checkpoint mismatch: missing={missing}, unexpected={unexpected}")
    else:
        model.load_state_dict(state, strict=True)
    return model


def build_change3d_model(args, ckpt_path=None, train=False):
    from model.trainer import Trainer

    cd_args = make_change3d_args(args)
    base_model = Trainer(cd_args)
    decoder_head = getattr(args, "decoder_head", "baseline")
    if decoder_head == "baseline":
        model = base_model
    elif decoder_head == "dpt_residual":
        model = ChangeTIPAlignModel(
            base_model,
            head_channels=getattr(args, "head_channels", 64),
            head_init_scale=getattr(args, "head_init_scale", 0.1),
            head_zero_init=bool(getattr(args, "head_zero_init", 1)),
            head_mode=getattr(args, "head_mode", "residual"),
        )
    else:
        raise ValueError(f"Unknown decoder_head: {decoder_head}")
    model = model.to(args.device)
    if ckpt_path:
        load_state_flexible(model, ckpt_path)
    model.train(train)
    return model


def set_trainable_policy(model, train_backbone_last=False, train_base_decoder=False):
    for param in model.parameters():
        param.requires_grad = False
    for name, param in model.named_parameters():
        if "spatial_head" in name:
            param.requires_grad = True
        if train_base_decoder and "decoder" in name:
            param.requires_grad = True
        if not isinstance(model, ChangeTIPAlignModel) and "decoder" in name:
            param.requires_grad = True
        if train_backbone_last and (
            "stage4" in name or "layer4" in name or "layers.3" in name or "blocks.3" in name
        ):
            param.requires_grad = True


class ModelEMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {
            name: param.detach().clone()
            for name, param in model.named_parameters()
            if param.requires_grad
        }
        self.backup = {}

    @torch.no_grad()
    def update(self, model):
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if name not in self.shadow:
                self.shadow[name] = param.detach().clone()
            else:
                self.shadow[name].mul_(self.decay).add_(param.detach(), alpha=1.0 - self.decay)

    @torch.no_grad()
    def apply_shadow(self, model):
        self.backup = {}
        for name, param in model.named_parameters():
            if name in self.shadow:
                self.backup[name] = param.detach().clone()
                param.copy_(self.shadow[name])

    @torch.no_grad()
    def restore(self, model):
        for name, param in model.named_parameters():
            if name in self.backup:
                param.copy_(self.backup[name])
        self.backup = {}


def save_model(path, model):
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    torch.save(model.state_dict(), path)
