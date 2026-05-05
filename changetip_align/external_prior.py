"""External prior encoder.

Provides frozen, pretrained image features that are independent of the
change-detection training set. The PRM consumes these features alongside
its own (training-set-derived) Change3D features so that the reward
signal contains structure that supervised loss cannot extract from GT
alone.

Default backbone is ImageNet-pretrained ResNet18 (lightweight, broadly
available via torchvision). DINOv2-small is also supported via torch.hub
when reachable.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ExternalPrior(nn.Module):
    def __init__(self, backbone: str = "resnet18"):
        super().__init__()
        self.backbone = backbone
        if backbone == "dinov2_vits14":
            try:
                self.encoder = torch.hub.load(
                    "facebookresearch/dinov2", "dinov2_vits14", trust_repo=True
                )
                self.feat_dim = 384
                self.patch_size = 14
            except Exception as exc:
                print(f"[ExternalPrior] DINOv2 unavailable ({exc}); falling back to resnet18.")
                self.backbone = "resnet18"
        if self.backbone == "resnet18":
            from torchvision.models import resnet18
            try:
                from torchvision.models import ResNet18_Weights
                m = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
            except Exception:
                m = resnet18(weights=None)
                print("[ExternalPrior] WARNING: resnet18 weights not loaded — prior is random.")
            self.layer1 = nn.Sequential(m.conv1, m.bn1, m.relu, m.maxpool, m.layer1)
            self.layer2 = m.layer2
            self.layer3 = m.layer3
            self.layer4 = m.layer4
            self.feat_dim = 64 + 128 + 256 + 512  # concat of 4 stages
            self.patch_size = 32
        for p in self.parameters():
            p.requires_grad = False
        self.eval()
        # ImageNet normalization stats
        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1), persistent=False
        )
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1), persistent=False
        )

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        # Inputs to the change-detection model are typically in [0, 1] or [-1, 1].
        # Heuristic: if min < -0.5 we assume [-1, 1]; rescale to [0, 1].
        if x.min() < -0.5:
            x = (x + 1.0) * 0.5
        return (x - self.mean) / self.std

    @torch.no_grad()
    def forward(self, pre: torch.Tensor, post: torch.Tensor) -> torch.Tensor:
        """Returns a single multi-scale feature tensor [B, feat_dim, H', W'].

        For BCD we concatenate features from the post image and the abs-diff
        between pre/post; the post image carries 'after' semantics, the diff
        captures 'where things changed'.
        """
        post_n = self._normalize(post)
        diff = (post - pre).abs()
        diff_n = self._normalize(diff)

        if self.backbone == "resnet18":
            feats_post = self._resnet_features(post_n)
            feats_diff = self._resnet_features(diff_n)
            # Use shallowest spatial resolution as common grid (1/4 of input).
            target_size = feats_post[0].shape[-2:]
            up = lambda f: F.interpolate(f, size=target_size, mode="bilinear", align_corners=False)
            stacked = torch.cat(
                [up(f) for f in feats_post] + [up(f) for f in feats_diff], dim=1
            )
            return stacked  # [B, 2*feat_dim_per_stage_sum, H/4, W/4]
        # dinov2 path
        return self._dinov2_features(post_n, diff_n)

    def _resnet_features(self, x):
        f1 = self.layer1(x)
        f2 = self.layer2(f1)
        f3 = self.layer3(f2)
        f4 = self.layer4(f3)
        return [f1, f2, f3, f4]

    def _dinov2_features(self, post_n, diff_n):
        outs = []
        for x in (post_n, diff_n):
            b, c, h, w = x.shape
            ph, pw = ((h + 13) // 14) * 14, ((w + 13) // 14) * 14
            if (ph, pw) != (h, w):
                x = F.interpolate(x, size=(ph, pw), mode="bilinear", align_corners=False)
            feats = self.encoder.forward_features(x)
            tokens = feats["x_norm_patchtokens"]  # [B, N, D]
            gh, gw = ph // 14, pw // 14
            outs.append(tokens.transpose(1, 2).reshape(b, self.feat_dim, gh, gw))
        return torch.cat(outs, dim=1)

    @property
    def out_channels(self) -> int:
        if self.backbone == "resnet18":
            return 2 * self.feat_dim
        return 2 * self.feat_dim
