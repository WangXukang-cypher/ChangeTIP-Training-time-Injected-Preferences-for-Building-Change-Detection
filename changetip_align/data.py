import os

import numpy as np
import torch
from skimage import io

from .imports import make_change3d_args


class MaskRatioFilteredDataset(torch.utils.data.Dataset):
    """Wrap a BCDDataset and keep only samples whose change-mask ratio >= min_ratio.

    Mask ratio = (positive pixels) / (total pixels). The label files are read
    once at init to compute ratios; the indices that survive filtering are
    stored and used by __getitem__. Only intended for the train split — val
    and test are passed through unchanged in build_loader.
    """

    def __init__(self, base_dataset, min_ratio: float, verbose: bool = True):
        self.base = base_dataset
        self.min_ratio = float(min_ratio)
        self.indices = self._compute_indices(verbose=verbose)

    def _compute_indices(self, verbose: bool):
        keep = []
        ratios = []
        label_paths = self.base.label_change
        for i, label_path in enumerate(label_paths):
            label = io.imread(label_path, as_gray=True)
            ratio = float((label > 0).sum()) / float(label.size)
            ratios.append(ratio)
            if ratio >= self.min_ratio:
                keep.append(i)
        if verbose:
            total = len(label_paths)
            kept = len(keep)
            mean_r = float(np.mean(ratios)) if ratios else 0.0
            print(
                f"[MaskRatioFilter] min_ratio={self.min_ratio:.4f} "
                f"kept {kept}/{total} ({100.0 * kept / max(1, total):.1f}%) "
                f"mean_ratio_all={mean_r:.4f}"
            )
        return keep

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.base[self.indices[idx]]


def build_loader(args, split: str, train: bool = False, min_ratio_override=None):
    """Build a BCD loader through the original Change3D data pipeline.

    If args.min_change_ratio > 0 and split == 'train', filter out samples whose
    change-mask ratio is below the threshold. val/test are never filtered.

    ``min_ratio_override`` (if not None) overrides args.min_change_ratio for
    this loader only — used by train_baseline.py to build two train loaders
    (full + filtered) and switch between them across the warmup/reward-active
    phases.
    """
    import data.dataset as RSDataset
    import data.transforms as RSTransforms

    cd_args = make_change3d_args(args)
    train_tf, val_tf = RSTransforms.BCDTransforms.get_transform_pipelines(cd_args)
    transform = train_tf if train else val_tf
    dataset = RSDataset.BCDDataset(file_root=args.data_root, split=split, transform=transform)

    if min_ratio_override is not None:
        min_ratio = float(min_ratio_override)
    else:
        min_ratio = float(getattr(args, "min_change_ratio", 0.0))
    if split == "train" and min_ratio > 0.0:
        dataset = MaskRatioFilteredDataset(dataset, min_ratio=min_ratio, verbose=True)

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=train,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=train,
    )


def split_batch(batch, device="cuda"):
    img, target = batch[:2]
    pre = img[:, 0:3].to(device, non_blocking=True).float()
    post = img[:, 3:6].to(device, non_blocking=True).float()
    target = target.to(device, non_blocking=True).float()
    target = (target > 0.5).float()
    return pre, post, target
