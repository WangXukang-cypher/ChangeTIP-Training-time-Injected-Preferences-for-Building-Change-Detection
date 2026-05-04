import torch

from .imports import make_change3d_args


def build_loader(args, split: str, train: bool = False):
    """Build a BCD loader through the original Change3D data pipeline."""
    import data.dataset as RSDataset
    import data.transforms as RSTransforms

    cd_args = make_change3d_args(args)
    train_tf, val_tf = RSTransforms.BCDTransforms.get_transform_pipelines(cd_args)
    transform = train_tf if train else val_tf
    dataset = RSDataset.BCDDataset(file_root=args.data_root, split=split, transform=transform)
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
