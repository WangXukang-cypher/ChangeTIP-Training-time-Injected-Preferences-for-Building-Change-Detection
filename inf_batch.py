import argparse
import os
import sys
from types import SimpleNamespace

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import shutil


ROOT = os.path.abspath(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from changetip_align.candidate import as_prob
from changetip_align.imports import add_change3d_root, make_change3d_args
from changetip_align.models import build_change3d_model


def parse_list(text):
    return [item.strip() for item in text.split(",") if item.strip()]


class IndexedDataset(torch.utils.data.Dataset):
    def __init__(self, base_dataset):
        self.base = base_dataset

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        img, target = self.base[index]
        return img, target, index


def build_split_loader(args, split):
    import data.dataset as RSDataset
    import data.transforms as RSTransforms

    cd_args = make_change3d_args(args)
    _, val_tf = RSTransforms.BCDTransforms.get_transform_pipelines(cd_args)
    base = RSDataset.BCDDataset(file_root=args.data_root, split=split, transform=val_tf)
    dataset = IndexedDataset(base)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    return base, loader


@torch.no_grad()
def predict_prob(args, model, pre, post):
    prob = as_prob(model.update_bcd(pre, post))
    if not args.eval_tta:
        return prob

    outputs = [prob]
    out_h = as_prob(model.update_bcd(torch.flip(pre, [3]), torch.flip(post, [3])))
    outputs.append(torch.flip(out_h, [3]))
    out_v = as_prob(model.update_bcd(torch.flip(pre, [2]), torch.flip(post, [2])))
    outputs.append(torch.flip(out_v, [2]))
    out_hv = as_prob(model.update_bcd(torch.flip(pre, [2, 3]), torch.flip(post, [2, 3])))
    outputs.append(torch.flip(out_hv, [2, 3]))
    return torch.stack(outputs, dim=0).mean(dim=0)


def save_uint8_png(path, array):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.fromarray(array.astype(np.uint8)).save(path)


def copy_image(src, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy2(src, dst)


def resize_uint8(array, size_hw):
    image = Image.fromarray(array.astype(np.uint8))
    return np.asarray(image.resize((size_hw[1], size_hw[0]), resample=Image.NEAREST))


def original_hw(base_dataset, index):
    label_path = base_dataset.label_change[index]
    with Image.open(label_path) as image:
        return image.size[1], image.size[0]


@torch.no_grad()
def infer_split(args, model, split):
    from utils.metric_tool import ConfuseMatrixMeter

    base_dataset, loader = build_split_loader(args, split)
    if args.flat_export:
        t1_dir = os.path.join(args.output_root, "T1")
        t2_dir = os.path.join(args.output_root, "T2")
        pred_dir = os.path.join(args.output_root, "label")
        prob_dir = os.path.join(args.output_root, "prob")
    else:
        pred_dir = os.path.join(args.output_root, split, "pred")
        prob_dir = os.path.join(args.output_root, split, "prob")
    os.makedirs(pred_dir, exist_ok=True)
    if args.flat_export:
        os.makedirs(t1_dir, exist_ok=True)
        os.makedirs(t2_dir, exist_ok=True)
    if args.save_prob != "none":
        os.makedirs(prob_dir, exist_ok=True)

    meter = ConfuseMatrixMeter(n_class=2)
    model.eval()

    for img, target, indices in loader:
        img = img.to(args.device, non_blocking=True).float()
        pre = img[:, 0:3]
        post = img[:, 3:6]
        target = target.to(args.device, non_blocking=True).float()
        target = (target > 0.5).float()

        prob = predict_prob(args, model, pre, post)
        pred = (prob >= args.threshold).long()
        meter.update_cm(pr=pred.cpu().numpy(), gt=target.cpu().numpy())

        prob_np = prob[:, 0].detach().cpu().numpy()
        pred_np = pred[:, 0].detach().cpu().numpy().astype(np.uint8) * 255

        for item_idx, dataset_idx in enumerate(indices.tolist()):
            src_name = base_dataset.file_list[dataset_idx]
            stem = os.path.splitext(src_name)[0]
            if args.flat_export and args.prefix_split:
                stem = f"{split}_{stem}"
            name = stem + ".png"
            out_pred = pred_np[item_idx]
            out_prob = np.clip(prob_np[item_idx] * 255.0, 0, 255).astype(np.uint8)

            if args.restore_size:
                hw = original_hw(base_dataset, dataset_idx)
                out_pred = resize_uint8(out_pred, hw)
                out_prob = resize_uint8(out_prob, hw)

            if args.flat_export:
                ext = os.path.splitext(src_name)[1]
                image_name = stem + ext
                copy_image(base_dataset.pre_images[dataset_idx], os.path.join(t1_dir, image_name))
                copy_image(base_dataset.post_images[dataset_idx], os.path.join(t2_dir, image_name))
            save_uint8_png(os.path.join(pred_dir, name), out_pred)
            if args.save_prob == "png":
                save_uint8_png(os.path.join(prob_dir, name), out_prob)
            elif args.save_prob == "npy":
                os.makedirs(prob_dir, exist_ok=True)
                np.save(os.path.join(prob_dir, name.replace(".png", ".npy")), prob_np[item_idx])

    scores = meter.get_scores()
    print(
        f"[{split}] F1={scores['F1']:.4f} IoU={scores['IoU']:.4f} "
        f"P={scores['precision']:.4f} R={scores['recall']:.4f} "
        f"saved={pred_dir}"
    )
    return scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--change3d_root", default="..")
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--pretrained", default="")
    parser.add_argument("--dataset", default="WHU-CD")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--in_height", type=int, default=256)
    parser.add_argument("--in_width", type=int, default=256)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--splits", default="train,val,test")
    parser.add_argument("--eval_tta", type=int, default=0)
    parser.add_argument("--restore_size", type=int, default=0)
    parser.add_argument("--save_prob", default="none", choices=["none", "png", "npy"])
    parser.add_argument("--flat_export", type=int, default=1)
    parser.add_argument("--prefix_split", type=int, default=1)
    parser.add_argument("--use_moe", type=int, default=0)
    parser.add_argument("--bcd_feature_mode", default="pre")
    parser.add_argument("--decoder_head", default="dpt_residual", choices=["baseline", "dpt_residual"])
    parser.add_argument("--head_channels", type=int, default=64)
    parser.add_argument("--head_init_scale", type=float, default=0.1)
    parser.add_argument("--enable_index_refine", type=int, default=0)
    parser.add_argument("--index_threshold", type=float, default=0.2)
    parser.add_argument("--index_topk_ratio", type=float, default=0.02)
    parser.add_argument("--index_dilate_kernel", type=int, default=3)
    args = parser.parse_args()
    args.eval_tta = bool(args.eval_tta)
    args.restore_size = bool(args.restore_size)
    args.flat_export = bool(args.flat_export)
    args.prefix_split = bool(args.prefix_split)

    add_change3d_root(args.change3d_root)
    model = build_change3d_model(args, args.ckpt, train=False)

    all_scores = {}
    for split in parse_list(args.splits):
        all_scores[split] = infer_split(args, model, split)

    os.makedirs(args.output_root, exist_ok=True)
    summary_path = os.path.join(args.output_root, "metrics_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as file:
        for split, scores in all_scores.items():
            file.write(
                f"{split}\tF1={scores['F1']:.6f}\tIoU={scores['IoU']:.6f}\t"
                f"P={scores['precision']:.6f}\tR={scores['recall']:.6f}\n"
            )
    print(f"[summary] {summary_path}")


if __name__ == "__main__":
    main()
