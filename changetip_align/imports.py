import os
import sys
from argparse import Namespace


def add_change3d_root(change3d_root: str) -> str:
    """Add the original Change3D repository to sys.path."""
    root = os.path.abspath(change3d_root)
    if not os.path.isdir(root):
        raise FileNotFoundError(f"Change3D root not found: {root}")
    if root not in sys.path:
        sys.path.insert(0, root)
    return root


def make_change3d_args(args) -> Namespace:
    """Build the minimal argument namespace expected by Change3D Trainer/datasets."""
    return Namespace(
        dataset=getattr(args, "dataset", "WHU-CD"),
        task="BCD",
        file_root=getattr(args, "data_root", ""),
        num_perception_frame=1,
        num_class=1,
        in_height=getattr(args, "in_height", 256),
        in_width=getattr(args, "in_width", 256),
        pretrained=getattr(args, "pretrained", ""),
        use_moe=getattr(args, "use_moe", 0),
        moe_num_experts=getattr(args, "moe_num_experts", 8),
        moe_shared_expert=getattr(args, "moe_shared_expert", 0),
        moe_temp=getattr(args, "moe_temp", 1.0),
        moe_lb_w=getattr(args, "moe_lb_w", 0.01),
        num_domains=getattr(args, "num_domains", 0),
        enable_index_refine=getattr(args, "enable_index_refine", 0),
        index_threshold=getattr(args, "index_threshold", 0.2),
        index_topk_ratio=getattr(args, "index_topk_ratio", 0.02),
        index_dilate_kernel=getattr(args, "index_dilate_kernel", 3),
        index_head_channels=getattr(args, "index_head_channels", 32),
        refine_channels=getattr(args, "refine_channels", 32),
        bcd_feature_mode=getattr(args, "bcd_feature_mode", "pre"),
    )
