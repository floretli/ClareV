"""Train a single shared ClareV bag feature extractor (no fold-wise isolation).

Used for hyperparameter sweeps where we trade strict nested isolation for
fast iteration: one extractor is trained on an 80/10/10 (train/val/test)
repertoire split and reused by every downstream classifier fold via
3.0.clarev_classification.py --extractor-source whole.
"""

import argparse
import json
import os
import random
import sys

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

SCRIPT_PATH = os.path.abspath(__file__)
CODE_ROOT = os.path.dirname(SCRIPT_PATH)
if CODE_ROOT not in sys.path:
    sys.path.insert(0, CODE_ROOT)

from clarev.data_loaders.bag_data_loader import ClusterBagDataset
from clarev.models.contrastive_model import ContrastiveModel
from clarev.trainers.CL_trainer import CL_Trainer
from clarev.utils.utils import load_pk


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "2. Train a single shared CL extractor (80/10/10 repertoire split). "
            "Output: <trained_model_dir>/best_model.pth, consumed by 3.0 in "
            "--extractor-source whole mode."
        )
    )
    parser.add_argument("--bag-data-dir", type=str, required=True,
                        help="Directory containing superbags.pk, smp_list.pk, "
                             "smp_labels.pk, weights.pk")
    parser.add_argument("--trained-model-dir", type=str, required=True,
                        help="Output directory; best_model.pth is written here.")

    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for split and training.")
    parser.add_argument("--test-size", type=float, default=0.10,
                        help="Hold-out test fraction (default 0.10).")
    parser.add_argument("--val-size-of-remaining", type=float, default=0.11,
                        help="Validation fraction of the train+val pool "
                             "(default 0.11 ~ 10%% of total).")

    parser.add_argument("--tcr-emb-dim", type=int, default=120)
    parser.add_argument("--vgene-emb-dim", type=int, default=120)
    parser.add_argument("--subbag-size", type=int, default=50)
    parser.add_argument("--cl-num-epochs", type=int, default=40,
                        help="CL training epochs. Default 40 matches the paper.")
    parser.add_argument("--cl-batch-size", type=int, default=32)

    parser.add_argument("--loss-type", type=str, default="triplet",
                        choices=["triplet", "cosine"])
    parser.add_argument("--margin", type=float, default=0.1,
                        help="Triplet margin gamma. Default 0.1 (paper convention; "
                             "per-dataset overrides used in run_*.sh).")
    parser.add_argument("--alpha", type=float, default=1e-4,
                        help="Extra similarity nudge. Default 1e-4 matches "
                             "original training.")
    parser.add_argument("--learning-rate", type=float, default=1e-4)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_all_seeds(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.trained_model_dir, exist_ok=True)
    print(f"Startup | Device: {device} | Seed: {args.seed}", flush=True)
    print(f"Startup | Output dir: {args.trained_model_dir}", flush=True)

    superbags = load_pk(os.path.join(args.bag_data_dir, "superbags.pk"))
    smp_list = load_pk(os.path.join(args.bag_data_dir, "smp_list.pk"))
    weights = load_pk(os.path.join(args.bag_data_dir, "weights.pk"))
    print(f"Startup | Loaded {len(superbags)} repertoires from {args.bag_data_dir}",
          flush=True)

    # 80/10/10 repertoire-level split.
    train_bags, test_bags, train_w, test_w = train_test_split(
        superbags, weights, test_size=args.test_size, random_state=args.seed,
    )
    train_bags, eval_bags, train_w, eval_w = train_test_split(
        train_bags, train_w,
        test_size=args.val_size_of_remaining, random_state=args.seed,
    )
    print(
        f"Startup | Split sizes: train={len(train_bags)} | "
        f"val={len(eval_bags)} | test={len(test_bags)}",
        flush=True,
    )

    train_ds = ClusterBagDataset(train_bags, weight=train_w,
                                 subbag_size=args.subbag_size, data_repeat=3)
    eval_ds = ClusterBagDataset(eval_bags, weight=eval_w,
                                subbag_size=args.subbag_size)
    test_ds = ClusterBagDataset(test_bags, weight=test_w,
                                subbag_size=args.subbag_size)

    train_loader = DataLoader(train_ds, batch_size=args.cl_batch_size,
                              shuffle=True, drop_last=True)
    eval_loader = DataLoader(eval_ds, batch_size=args.cl_batch_size,
                             shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)

    print(
        f"CL | loss={args.loss_type} margin={args.margin} alpha={args.alpha} "
        f"epochs={args.cl_num_epochs} batch={args.cl_batch_size} "
        f"subbag_size={args.subbag_size}",
        flush=True,
    )

    model = ContrastiveModel(input_dim=args.tcr_emb_dim,
                             output_dim=args.vgene_emb_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    trainer = CL_Trainer(
        model, optimizer, device=device, save_dir=args.trained_model_dir,
        loss_type=args.loss_type, margin=args.margin, alpha=args.alpha,
    )

    _final_model, train_metrics = trainer.train(
        train_loader, eval_loader, epochs=args.cl_num_epochs,
    )
    trainer.record_metrics(train_metrics)

    best_model_path = os.path.join(args.trained_model_dir, "best_model.pth")
    best_model = ContrastiveModel(
        input_dim=args.tcr_emb_dim, output_dim=args.vgene_emb_dim,
    ).to(device)
    best_model.load_state_dict(torch.load(best_model_path, map_location=device))
    test_metrics = trainer.evaluate(best_model, test_loader)
    print(
        f"Eval | test_loss={test_metrics['loss']:.4f} | "
        f"test_acc={test_metrics['acc']:.4f}",
        flush=True,
    )

    meta = {
        "seed": int(args.seed),
        "test_size": float(args.test_size),
        "val_size_of_remaining": float(args.val_size_of_remaining),
        "split_sizes": {"train": len(train_bags),
                        "val": len(eval_bags),
                        "test": len(test_bags)},
        "tcr_emb_dim": int(args.tcr_emb_dim),
        "vgene_emb_dim": int(args.vgene_emb_dim),
        "subbag_size": int(args.subbag_size),
        "cl_num_epochs": int(args.cl_num_epochs),
        "cl_batch_size": int(args.cl_batch_size),
        "loss_type": str(args.loss_type),
        "margin": float(args.margin),
        "alpha": float(args.alpha),
        "learning_rate": float(args.learning_rate),
        "test_acc": float(test_metrics["acc"]),
        "test_loss": float(test_metrics["loss"]),
    }
    meta_path = os.path.join(args.trained_model_dir, "extractor_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved | best_model.pth + extractor_meta.json -> "
          f"{args.trained_model_dir}", flush=True)


if __name__ == "__main__":
    main()
