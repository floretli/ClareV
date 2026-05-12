import argparse
import json
import os
import random
import sys
from typing import List, Optional, Set, Tuple

import numpy as np
import torch
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
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


def load_or_create_splits(
    split_file: str,
    sample_count: int,
    labels: np.ndarray,
    n_splits: int,
    random_seed: int,
    inner_n_splits: int = 5,
    force_regenerate: bool = False,
) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    os.makedirs(os.path.dirname(split_file), exist_ok=True)

    if os.path.exists(split_file) and not force_regenerate:
        split_data = np.load(split_file, allow_pickle=True)
        file_sample_count = int(split_data["sample_count"])
        if file_sample_count != sample_count:
            raise ValueError(
                f"Split file sample_count mismatch: file={file_sample_count}, current={sample_count}"
            )
        train_splits = split_data["train_splits"]
        val_splits = split_data["val_splits"]
        test_splits = split_data["test_splits"]
        return [
            (
                np.array(train_idx, dtype=int),
                np.array(val_idx, dtype=int),
                np.array(test_idx, dtype=int),
            )
            for train_idx, val_idx, test_idx in zip(train_splits, val_splits, test_splits)
        ]

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
    folds = list(kf.split(np.arange(sample_count)))
    prepared_folds: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for fold_id, (outer_train_idx, test_idx) in enumerate(folds):
        train_labels = labels[outer_train_idx]
        unique_labels = np.unique(train_labels.astype(int))
        if len(unique_labels) >= 2 and np.min(np.bincount(train_labels.astype(int))) >= inner_n_splits:
            inner_splitter = StratifiedKFold(
                n_splits=inner_n_splits,
                shuffle=True,
                random_state=random_seed + fold_id,
            )
            train_rel, val_rel = next(
                inner_splitter.split(np.zeros(len(outer_train_idx)), train_labels)
            )
            train_idx = outer_train_idx[train_rel]
            val_idx = outer_train_idx[val_rel]
        else:
            rng = np.random.default_rng(random_seed + fold_id)
            shuffled = outer_train_idx.copy()
            rng.shuffle(shuffled)
            val_size = max(1, int(round(len(shuffled) / inner_n_splits)))
            val_idx = shuffled[:val_size]
            train_idx = shuffled[val_size:]

        prepared_folds.append(
            (
                np.array(train_idx, dtype=int),
                np.array(val_idx, dtype=int),
                np.array(test_idx, dtype=int),
            )
        )

    train_splits = np.array([fold[0] for fold in prepared_folds], dtype=object)
    val_splits = np.array([fold[1] for fold in prepared_folds], dtype=object)
    test_splits = np.array([fold[2] for fold in prepared_folds], dtype=object)
    np.savez(
        split_file,
        train_splits=train_splits,
        val_splits=val_splits,
        test_splits=test_splits,
        sample_count=sample_count,
        n_splits=n_splits,
        random_seed=random_seed,
        inner_n_splits=inner_n_splits,
    )
    return prepared_folds


def parse_fold_indices(fold_indices_text: str, n_splits: int) -> Optional[Set[int]]:
    if fold_indices_text.lower() == "all":
        return None
    selected_ids = set()
    for token in fold_indices_text.split(","):
        token = token.strip()
        if not token:
            continue
        fold_id = int(token)
        if fold_id < 0 or fold_id >= n_splits:
            raise ValueError(f"Invalid fold id {fold_id}, expected in [0, {n_splits - 1}]")
        selected_ids.add(fold_id)
    if not selected_ids:
        raise ValueError("No valid fold ids provided in --fold-indices")
    return selected_ids


def train_fold_extractor(
    fold_id: int,
    pool_indices: np.ndarray,
    superbags: list,
    weights: list,
    model_root: str,
    tcr_emb_dim: int,
    vgene_emb_dim: int,
    subbag_size: int,
    cl_batch_size: int,
    cl_num_epochs: int,
    extractor_val_fraction: float,
    seed: int,
    skip_existing: bool,
    resume_from_existing: bool,
    device: torch.device,
    loss_type: str = "triplet",
    margin: float = 0.1,
    alpha: float = 1e-4,
    learning_rate: float = 1e-4,
) -> str:
    fold_dir = os.path.join(model_root, f"fold_{fold_id}")
    os.makedirs(fold_dir, exist_ok=True)
    model_path = os.path.join(fold_dir, "best_model.pth")

    if skip_existing and os.path.exists(model_path):
        print(f"Extractor | Fold {fold_id} | Reuse existing: {model_path}", flush=True)
        return model_path

    fold_seed = seed + fold_id * 1000
    set_all_seeds(fold_seed)

    pool_bags = [superbags[int(i)] for i in pool_indices]
    pool_weights = [weights[int(i)] for i in pool_indices]
    tr_bags, ev_bags, tr_w, ev_w = train_test_split(
        pool_bags,
        pool_weights,
        test_size=extractor_val_fraction,
        random_state=fold_seed,
        shuffle=True,
    )

    train_dataset = ClusterBagDataset(tr_bags, weight=tr_w, subbag_size=subbag_size, data_repeat=3)
    eval_dataset = ClusterBagDataset(ev_bags, weight=ev_w, subbag_size=subbag_size)
    train_loader = DataLoader(train_dataset, batch_size=cl_batch_size, shuffle=True, drop_last=True)
    eval_loader = DataLoader(eval_dataset, batch_size=cl_batch_size, shuffle=False)

    model = ContrastiveModel(input_dim=tcr_emb_dim, output_dim=vgene_emb_dim)
    if resume_from_existing and os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Extractor | Fold {fold_id} | Resume from: {model_path}", flush=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    trainer = CL_Trainer(
        model,
        optimizer,
        device=device,
        save_dir=fold_dir,
        loss_type=loss_type,
        margin=margin,
        alpha=alpha,
    )

    print(
        (
            f"Extractor | Fold {fold_id} | Pool samples: {len(pool_bags)} | "
            f"Train/Eval pool split: {len(tr_bags)}/{len(ev_bags)} | Epochs: {cl_num_epochs}"
        ),
        flush=True,
    )

    _final_model, train_metrics = trainer.train(train_loader, eval_loader, epochs=cl_num_epochs)
    trainer.record_metrics(train_metrics)

    meta = {
        "fold_id": fold_id,
        "seed": fold_seed,
        "pool_size": int(len(pool_bags)),
        "train_pool_size": int(len(tr_bags)),
        "eval_pool_size": int(len(ev_bags)),
        "extractor_val_fraction": float(extractor_val_fraction),
        "cl_num_epochs": int(cl_num_epochs),
        "subbag_size": int(subbag_size),
        "loss_type": str(loss_type),
        "margin": float(margin),
        "alpha": float(alpha),
        "pool_indices": pool_indices.tolist(),
        "resumed_from_existing": bool(resume_from_existing),
    }
    with open(os.path.join(fold_dir, "extractor_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Extractor best model not saved for fold {fold_id}: {model_path}")

    print(f"Extractor | Fold {fold_id} | Saved: {model_path}", flush=True)
    return model_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="2.1 Fold-wise CL extractor training.")
    parser.add_argument("--bag-data-dir", type=str, required=True)
    parser.add_argument("--trained-model-dir", type=str, required=True)
    parser.add_argument("--split-file", type=str, required=True)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--split-seed", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--fold-indices", type=str, default="all")
    parser.add_argument("--force-regenerate-splits", action="store_true")

    parser.add_argument("--tcr-emb-dim", type=int, default=120)
    parser.add_argument("--vgene-emb-dim", type=int, default=120)
    parser.add_argument("--subbag-size", type=int, default=50)

    parser.add_argument("--cl-num-epochs", type=int, default=40,
                        help="CL training epochs. Default 40 matches the paper.")
    parser.add_argument("--cl-batch-size", type=int, default=32)
    parser.add_argument("--extractor-val-fraction", type=float, default=0.1)
    parser.add_argument("--loss-type", type=str, default="triplet", choices=["triplet", "cosine"])
    parser.add_argument("--margin", type=float, default=0.1,
                        help="Triplet margin gamma. Default 0.1 (paper convention; "
                             "per-dataset overrides used in run_*.sh).")
    parser.add_argument("--alpha", type=float, default=1e-4,
                        help="Extra similarity nudge. Default 1e-4 matches original training.")
    parser.add_argument("--learning-rate", type=float, default=1e-4,
                        help="Adam learning rate for the contrastive extractor. Default 1e-4 "
                             "matches original training.")
    parser.add_argument("--extractor-use-train-only", action="store_true")
    parser.add_argument("--skip-existing-fold-models", action="store_true")
    parser.add_argument("--resume-from-existing", action="store_true")
    parser.add_argument("--inner-n-splits", type=int, default=5)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_all_seeds(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Startup | Device: {device} | Seed: {args.seed}", flush=True)

    required_paths = [
        args.bag_data_dir,
        os.path.join(args.bag_data_dir, "superbags.pk"),
        os.path.join(args.bag_data_dir, "smp_list.pk"),
        os.path.join(args.bag_data_dir, "smp_labels.pk"),
        os.path.join(args.bag_data_dir, "weights.pk"),
    ]
    missing_paths = [p for p in required_paths if not os.path.exists(p)]
    if missing_paths:
        raise FileNotFoundError("Missing required paths:\n" + "\n".join(missing_paths))
    os.makedirs(args.trained_model_dir, exist_ok=True)

    superbags = load_pk(os.path.join(args.bag_data_dir, "superbags.pk"))
    smp_list = load_pk(os.path.join(args.bag_data_dir, "smp_list.pk"))
    smp_labels = np.asarray(load_pk(os.path.join(args.bag_data_dir, "smp_labels.pk")), dtype=int)
    weights = load_pk(os.path.join(args.bag_data_dir, "weights.pk"))

    folds = load_or_create_splits(
        split_file=args.split_file,
        sample_count=len(smp_list),
        labels=smp_labels,
        n_splits=args.n_splits,
        random_seed=args.split_seed,
        inner_n_splits=args.inner_n_splits,
        force_regenerate=args.force_regenerate_splits,
    )
    selected_fold_ids = parse_fold_indices(args.fold_indices, args.n_splits)

    for fold_id, fold in enumerate(folds):
        if selected_fold_ids is not None and fold_id not in selected_fold_ids:
            continue

        train_idx = np.array(fold[0], dtype=int)
        val_idx = np.array(fold[1], dtype=int)
        test_idx = np.array(fold[2], dtype=int)

        if args.extractor_use_train_only:
            pool_idx = train_idx
            pool_name = "train_only"
        else:
            pool_idx = np.concatenate([train_idx, val_idx])
            pool_name = "train_plus_val"

        overlap = set(pool_idx.tolist()) & set(test_idx.tolist())
        if overlap:
            raise ValueError(
                f"Extractor pool overlaps with outer test fold {fold_id}: {len(overlap)} samples"
            )

        print(
            (
                f"Extractor | Fold {fold_id} | Pool mode: {pool_name} | "
                f"Pool size: {len(pool_idx)} | Outer test size: {len(test_idx)}"
            ),
            flush=True,
        )

        train_fold_extractor(
            fold_id=fold_id,
            pool_indices=pool_idx,
            superbags=superbags,
            weights=weights,
            model_root=args.trained_model_dir,
            tcr_emb_dim=args.tcr_emb_dim,
            vgene_emb_dim=args.vgene_emb_dim,
            subbag_size=args.subbag_size,
            cl_batch_size=args.cl_batch_size,
            cl_num_epochs=args.cl_num_epochs,
            extractor_val_fraction=args.extractor_val_fraction,
            seed=args.seed,
            skip_existing=args.skip_existing_fold_models,
            resume_from_existing=args.resume_from_existing,
            device=device,
            loss_type=args.loss_type,
            margin=args.margin,
            alpha=args.alpha,
            learning_rate=args.learning_rate,
        )


if __name__ == "__main__":
    main()
