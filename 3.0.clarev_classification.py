import argparse
import os
import random
import sys
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import torch
from sklearn.model_selection import KFold, StratifiedKFold


SCRIPT_PATH = os.path.abspath(__file__)
CODE_ROOT = os.path.dirname(SCRIPT_PATH)
if CODE_ROOT not in sys.path:
    sys.path.insert(0, CODE_ROOT)

from clarev.data_loaders.bag_data_loader import RepertoireDataset
from clarev.models.contrastive_model import ContrastiveModel
from clarev.utils.utils import load_pk
from clarev.trainers.clarev_classifier import (
    bagdata_to_vfreq,
    run_clarev_classification,
)


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
        if {"train_splits", "val_splits", "test_splits"}.issubset(split_data.files):
            train_splits = split_data["train_splits"]
            val_splits = split_data["val_splits"]
            test_splits = split_data["test_splits"]
            return [
                (
                    np.array(train_idx, dtype=int),
                    np.array(val_idx, dtype=int),
                    np.array(test_idx, dtype=int),
                )
                for train_idx, val_idx, test_idx in zip(
                    train_splits, val_splits, test_splits
                )
            ]
        train_splits = split_data["train_splits"]
        test_splits = split_data["test_splits"]
        return [
            (np.array(train_idx, dtype=int), np.array([], dtype=int), np.array(test_idx, dtype=int))
            for train_idx, test_idx in zip(train_splits, test_splits)
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


def collect_fold_model_paths(
    model_root: str,
    n_splits: int,
    selected_fold_ids: Optional[Set[int]] = None,
) -> Dict[int, str]:
    fold_model_state_paths: Dict[int, str] = {}
    for fold_id in range(n_splits):
        if selected_fold_ids is not None and fold_id not in selected_fold_ids:
            continue
        model_path = os.path.join(model_root, f"fold_{fold_id}", "best_model.pth")
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Missing fold-specific extractor: {model_path}. "
                "Please run 2.1.train_foldwise_bag_feature_extractor.py first."
            )
        fold_model_state_paths[fold_id] = model_path
    return fold_model_state_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="3.0 ClareV repertoire classification (read pretrained extractors only)."
    )
    parser.add_argument(
        "--bag-data-dir",
        type=str,
        default=os.path.join(CODE_ROOT, "data", "processed_data", "Emerson2017_vgene"),
    )
    parser.add_argument(
        "--trained-model-dir",
        type=str,
        default=os.path.join(CODE_ROOT, "trained_models", "Emerson2017_vgene_weight", "fold_wise"),
        help=(
            "For whole mode: directory containing best_model.pth; "
            "for foldwise mode: directory containing fold_i/best_model.pth"
        ),
    )
    parser.add_argument(
        "--whole-model-path",
        type=str,
        default=None,
        help="Optional explicit model path for whole extractor mode.",
    )
    parser.add_argument(
        "--extractor-source",
        type=str,
        choices=["whole", "foldwise"],
        default="whole",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.path.join(
            CODE_ROOT, "exp_output", "repertoire_classification", "clarev_vgene_weight"
        ),
    )
    parser.add_argument(
        "--split-file",
        type=str,
        default=os.path.join(
            CODE_ROOT, "data", "splits", "emerson2017_nested_kfold5.npz",
        ),
    )
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--split-seed", type=int, default=1)
    parser.add_argument("--num-epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--test-batch-size", type=int, default=1)
    parser.add_argument("--subbag-size", type=int, default=50)
    parser.add_argument("--vgene-emb-dim", type=int, default=120)
    parser.add_argument("--tcr-emb-dim", type=int, default=120)
    parser.add_argument("--force-regenerate-splits", action="store_true")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--nested-mode", action="store_true", default=True)
    parser.add_argument("--no-nested-mode", dest="nested_mode", action="store_false")
    parser.add_argument("--inner-n-splits", type=int, default=5)
    parser.add_argument("--early-stopping-patience", type=int, default=5)
    parser.add_argument(
        "--methods",
        type=str,
        choices=["all", "vbag_vfreq", "vbag_only"],
        default="all",
    )
    parser.add_argument(
        "--vfreq-fusion-backend",
        type=str,
        choices=["mlp", "rf"],
        default="rf",
        help="Backend for the V-usage branch in the fusion model. "
             "'rf' (default) uses the RFFusion classifier (random-forest "
             "probability fed into a learnable fusion head with an RF-logit "
             "residual skip). 'mlp' falls back to the legacy MLP fusion.",
    )
    parser.add_argument("--rf-n-estimators", type=int, default=300)
    parser.add_argument("--rf-seed", type=int, default=1)
    parser.add_argument("--clf-lr", type=float, default=1e-4,
                        help="Learning rate for the downstream RFFusion classifier (AdamW). "
                             "Default 1e-4 matches original training.")
    parser.add_argument("--clf-weight-decay", type=float, default=1e-5,
                        help="Weight decay for the downstream RFFusion classifier (AdamW). "
                             "Default 1e-5 matches original training.")
    parser.add_argument("--rffusion-agg-type", type=str, default="flatten",
                        choices=["flatten", "per_v", "per_v_attn"],
                        help="Embedding aggregator in RFFusion. "
                             "'flatten' (default) matches original training.")
    parser.add_argument("--rffusion-per-v-dim", type=int, default=16,
                        help="Hidden dim of shared per-V projection in RFFusion. "
                             "Only used when --rffusion-agg-type is per_v or per_v_attn.")
    parser.add_argument("--rffusion-bottleneck-dim", type=int, default=128,
                        help="Bottleneck dim of the RFFusion main path. Default 128.")
    parser.add_argument(
        "--fold-indices",
        type=str,
        default="all",
        help="Comma-separated fold ids (e.g. 1 or 0,2,4), or all.",
    )
    parser.add_argument("--save-eval-artifacts", action="store_true")
    parser.add_argument(
        "--eval-artifacts-dir",
        type=str,
        default=os.path.join(
            CODE_ROOT,
            "exp_output",
            "repertoire_classification",
            "clarev_vgene_weight",
            "eval_artifacts",
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Startup | Seed: {args.seed}", flush=True)
    print(f"Startup | Nested mode: {args.nested_mode}", flush=True)
    print(f"Startup | Extractor source: {args.extractor_source}", flush=True)

    required_paths = [
        args.bag_data_dir,
        os.path.join(args.bag_data_dir, "superbags.pk"),
        os.path.join(args.bag_data_dir, "v_gene_order.pk"),
        os.path.join(args.bag_data_dir, "smp_list.pk"),
        os.path.join(args.bag_data_dir, "smp_labels.pk"),
        os.path.join(args.bag_data_dir, "weights.pk"),
    ]
    if args.extractor_source == "whole":
        model_path = args.whole_model_path or os.path.join(args.trained_model_dir, "best_model.pth")
        required_paths.append(model_path)

    missing_paths = [p for p in required_paths if not os.path.exists(p)]
    if missing_paths:
        raise FileNotFoundError("Missing required paths:\n" + "\n".join(missing_paths))
    print("Startup | Path validation: passed", flush=True)

    superbags = load_pk(os.path.join(args.bag_data_dir, "superbags.pk"))
    v_gene_order = load_pk(os.path.join(args.bag_data_dir, "v_gene_order.pk"))
    smp_list = load_pk(os.path.join(args.bag_data_dir, "smp_list.pk"))
    smp_labels = load_pk(os.path.join(args.bag_data_dir, "smp_labels.pk"))
    weights = load_pk(os.path.join(args.bag_data_dir, "weights.pk"))

    sample_count = len(smp_list)
    folds = load_or_create_splits(
        split_file=args.split_file,
        sample_count=sample_count,
        labels=smp_labels,
        n_splits=args.n_splits,
        random_seed=args.split_seed,
        inner_n_splits=args.inner_n_splits,
        force_regenerate=args.force_regenerate_splits,
    )
    print(f"Startup | Shared split file: {args.split_file}", flush=True)
    selected_fold_ids = parse_fold_indices(args.fold_indices, args.n_splits)
    if selected_fold_ids is None:
        print("Startup | Selected folds: all", flush=True)
    else:
        print(f"Startup | Selected folds: {sorted(selected_fold_ids)}", flush=True)

    num_vgene = len(v_gene_order)
    v_freq = bagdata_to_vfreq(superbags, v_gene_order, smp_list, norm=True)

    if args.extractor_source == "whole":
        model_path = args.whole_model_path or os.path.join(args.trained_model_dir, "best_model.pth")
        best_model = ContrastiveModel(
            input_dim=args.tcr_emb_dim,
            output_dim=args.vgene_emb_dim,
        ).to(device)
        best_model.load_state_dict(torch.load(model_path, map_location=device))
        fold_model_state_paths = None
        print(f"Startup | Whole extractor path: {model_path}", flush=True)
    else:
        best_model = ContrastiveModel(
            input_dim=args.tcr_emb_dim,
            output_dim=args.vgene_emb_dim,
        ).to(device)
        fold_model_state_paths = collect_fold_model_paths(
            model_root=args.trained_model_dir,
            n_splits=args.n_splits,
            selected_fold_ids=selected_fold_ids,
        )
        print(
            f"Startup | Foldwise extractor root: {args.trained_model_dir}",
            flush=True,
        )

    whole_dataset = RepertoireDataset(
        superbags,
        labels=smp_labels,
        v_freq_mtx=v_freq,
        weight=weights,
        subbag_size=args.subbag_size,
    )

    print(f"Startup | Output directory: {args.output_dir}", flush=True)
    artifacts_dir = None
    if args.save_eval_artifacts:
        artifacts_dir = args.eval_artifacts_dir
        os.makedirs(artifacts_dir, exist_ok=True)
        print(f"Startup | Eval artifacts: enabled | Dir: {artifacts_dir}", flush=True)

    if args.methods in ("all", "vbag_vfreq"):
        vfreq_method_name = (
            "Vbag_RFFreq_feature_MLP" if args.vfreq_fusion_backend == "rf"
            else "Vbag_Vfreq_feature_MLP"
        )
        run_clarev_classification(
            whole_dataset=whole_dataset,
            folds=folds,
            best_model=best_model,
            device=device,
            vgene_emb_dim=args.vgene_emb_dim,
            num_vgene=num_vgene,
            num_epochs=args.num_epochs,
            batch_size=args.batch_size,
            test_batch_size=args.test_batch_size,
            use_vfreq=True,
            method_name=vfreq_method_name,
            output_dir=args.output_dir,
            artifacts_dir=artifacts_dir,
            save_predictions=True,
            selected_fold_ids=selected_fold_ids,
            nested_mode=args.nested_mode,
            labels=smp_labels,
            nested_seed=args.seed,
            inner_n_splits=args.inner_n_splits,
            early_stopping_patience=args.early_stopping_patience,
            fold_model_state_paths=fold_model_state_paths,
            vfreq_fusion_backend=args.vfreq_fusion_backend,
            rf_n_estimators=args.rf_n_estimators,
            rf_seed=args.rf_seed,
            clf_lr=args.clf_lr,
            clf_weight_decay=args.clf_weight_decay,
            rffusion_agg_type=args.rffusion_agg_type,
            rffusion_per_v_dim=args.rffusion_per_v_dim,
            rffusion_bottleneck_dim=args.rffusion_bottleneck_dim,
        )

    if args.methods in ("all", "vbag_only"):
        run_clarev_classification(
            whole_dataset=whole_dataset,
            folds=folds,
            best_model=best_model,
            device=device,
            vgene_emb_dim=args.vgene_emb_dim,
            num_vgene=num_vgene,
            num_epochs=args.num_epochs,
            batch_size=args.batch_size,
            test_batch_size=args.test_batch_size,
            use_vfreq=False,
            method_name="Vbag_feature_MLP",
            output_dir=args.output_dir,
            artifacts_dir=artifacts_dir,
            save_predictions=True,
            predictions_dir=args.output_dir,
            selected_fold_ids=selected_fold_ids,
            nested_mode=args.nested_mode,
            labels=smp_labels,
            nested_seed=args.seed,
            inner_n_splits=args.inner_n_splits,
            early_stopping_patience=args.early_stopping_patience,
            fold_model_state_paths=fold_model_state_paths,
        )


if __name__ == "__main__":
    main()
