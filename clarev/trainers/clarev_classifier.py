"""Repertoire-level ClareV classification orchestrator.

`run_clarev_classification` runs the per-fold downstream classifier on top of a
ClareV contrastive extractor (frozen). It supports both the embedding-only
classifier (use_vfreq=False) and the V-bag + V-usage fusion classifier
(use_vfreq=True), with the V-usage branch implemented either as the legacy MLP
fusion or the RFFusion classifier (random-forest probability fed into a
learnable fusion head with an RF-logit residual skip).

The entry point is `run_clarev_classification`. Its dependencies
(`bagdata_to_vfreq`, `save_metrics_to_csv`, `create_inner_train_val_indices`)
are also exposed at module level for direct use by top-level pipeline scripts.
"""
import copy
import os
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Subset

from clarev.data_loaders.bag_data_loader import RepertoireDataset
from clarev.models.contrastive_model import ContrastiveModel
from clarev.trainers.CL_evaluation import vfeature_classification
from clarev.trainers.CL_evaluation_rfhybrid import vfeature_classification_rfhybrid
from clarev.utils.eval_reporting import (
    save_confusion_matrix_artifacts,
    save_fold_predictions,
    save_method_predictions,
)


def bagdata_to_vfreq(
    superbags: list, v_gene_order: list, smp_list: list, norm: bool = True
) -> np.ndarray:
    """Convert per-repertoire V-bag clonotype counts into a V-frequency matrix.

    Returns an ``(n_samples, n_v_categories)`` array whose rows are normalized
    to sum to 1 when ``norm=True``.
    """
    v_freq = np.zeros((len(smp_list), len(v_gene_order)))
    for i, sample in enumerate(superbags):
        for v_idx in range(len(v_gene_order)):
            v_freq[i][v_idx] = len(sample[v_idx])

    if norm:
        row_sum = v_freq.sum(axis=1, keepdims=True)
        row_sum[row_sum == 0] = 1.0
        v_freq = v_freq / row_sum
    return v_freq


def save_metrics_to_csv(
    metrics_dict: Dict[str, Dict[str, float]], method_name: str, output_dir: str
) -> None:
    records = []
    for fold_name, metrics in metrics_dict.items():
        row = {"fold": fold_name}
        row.update(metrics)
        records.append(row)

    df = pd.DataFrame(records)
    stats = pd.DataFrame(
        {
            "fold": ["Mean", "Std"],
            "acc": [df["acc"].mean(), df["acc"].std()],
            "auc": [df["auc"].mean(), df["auc"].std()],
            "f1": [df["f1"].mean(), df["f1"].std()],
        }
    )
    final_df = pd.concat([df, stats], ignore_index=True)

    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"{method_name}_results.csv")
    final_df.to_csv(output_file, index=False, float_format="%.4f")
    print(f"Method: {method_name} | Saved results: {output_file}", flush=True)
    print(final_df, flush=True)


def create_inner_train_val_indices(
    train_idx: np.ndarray,
    labels: np.ndarray,
    fold_id: int,
    nested_seed: int,
    inner_n_splits: int = 5,
) -> Tuple[np.ndarray, np.ndarray]:
    """Split outer-train repertoires into inner-train / inner-val for early
    stopping in nested CV. Stratified when both classes have enough samples;
    falls back to a simple shuffle split otherwise."""
    train_labels = labels[train_idx]
    unique_labels = np.unique(train_labels.astype(int))
    if len(unique_labels) < 2:
        min_class_count = 0
    else:
        min_class_count = np.min(np.bincount(train_labels.astype(int)))

    if min_class_count >= inner_n_splits:
        inner_splitter = StratifiedKFold(
            n_splits=inner_n_splits,
            shuffle=True,
            random_state=nested_seed + fold_id,
        )
        inner_train_rel, inner_val_rel = next(
            inner_splitter.split(np.zeros(len(train_idx)), train_labels)
        )
    else:
        rng = np.random.default_rng(nested_seed + fold_id)
        shuffled = train_idx.copy()
        rng.shuffle(shuffled)
        inner_val_size = max(1, int(0.2 * len(shuffled)))
        inner_val_idx = shuffled[:inner_val_size]
        inner_train_idx = shuffled[inner_val_size:]
        return inner_train_idx, inner_val_idx

    inner_train_idx = train_idx[inner_train_rel]
    inner_val_idx = train_idx[inner_val_rel]
    return inner_train_idx, inner_val_idx


def run_clarev_classification(
    whole_dataset: RepertoireDataset,
    folds: Sequence[Tuple[np.ndarray, ...]],
    best_model: ContrastiveModel,
    device: torch.device,
    vgene_emb_dim: int,
    num_vgene: int,
    num_epochs: int,
    batch_size: int,
    test_batch_size: int,
    use_vfreq: bool,
    method_name: str,
    output_dir: str,
    artifacts_dir: Optional[str] = None,
    save_predictions: bool = False,
    predictions_dir: Optional[str] = None,
    selected_fold_ids: Optional[Set[int]] = None,
    nested_mode: bool = False,
    labels: Optional[np.ndarray] = None,
    nested_seed: int = 42,
    inner_n_splits: int = 5,
    early_stopping_patience: Optional[int] = None,
    fold_model_state_paths: Optional[Dict[int, str]] = None,
    vfreq_fusion_backend: str = "mlp",
    rf_n_estimators: int = 300,
    rf_seed: int = 1,
    clf_lr: float = 1e-4,
    clf_weight_decay: float = 1e-5,
    rffusion_agg_type: str = "flatten",
    rffusion_per_v_dim: int = 16,
    rffusion_bottleneck_dim: int = 128,
) -> None:
    all_metrics: Dict[str, Dict[str, float]] = {}
    fold_predictions = {"y_true": [], "y_pred": [], "y_prob": []}
    fold_ids: List[np.ndarray] = []
    epoch_log_records: List[Dict[str, float]] = []
    for fold_id, fold in enumerate(folds):
        if len(fold) == 3:
            train_idx = np.array(fold[0], dtype=int)
            val_idx_from_split = np.array(fold[1], dtype=int)
            test_idx = np.array(fold[2], dtype=int)
        elif len(fold) == 2:
            train_idx = np.array(fold[0], dtype=int)
            val_idx_from_split = None
            test_idx = np.array(fold[1], dtype=int)
        else:
            raise ValueError("Each fold must be (train, test) or (train, val, test).")

        if selected_fold_ids is not None and fold_id not in selected_fold_ids:
            continue

        fold_encoder = best_model
        if fold_model_state_paths is not None:
            if fold_id not in fold_model_state_paths:
                raise ValueError(
                    f"Missing fold-specific extractor for fold {fold_id}. "
                    "Please provide fold_model_state_paths for all selected folds."
                )
            model_path = fold_model_state_paths[fold_id]
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Fold {fold_id} extractor not found: {model_path}")
            # Clone architecture from the template and load fold-specific CL weights.
            fold_encoder = copy.deepcopy(best_model).to(device)
            fold_encoder.load_state_dict(torch.load(model_path, map_location=device))
            fold_encoder.eval()
            print(
                f"Method: {method_name} | Fold: {fold_id} | Loaded fold-specific extractor: {model_path}",
                flush=True,
            )
        print(f"Method: {method_name} | Fold: {fold_id} | Status: start", flush=True)
        if nested_mode:
            if val_idx_from_split is None and labels is None:
                raise ValueError("labels are required when nested_mode=True")
            if val_idx_from_split is None:
                inner_train_idx, inner_val_idx = create_inner_train_val_indices(
                    train_idx=train_idx,
                    labels=labels,
                    fold_id=fold_id,
                    nested_seed=nested_seed,
                    inner_n_splits=inner_n_splits,
                )
            else:
                inner_train_idx, inner_val_idx = train_idx, val_idx_from_split
            train_dataset = Subset(whole_dataset, inner_train_idx)
            val_dataset = Subset(whole_dataset, inner_val_idx)
            test_dataset = Subset(whole_dataset, test_idx)
            val_loader = DataLoader(val_dataset, batch_size=test_batch_size, shuffle=False)
            print(
                (
                    f"Method: {method_name} | Fold: {fold_id} | "
                    f"Inner train size: {len(inner_train_idx)} | Inner val size: {len(inner_val_idx)} | "
                    f"Outer test size: {len(test_idx)}"
                ),
                flush=True,
            )
        else:
            train_indices = (
                np.concatenate([train_idx, val_idx_from_split])
                if val_idx_from_split is not None
                else train_idx
            )
            train_dataset = Subset(whole_dataset, train_indices)
            test_dataset = Subset(whole_dataset, test_idx)
            val_loader = None

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)

        need_details = (artifacts_dir is not None) or save_predictions

        if use_vfreq and vfreq_fusion_backend == "rf":
            eval_fn = vfeature_classification_rfhybrid
            eval_kwargs = {
                "rf_n_estimators": rf_n_estimators,
                "rf_seed": rf_seed + fold_id,
                "clf_lr": clf_lr,
                "clf_weight_decay": clf_weight_decay,
                "rffusion_agg_type": rffusion_agg_type,
                "rffusion_per_v_dim": rffusion_per_v_dim,
                "rffusion_bottleneck_dim": rffusion_bottleneck_dim,
            }
        else:
            eval_fn = vfeature_classification
            eval_kwargs = {}

        if need_details:
            fold_metrics, fold_details = eval_fn(
                train_loader=train_loader,
                test_loader=test_loader,
                encoder=fold_encoder,
                device=device,
                emb_dim=vgene_emb_dim,
                num_vgene=num_vgene,
                num_epochs=num_epochs,
                class_num=2,
                use_vfreq=use_vfreq,
                return_details=True,
                val_loader=val_loader,
                nested_mode=nested_mode,
                early_stopping_patience=early_stopping_patience,
                **eval_kwargs,
            )
            if artifacts_dir is not None:
                save_fold_predictions(
                    output_dir=artifacts_dir,
                    method_name=method_name,
                    fold_id=fold_id,
                    y_true=fold_details["labels"],
                    y_pred=fold_details["preds"],
                    y_prob=fold_details["pred_probs"],
                )
            fold_predictions["y_true"].append(fold_details["labels"].copy())
            fold_predictions["y_pred"].append(fold_details["preds"].copy())
            fold_predictions["y_prob"].append(fold_details["pred_probs"].copy())
            fold_ids.append(
                np.full(shape=len(fold_details["labels"]), fill_value=fold_id, dtype=int)
            )
            fold_epoch_logs = fold_details.get("epoch_logs", [])
        else:
            fold_metrics = eval_fn(
                train_loader=train_loader,
                test_loader=test_loader,
                encoder=fold_encoder,
                device=device,
                emb_dim=vgene_emb_dim,
                num_vgene=num_vgene,
                num_epochs=num_epochs,
                class_num=2,
                use_vfreq=use_vfreq,
                val_loader=val_loader,
                nested_mode=nested_mode,
                early_stopping_patience=early_stopping_patience,
                **eval_kwargs,
            )
            fold_epoch_logs = fold_metrics.get("epoch_logs", [])

        for row in fold_epoch_logs:
            row_with_fold = dict(row)
            row_with_fold["method"] = method_name
            row_with_fold["fold"] = fold_id
            epoch_log_records.append(row_with_fold)
        all_metrics[f"fold{fold_id}"] = {
            "acc": fold_metrics["acc"],
            "auc": fold_metrics["auc"],
            "f1": fold_metrics["f1"],
        }
        print(
            (
                f"Method: {method_name} | Fold: {fold_id} | "
                f"Best Val Epoch: {fold_metrics['epoch']} | "
                f"Test AUC: {fold_metrics['auc']:.4f} | Test ACC: {fold_metrics['acc']:.4f} | "
                f"Test F1: {fold_metrics['f1']:.4f}"
            ),
            flush=True,
        )

    save_metrics_to_csv(all_metrics, method_name=method_name, output_dir=output_dir)
    if epoch_log_records:
        epoch_log_df = pd.DataFrame(epoch_log_records)
        epoch_log_df = epoch_log_df[
            [
                "method",
                "fold",
                "epoch",
                "train_loss",
                "train_acc",
                "train_auc",
                "val_loss",
                "val_auc",
                "val_acc",
                "test_loss",
                "test_auc",
                "test_acc",
                "best_val_auc",
                "best_val_epoch",
            ]
        ]
        epoch_log_file = os.path.join(output_dir, f"{method_name}_epoch_logs.csv")
        epoch_log_df.to_csv(epoch_log_file, index=False, float_format="%.6f")
        print(f"Method: {method_name} | Saved epoch logs: {epoch_log_file}", flush=True)

    if save_predictions:
        prediction_output_dir = predictions_dir if predictions_dir is not None else output_dir
        prediction_file = save_method_predictions(
            output_dir=prediction_output_dir,
            method_name=method_name,
            y_true=np.concatenate(fold_predictions["y_true"]),
            y_pred=np.concatenate(fold_predictions["y_pred"]),
            y_prob=np.concatenate(fold_predictions["y_prob"]),
            fold_ids=np.concatenate(fold_ids),
        )
        print(f"Method: {method_name} | Saved predictions: {prediction_file}", flush=True)
    if artifacts_dir is not None:
        outputs = save_confusion_matrix_artifacts(
            output_dir=artifacts_dir,
            method_name=method_name,
            y_true=np.concatenate(fold_predictions["y_true"]),
            y_pred=np.concatenate(fold_predictions["y_pred"]),
        )
        print(f"Method: {method_name} | Saved confusion artifacts: {outputs}", flush=True)
