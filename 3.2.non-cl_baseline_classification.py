#!/usr/bin/env python3
"""3.2 Non-contrastive learned V-bag baselines for repertoire classification.

This script consolidates two non-contrastive learned-bag baselines that share
ClareV's V-bag inputs but replace contrastive pretraining with end-to-end
supervised training:

  * **Supervised_Vbag_Encoder** — same bag-encoder architecture as ClareV's
    contrastive extractor (Linear -> LayerNorm -> ReLU -> mean pool ->
    Linear), trained jointly with a small repertoire-level classifier.
  * **Attention_MIL_Vbag_Encoder** — replaces mean pooling with a learned
    gated-attention aggregator over clonotypes within each V bag (a more
    expressive non-contrastive learned-bag alternative).

Both methods follow the same nested 5-fold split protocol as the rest of the
pipeline, are trained with early stopping on the inner validation fold, and
are evaluated once on the outer test fold per outer fold. By default both
methods run sequentially; pass ``--methods supervised_vbag`` or
``--methods attention_mil`` to limit to one.
"""

import argparse
import copy
import os
import random
import sys
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader, Subset


SCRIPT_PATH = os.path.abspath(__file__)
CODE_ROOT = os.path.dirname(SCRIPT_PATH)
if CODE_ROOT not in sys.path:
    sys.path.insert(0, CODE_ROOT)

from clarev.data_loaders.bag_data_loader import RepertoireDataset
from clarev.utils.utils import load_pk


# =============================================================================
# Models
# =============================================================================


class SupervisedVBagEncoder(nn.Module):
    """ClareV-style V-bag encoder trained without contrastive loss."""

    def __init__(self, input_dim: int = 120, output_dim: int = 120,
                 hidden_dim: int = 128, dropout: float = 0.2):
        super().__init__()
        self.fc1 = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.Dropout(dropout))
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, num_v, bag_size, input_dim]
        x = x.float()
        h = torch.relu(self.layer_norm(self.fc1(x)))
        h = torch.mean(h, dim=-2)
        return self.fc2(h)  # [batch, num_v, output_dim]


class GatedAttentionPooling(nn.Module):
    """Gated attention pooling for multiple-instance learning."""

    def __init__(self, input_dim: int, attention_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        self.tanh_layer = nn.Linear(input_dim, attention_dim)
        self.sigmoid_layer = nn.Linear(input_dim, attention_dim)
        self.score_layer = nn.Linear(attention_dim, 1, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # h: [batch, num_v, bag_size, hidden_dim]
        gated = torch.tanh(self.tanh_layer(h)) * torch.sigmoid(self.sigmoid_layer(h))
        gated = self.dropout(gated)
        scores = self.score_layer(gated).squeeze(-1)
        weights = torch.softmax(scores, dim=-1)
        pooled = torch.sum(weights.unsqueeze(-1) * h, dim=-2)
        return pooled, weights


class AttentionMILVBagEncoder(nn.Module):
    """Non-contrastive attention-MIL encoder over clonotypes within each V bag."""

    def __init__(self, input_dim: int = 120, output_dim: int = 120,
                 hidden_dim: int = 128, attention_dim: int = 64, dropout: float = 0.2):
        super().__init__()
        self.instance_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.attention_pool = GatedAttentionPooling(
            hidden_dim, attention_dim=attention_dim, dropout=dropout * 0.5,
        )
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, num_v, bag_size, input_dim]
        h = self.instance_encoder(x.float())
        pooled, _ = self.attention_pool(h)
        return self.output_layer(pooled)


class RepertoireClassifier(nn.Module):
    """Repertoire-level classifier on top of (B, num_v, emb_dim) features."""

    def __init__(self, num_v: int, emb_dim: int = 120, hidden_dim: int = 128,
                 output_dim: int = 2, dropout: float = 0.2):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(num_v * emb_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.flatten(x)
        x = self.dropout(torch.relu(self.fc1(x)))
        return self.fc2(x)


class SupervisedVBagClassifier(nn.Module):
    """End-to-end non-contrastive supervised V-bag baseline."""

    def __init__(self, num_v: int, input_dim: int = 120, emb_dim: int = 120,
                 hidden_dim: int = 128, dropout: float = 0.2):
        super().__init__()
        self.encoder = SupervisedVBagEncoder(
            input_dim=input_dim, output_dim=emb_dim,
            hidden_dim=hidden_dim, dropout=dropout,
        )
        self.classifier = RepertoireClassifier(
            num_v=num_v, emb_dim=emb_dim, hidden_dim=hidden_dim, dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.encoder(x))


class AttentionMILVBagClassifier(nn.Module):
    """End-to-end attention-MIL V-bag classifier without contrastive loss."""

    def __init__(self, num_v: int, input_dim: int = 120, emb_dim: int = 120,
                 hidden_dim: int = 128, attention_dim: int = 64, dropout: float = 0.2):
        super().__init__()
        self.encoder = AttentionMILVBagEncoder(
            input_dim=input_dim, output_dim=emb_dim, hidden_dim=hidden_dim,
            attention_dim=attention_dim, dropout=dropout,
        )
        self.classifier = RepertoireClassifier(
            num_v=num_v, emb_dim=emb_dim, hidden_dim=hidden_dim, dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.encoder(x))


# =============================================================================
# Helpers
# =============================================================================


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_shared_splits(split_file: str, sample_count: int):
    if not os.path.exists(split_file):
        raise FileNotFoundError(f"Shared split file not found: {split_file}")
    split_data = np.load(split_file, allow_pickle=True)
    file_sample_count = int(split_data["sample_count"])
    if file_sample_count != sample_count:
        raise ValueError(
            f"Split file sample_count mismatch: file={file_sample_count}, "
            f"current={sample_count}"
        )
    if not {"train_splits", "val_splits", "test_splits"}.issubset(split_data.files):
        raise ValueError(
            "Expected train_splits, val_splits, and test_splits in the shared split file"
        )
    return [
        (
            np.array(train_idx, dtype=int),
            np.array(val_idx, dtype=int),
            np.array(test_idx, dtype=int),
        )
        for train_idx, val_idx, test_idx in zip(
            split_data["train_splits"], split_data["val_splits"], split_data["test_splits"]
        )
    ]


def parse_fold_indices(fold_indices_text: str, n_splits: int) -> Optional[Set[int]]:
    if fold_indices_text.lower() == "all":
        return None
    selected: Set[int] = set()
    for token in fold_indices_text.split(","):
        token = token.strip()
        if not token:
            continue
        fold_id = int(token)
        if fold_id < 0 or fold_id >= n_splits:
            raise ValueError(f"Invalid fold id {fold_id}, expected in [0, {n_splits - 1}]")
        selected.add(fold_id)
    if not selected:
        raise ValueError("No valid fold ids provided in --fold-indices")
    return selected


def bagdata_to_vfreq(superbags: list, v_gene_order: list, smp_list: list,
                    norm: bool = True) -> np.ndarray:
    v_freq = np.zeros((len(smp_list), len(v_gene_order)))
    for i, sample in enumerate(superbags):
        for v_idx in range(len(v_gene_order)):
            v_freq[i][v_idx] = len(sample[v_idx])
    if norm:
        row_sum = v_freq.sum(axis=1, keepdims=True)
        row_sum[row_sum == 0] = 1.0
        v_freq = v_freq / row_sum
    return v_freq


def evaluate_model(model: nn.Module, loader: DataLoader, criterion: nn.Module,
                   device: torch.device) -> Dict[str, object]:
    model.eval()
    total_loss = 0.0
    y_true: List[np.ndarray] = []
    y_pred: List[np.ndarray] = []
    y_prob: List[np.ndarray] = []
    with torch.no_grad():
        for batch_x, batch_y, _batch_vfreq in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            probs = torch.softmax(logits, dim=1)[:, 1]
            preds = torch.argmax(logits, dim=1)
            total_loss += loss.item() * len(batch_y)
            y_true.append(batch_y.cpu().numpy())
            y_pred.append(preds.cpu().numpy())
            y_prob.append(probs.cpu().numpy())
    labels = np.concatenate(y_true)
    preds = np.concatenate(y_pred)
    probs = np.concatenate(y_prob)
    try:
        auc = roc_auc_score(labels, probs)
    except ValueError:
        auc = np.nan
    return {
        "loss": total_loss / len(loader.dataset),
        "auc": auc,
        "acc": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds),
        "labels": labels,
        "preds": preds,
        "pred_probs": probs,
    }


def save_metrics_to_csv(metrics_dict: Dict[str, Dict[str, float]], method_name: str,
                        output_dir: str) -> None:
    rows = []
    for fold_name, metrics in metrics_dict.items():
        row = {"fold": fold_name}
        row.update(metrics)
        rows.append(row)
    df = pd.DataFrame(rows)
    stats = pd.DataFrame(
        {
            "fold": ["Mean", "Std"],
            "acc": [df["acc"].mean(), df["acc"].std()],
            "auc": [df["auc"].mean(), df["auc"].std()],
            "f1": [df["f1"].mean(), df["f1"].std()],
            "epoch": [df["epoch"].mean(), df["epoch"].std()],
        }
    )
    final_df = pd.concat([df, stats], ignore_index=True)
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"{method_name}_results.csv")
    final_df.to_csv(output_file, index=False, float_format="%.4f")
    print(f"Saved results: {output_file}", flush=True)
    print(final_df, flush=True)


def save_predictions(output_dir: str, method_name: str, y_true: np.ndarray,
                     y_pred: np.ndarray, y_prob: np.ndarray,
                     fold_ids: np.ndarray) -> None:
    os.makedirs(output_dir, exist_ok=True)
    df = pd.DataFrame({"fold": fold_ids, "y_true": y_true, "y_pred": y_pred,
                       "y_prob": y_prob})
    output_file = os.path.join(output_dir, f"{method_name}_predictions.csv")
    df.to_csv(output_file, index=False, float_format="%.6f")
    print(f"Saved predictions: {output_file}", flush=True)


# =============================================================================
# Per-fold training (shared by both methods)
# =============================================================================


def build_model(method: str, num_v: int, args: argparse.Namespace) -> nn.Module:
    if method == "supervised_vbag":
        return SupervisedVBagClassifier(
            num_v=num_v, input_dim=args.input_dim, emb_dim=args.emb_dim,
            hidden_dim=args.hidden_dim, dropout=args.dropout,
        )
    if method == "attention_mil":
        return AttentionMILVBagClassifier(
            num_v=num_v, input_dim=args.input_dim, emb_dim=args.emb_dim,
            hidden_dim=args.hidden_dim, attention_dim=args.attention_dim,
            dropout=args.dropout,
        )
    raise ValueError(f"Unknown method {method}")


def train_one_fold(
    method: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    num_v: int,
    args: argparse.Namespace,
    device: torch.device,
    fold_seed: int,
) -> Tuple[Dict[str, object], List[Dict[str, float]]]:
    model = build_model(method, num_v, args).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                   weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()

    best_val_auc = -np.inf
    best_state = None
    best_epoch = 0
    no_improve = 0
    epoch_logs: List[Dict[str, float]] = []

    for epoch in range(1, args.num_epochs + 1):
        model.train()
        train_losses: List[float] = []
        train_labels: List[np.ndarray] = []
        train_probs: List[np.ndarray] = []
        train_preds: List[np.ndarray] = []
        for batch_x, batch_y, _batch_vfreq in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            probs = torch.softmax(logits.detach(), dim=1)[:, 1]
            preds = torch.argmax(logits.detach(), dim=1)
            train_losses.append(loss.item() * len(batch_y))
            train_labels.append(batch_y.detach().cpu().numpy())
            train_probs.append(probs.cpu().numpy())
            train_preds.append(preds.cpu().numpy())

        tr_labels = np.concatenate(train_labels)
        tr_probs = np.concatenate(train_probs)
        tr_preds = np.concatenate(train_preds)
        try:
            train_auc = roc_auc_score(tr_labels, tr_probs)
        except ValueError:
            train_auc = np.nan
        train_loss = float(np.sum(train_losses) / len(train_loader.dataset))
        train_acc = accuracy_score(tr_labels, tr_preds)

        # Reproducible val sub-bag sampling (seeded by fold + epoch).
        set_all_seeds(fold_seed + 10_000 + epoch)
        val_metrics = evaluate_model(model, val_loader, criterion, device)
        val_auc = val_metrics["auc"]
        improved = not np.isnan(val_auc) and val_auc > best_val_auc
        if improved:
            best_val_auc = float(val_auc)
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            no_improve = 0
        else:
            no_improve += 1

        epoch_logs.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": float(train_acc),
            "train_auc": float(train_auc),
            "val_loss": float(val_metrics["loss"]),
            "val_auc": float(val_metrics["auc"]),
            "val_acc": float(val_metrics["acc"]),
            "test_loss": np.nan,
            "test_auc": np.nan,
            "test_acc": np.nan,
            "best_val_auc": float(best_val_auc),
            "best_val_epoch": float(best_epoch),
        })
        print(
            f"  Epoch {epoch}/{args.num_epochs} | Train AUC: {train_auc:.4f} | "
            f"Val AUC: {val_metrics['auc']:.4f} | "
            f"Best Val AUC: {best_val_auc:.4f} (epoch {best_epoch})",
            flush=True,
        )
        if args.early_stopping_patience > 0 and no_improve >= args.early_stopping_patience:
            print(
                f"  Early stop at epoch {epoch}; no Val AUC improvement for "
                f"{args.early_stopping_patience} epochs",
                flush=True,
            )
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    set_all_seeds(fold_seed + 20_000)
    final_test = evaluate_model(model, test_loader, criterion, device)
    final_test["epoch"] = best_epoch
    final_test["epoch_logs"] = epoch_logs
    return final_test, epoch_logs


# =============================================================================
# Per-method runner
# =============================================================================


_METHOD_NAMES = {
    "supervised_vbag": "Supervised_Vbag_Encoder",
    "attention_mil": "Attention_MIL_Vbag_Encoder",
}


def run_method(
    method: str,
    folds,
    whole_dataset: RepertoireDataset,
    num_v: int,
    args: argparse.Namespace,
    device: torch.device,
    selected_fold_ids: Optional[Set[int]],
) -> None:
    method_name = _METHOD_NAMES[method]
    print(f"\n=========== Method: {method_name} ===========", flush=True)
    all_metrics: Dict[str, Dict[str, float]] = {}
    epoch_log_rows: List[Dict[str, float]] = []
    pred_labels: List[np.ndarray] = []
    pred_preds: List[np.ndarray] = []
    pred_probs: List[np.ndarray] = []
    pred_folds: List[np.ndarray] = []

    for fold_id, (train_idx, val_idx, test_idx) in enumerate(folds):
        if selected_fold_ids is not None and fold_id not in selected_fold_ids:
            continue
        if (set(train_idx.tolist()) & set(val_idx.tolist())
                or set(train_idx.tolist()) & set(test_idx.tolist())
                or set(val_idx.tolist()) & set(test_idx.tolist())):
            raise ValueError(f"Fold {fold_id} has overlapping train/val/test indices")
        fold_seed = args.seed + fold_id * 1000
        set_all_seeds(fold_seed)
        print(
            f"Fold {fold_id} | Train: {len(train_idx)} | Val: {len(val_idx)} | "
            f"Test: {len(test_idx)} | Fold seed: {fold_seed}",
            flush=True,
        )
        train_loader = DataLoader(Subset(whole_dataset, train_idx),
                                   batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(Subset(whole_dataset, val_idx),
                                 batch_size=args.test_batch_size, shuffle=False)
        test_loader = DataLoader(Subset(whole_dataset, test_idx),
                                  batch_size=args.test_batch_size, shuffle=False)
        fold_metrics, fold_epoch_logs = train_one_fold(
            method=method, train_loader=train_loader, val_loader=val_loader,
            test_loader=test_loader, num_v=num_v, args=args,
            device=device, fold_seed=fold_seed,
        )
        all_metrics[f"fold{fold_id}"] = {
            "acc": float(fold_metrics["acc"]),
            "auc": float(fold_metrics["auc"]),
            "f1": float(fold_metrics["f1"]),
            "epoch": float(fold_metrics["epoch"]),
        }
        for row in fold_epoch_logs:
            out_row = dict(row)
            out_row["method"] = method_name
            out_row["fold"] = fold_id
            epoch_log_rows.append(out_row)
        pred_labels.append(fold_metrics["labels"].copy())
        pred_preds.append(fold_metrics["preds"].copy())
        pred_probs.append(fold_metrics["pred_probs"].copy())
        pred_folds.append(np.full(len(fold_metrics["labels"]), fold_id, dtype=int))
        print(
            f"Fold {fold_id} final | AUC: {fold_metrics['auc']:.4f} | "
            f"ACC: {fold_metrics['acc']:.4f} | F1: {fold_metrics['f1']:.4f}",
            flush=True,
        )

    save_metrics_to_csv(all_metrics, method_name, args.output_dir)
    if epoch_log_rows:
        epoch_df = pd.DataFrame(epoch_log_rows)
        cols = [
            "method", "fold", "epoch",
            "train_loss", "train_acc", "train_auc",
            "val_loss", "val_auc", "val_acc",
            "test_loss", "test_auc", "test_acc",
            "best_val_auc", "best_val_epoch",
        ]
        epoch_df = epoch_df[cols]
        epoch_file = os.path.join(args.output_dir, f"{method_name}_epoch_logs.csv")
        epoch_df.to_csv(epoch_file, index=False, float_format="%.6f")
        print(f"Saved epoch logs: {epoch_file}", flush=True)
    save_predictions(
        args.output_dir, method_name,
        np.concatenate(pred_labels), np.concatenate(pred_preds),
        np.concatenate(pred_probs), np.concatenate(pred_folds),
    )


# =============================================================================
# Entry point
# =============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="3.2 Non-contrastive learned V-bag baselines "
                    "(supervised V-bag encoder + attention-MIL V-bag encoder)."
    )
    parser.add_argument(
        "--bag-data-dir", type=str,
        default=os.path.join(CODE_ROOT, "data", "processed_data", "Emerson2017_vgene"),
    )
    parser.add_argument(
        "--split-file", type=str,
        default=os.path.join(CODE_ROOT, "data", "splits", "emerson2017_nested_kfold5.npz"),
    )
    parser.add_argument(
        "--output-dir", type=str,
        default=os.path.join(CODE_ROOT, "exp_output", "repertoire_classification",
                             "non_cl_baselines_vgene"),
    )
    parser.add_argument(
        "--methods", type=str, default="all",
        choices=["all", "supervised_vbag", "attention_mil"],
        help="Which non-CL baseline(s) to run. Default 'all' runs both.",
    )
    parser.add_argument("--fold-indices", type=str, default="all")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--num-epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--test-batch-size", type=int, default=1)
    parser.add_argument("--subbag-size", type=int, default=50)
    parser.add_argument("--input-dim", type=int, default=120)
    parser.add_argument("--emb-dim", type=int, default=120)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--attention-dim", type=int, default=64,
                        help="Attention head dim (used by attention_mil only).")
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--early-stopping-patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_all_seeds(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Startup | Device: {device} | Seed: {args.seed}", flush=True)

    required_files = [
        os.path.join(args.bag_data_dir, "superbags.pk"),
        os.path.join(args.bag_data_dir, "v_gene_order.pk"),
        os.path.join(args.bag_data_dir, "smp_list.pk"),
        os.path.join(args.bag_data_dir, "smp_labels.pk"),
        os.path.join(args.bag_data_dir, "weights.pk"),
        args.split_file,
    ]
    missing = [p for p in required_files if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError("Missing required files:\n" + "\n".join(missing))

    superbags = load_pk(os.path.join(args.bag_data_dir, "superbags.pk"))
    v_gene_order = load_pk(os.path.join(args.bag_data_dir, "v_gene_order.pk"))
    smp_list = load_pk(os.path.join(args.bag_data_dir, "smp_list.pk"))
    smp_labels = np.asarray(load_pk(os.path.join(args.bag_data_dir, "smp_labels.pk")), dtype=int)
    weights = load_pk(os.path.join(args.bag_data_dir, "weights.pk"))
    num_v = len(v_gene_order)
    print(f"Startup | Samples: {len(smp_list)} | V categories: {num_v}", flush=True)

    v_freq = bagdata_to_vfreq(superbags, v_gene_order, smp_list, norm=True)
    whole_dataset = RepertoireDataset(
        superbags, labels=smp_labels, v_freq_mtx=v_freq,
        weight=weights, subbag_size=args.subbag_size,
    )
    folds = load_shared_splits(args.split_file, sample_count=len(smp_list))
    selected_fold_ids = parse_fold_indices(args.fold_indices, args.n_splits)
    print(f"Startup | Split file: {args.split_file}", flush=True)
    print(
        f"Startup | Selected folds: "
        f"{'all' if selected_fold_ids is None else sorted(selected_fold_ids)}",
        flush=True,
    )

    methods_to_run = (
        ["supervised_vbag", "attention_mil"] if args.methods == "all" else [args.methods]
    )
    print(f"Startup | Methods to run: {methods_to_run}", flush=True)

    os.makedirs(args.output_dir, exist_ok=True)
    for method in methods_to_run:
        run_method(method, folds, whole_dataset, num_v, args, device, selected_fold_ids)


if __name__ == "__main__":
    main()
