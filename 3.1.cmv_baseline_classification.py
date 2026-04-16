import argparse
import copy
import os
import sys
import warnings
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from torch.utils.data import DataLoader, TensorDataset


SCRIPT_PATH = os.path.abspath(__file__)
CODE_ROOT = os.path.dirname(SCRIPT_PATH)
if CODE_ROOT not in sys.path:
    sys.path.insert(0, CODE_ROOT)

from clarev.utils.eval_reporting import (
    save_confusion_matrix_artifacts,
    save_method_predictions,
)
from clarev.utils.utils import load_pk


def bagdata_to_vfreq(superbags: list, v_gene_order: list, smp_list: list, norm: bool = True) -> np.ndarray:
    v_freq = np.zeros((len(smp_list), len(v_gene_order)))
    for i, sample in enumerate(superbags):
        for v_idx in range(len(v_gene_order)):
            v_freq[i][v_idx] = len(sample[v_idx])

    if norm:
        row_sum = v_freq.sum(axis=1, keepdims=True)
        row_sum[row_sum == 0] = 1.0
        v_freq = v_freq / row_sum
    return v_freq


def validate_required_paths(required_paths: List[str]) -> None:
    missing_paths = [path for path in required_paths if not os.path.exists(path)]
    if missing_paths:
        raise FileNotFoundError("Missing required paths:\n" + "\n".join(missing_paths))


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


class CNN(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.conv = nn.Conv1d(1, 8, kernel_size=3)
        self.fc = nn.Linear(8 * (input_dim - 2), 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return torch.sigmoid(self.fc(x))


class TorchClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, model_class, input_dim: int, epochs: int = 10, lr: float = 0.0001):
        self.model_class = model_class
        self.input_dim = input_dim
        self.epochs = epochs
        self.lr = lr
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def fit(self, X: np.ndarray, y: np.ndarray):
        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.FloatTensor(y).to(self.device)

        self.model = self.model_class(self.input_dim).to(self.device)
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.BCELoss()
        loader = DataLoader(TensorDataset(X_tensor, y_tensor), batch_size=32, shuffle=True)

        for _ in range(self.epochs):
            for batch_x, batch_y in loader:
                optimizer.zero_grad()
                outputs = self.model(batch_x).squeeze()
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X_tensor = torch.FloatTensor(X).to(self.device)
        with torch.no_grad():
            proba = self.model(X_tensor).cpu().numpy()
        return np.hstack([1 - proba, proba])


def save_metrics_to_csv(
    metrics_dict: Dict[str, Dict[str, float]],
    method_name: str,
    output_dir: str,
) -> None:
    records = [{"fold": fold_name, **metrics} for fold_name, metrics in metrics_dict.items()]
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


def train_mlp_with_validation(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    max_epochs: int = 100,
    patience: int = 5,
    random_state: int = 0,
) -> Tuple[MLPClassifier, Dict[str, float]]:
    model = MLPClassifier(
        hidden_layer_sizes=(128,),
        max_iter=1,
        warm_start=True,
        early_stopping=False,
        random_state=random_state,
    )
    best_model = None
    best_val_auc = -np.inf
    best_epoch = 0
    no_improve_epochs = 0
    stop_epoch = max_epochs

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        for epoch in range(1, max_epochs + 1):
            model.fit(x_train, y_train)
            val_prob = model.predict_proba(x_val)[:, 1]
            try:
                val_auc = roc_auc_score(y_val, val_prob)
            except ValueError:
                val_auc = np.nan

            if not np.isnan(val_auc) and val_auc > best_val_auc:
                best_val_auc = val_auc
                best_epoch = epoch
                best_model = copy.deepcopy(model)
                no_improve_epochs = 0
            else:
                no_improve_epochs += 1

            if no_improve_epochs >= patience:
                stop_epoch = epoch
                break

    if best_model is None:
        best_model = model
        best_val_auc = np.nan
        best_epoch = stop_epoch

    return best_model, {
        "best_val_auc": float(best_val_auc),
        "best_epoch": float(best_epoch),
        "stop_epoch": float(stop_epoch),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="3.1 Baseline CMV classification with shared split file."
    )
    parser.add_argument(
        "--bag-data-dir",
        type=str,
        default=os.path.join(CODE_ROOT, "data", "processed_data", "Emerson2017_vgene"),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.path.join(
            CODE_ROOT, "exp_output", "cmv_classification", "baseline_vgene_freq"
        ),
    )
    parser.add_argument(
        "--split-file",
        type=str,
        default=os.path.join(
            CODE_ROOT,
            "exp_output",
            "cmv_classification",
            "shared_splits",
            "emerson2017_vgene_kfold5_seed1_inner5_tvt.npz",
        ),
    )
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--split-seed", type=int, default=1)
    parser.add_argument("--inner-n-splits", type=int, default=5)
    parser.add_argument("--force-regenerate-splits", action="store_true")
    parser.add_argument("--save-eval-artifacts", action="store_true")
    parser.add_argument(
        "--eval-artifacts-dir",
        type=str,
        default=os.path.join(
            CODE_ROOT,
            "exp_output",
            "cmv_classification",
            "baseline_vgene_freq",
            "eval_artifacts",
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    required_paths = [
        args.bag_data_dir,
        os.path.join(args.bag_data_dir, "superbags.pk"),
        os.path.join(args.bag_data_dir, "v_gene_order.pk"),
        os.path.join(args.bag_data_dir, "smp_list.pk"),
        os.path.join(args.bag_data_dir, "smp_labels.pk"),
    ]
    validate_required_paths(required_paths)
    print("Startup | Path validation: passed", flush=True)

    superbags = load_pk(os.path.join(args.bag_data_dir, "superbags.pk"))
    v_gene_order = load_pk(os.path.join(args.bag_data_dir, "v_gene_order.pk"))
    smp_list = load_pk(os.path.join(args.bag_data_dir, "smp_list.pk"))
    smp_labels = load_pk(os.path.join(args.bag_data_dir, "smp_labels.pk"))

    v_freq = bagdata_to_vfreq(superbags, v_gene_order, smp_list, norm=True)
    folds = load_or_create_splits(
        split_file=args.split_file,
        sample_count=len(smp_list),
        labels=smp_labels,
        n_splits=args.n_splits,
        random_seed=args.split_seed,
        inner_n_splits=args.inner_n_splits,
        force_regenerate=args.force_regenerate_splits,
    )
    print(f"Startup | Shared split file: {args.split_file}", flush=True)

    feature_dim = int(v_freq.shape[1])
    methods = {
        "LogisticRegression": LogisticRegression(random_state=0, class_weight="balanced"),
        "RandomForest": RandomForestClassifier(random_state=0),
        "SVM_linear": SVC(kernel="linear", probability=True, random_state=0),
        "SVM_rbf": SVC(kernel="rbf", probability=True, random_state=0),
        "CNN": TorchClassifier(CNN, feature_dim, epochs=10),
        "MLP": MLPClassifier(hidden_layer_sizes=(128,), max_iter=100, random_state=0),
    }

    artifacts_dir = None
    if args.save_eval_artifacts:
        artifacts_dir = args.eval_artifacts_dir
        os.makedirs(artifacts_dir, exist_ok=True)
        print(f"Startup | Eval artifacts: enabled | Dir: {artifacts_dir}", flush=True)

    os.makedirs(args.output_dir, exist_ok=True)
    for method_name, clf in methods.items():
        method_metrics: Dict[str, Dict[str, float]] = {}
        all_true = []
        all_pred = []
        all_prob = []
        all_fold_ids = []

        for fold_id, (train_idx, val_idx, test_idx) in enumerate(folds):
            print(f"Method: {method_name} | Fold: {fold_id} | Status: start", flush=True)
            x_train, y_train = v_freq[train_idx], smp_labels[train_idx]
            x_test, y_test = v_freq[test_idx], smp_labels[test_idx]

            print(
                (
                    f"Method: {method_name} | Fold: {fold_id} | "
                    f"Train size: {len(train_idx)} | Val size: {len(val_idx)} | Test size: {len(test_idx)}"
                ),
                flush=True,
            )

            selection_note = "train_only"
            if method_name == "MLP" and len(val_idx) > 0:
                x_val, y_val = v_freq[val_idx], smp_labels[val_idx]
                model, selection_info = train_mlp_with_validation(
                    x_train=x_train,
                    y_train=y_train,
                    x_val=x_val,
                    y_val=y_val,
                    max_epochs=100,
                    patience=5,
                    random_state=0,
                )
                if np.isnan(selection_info["best_val_auc"]):
                    selection_note = (
                        f"val_selected | best_epoch={int(selection_info['best_epoch'])} | "
                        f"best_val_auc=nan | stop_epoch={int(selection_info['stop_epoch'])}"
                    )
                else:
                    selection_note = (
                        f"val_selected | best_epoch={int(selection_info['best_epoch'])} | "
                        f"best_val_auc={selection_info['best_val_auc']:.4f} | "
                        f"stop_epoch={int(selection_info['stop_epoch'])}"
                    )
            else:
                model = clone(clf).fit(x_train, y_train)

            y_prob = model.predict_proba(x_test)
            y_pred = y_prob.argmax(axis=1)

            method_metrics[f"fold{fold_id}"] = {
                "acc": accuracy_score(y_test, y_pred),
                "auc": roc_auc_score(y_test, y_prob[:, 1]),
                "f1": f1_score(y_test, y_pred),
            }
            print(
                (
                    f"Method: {method_name} | Fold: {fold_id} | "
                    f"Selection: {selection_note} | "
                    f"Test AUC: {method_metrics[f'fold{fold_id}']['auc']:.4f} | "
                    f"Test ACC: {method_metrics[f'fold{fold_id}']['acc']:.4f} | "
                    f"Test F1: {method_metrics[f'fold{fold_id}']['f1']:.4f}"
                ),
                flush=True,
            )

            all_true.append(y_test.copy())
            all_pred.append(y_pred.copy())
            all_prob.append(y_prob[:, 1].copy())
            all_fold_ids.append(np.full(shape=len(y_test), fill_value=fold_id, dtype=int))

        save_metrics_to_csv(method_metrics, method_name=method_name, output_dir=args.output_dir)

        prediction_file = save_method_predictions(
            output_dir=args.output_dir,
            method_name=method_name,
            y_true=np.concatenate(all_true),
            y_pred=np.concatenate(all_pred),
            y_prob=np.concatenate(all_prob),
            fold_ids=np.concatenate(all_fold_ids),
        )
        print(f"Method: {method_name} | Saved predictions: {prediction_file}", flush=True)

        if artifacts_dir is not None:
            outputs = save_confusion_matrix_artifacts(
                output_dir=artifacts_dir,
                method_name=method_name,
                y_true=np.concatenate(all_true),
                y_pred=np.concatenate(all_pred),
            )
            print(f"Method: {method_name} | Saved confusion artifacts: {outputs}", flush=True)


if __name__ == "__main__":
    main()
