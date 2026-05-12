import os
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_fold_predictions(
    output_dir: str,
    method_name: str,
    fold_id: int,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
) -> str:
    _ensure_dir(output_dir)
    data = {
        "fold": np.full(shape=len(y_true), fill_value=fold_id, dtype=int),
        "y_true": y_true.astype(int),
        "y_pred": y_pred.astype(int),
    }
    if y_prob is not None:
        data["y_prob_pos"] = y_prob.astype(float)
    df = pd.DataFrame(data)
    file_path = os.path.join(output_dir, f"{method_name}_fold{fold_id}_predictions.csv")
    df.to_csv(file_path, index=False, float_format="%.6f")
    return file_path


def save_method_predictions(
    output_dir: str,
    method_name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    fold_ids: Optional[np.ndarray] = None,
) -> str:
    _ensure_dir(output_dir)
    data = {
        "y_true": y_true.astype(int),
        "y_pred": y_pred.astype(int),
    }
    if fold_ids is not None:
        data["fold"] = fold_ids.astype(int)
    if y_prob is not None:
        data["y_prob_pos"] = y_prob.astype(float)
    df = pd.DataFrame(data)
    file_path = os.path.join(output_dir, f"{method_name}_predictions.csv")
    df.to_csv(file_path, index=False, float_format="%.6f")
    return file_path


def save_confusion_matrix_artifacts(
    output_dir: str,
    method_name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Tuple[str, str] = ("CMV-", "CMV+"),
) -> Dict[str, str]:
    _ensure_dir(output_dir)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    cm_norm = cm.astype(float) / np.maximum(cm.sum(axis=1, keepdims=True), 1.0)

    cm_df = pd.DataFrame(
        cm,
        index=[f"true_{class_names[0]}", f"true_{class_names[1]}"],
        columns=[f"pred_{class_names[0]}", f"pred_{class_names[1]}"],
    )
    cm_path = os.path.join(output_dir, f"{method_name}_confusion_matrix.csv")
    cm_df.to_csv(cm_path)

    cm_norm_df = pd.DataFrame(
        cm_norm,
        index=[f"true_{class_names[0]}", f"true_{class_names[1]}"],
        columns=[f"pred_{class_names[0]}", f"pred_{class_names[1]}"],
    )
    cm_norm_path = os.path.join(output_dir, f"{method_name}_confusion_matrix_normalized.csv")
    cm_norm_df.to_csv(cm_norm_path)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, matrix, title, fmt in [
        (axes[0], cm, f"{method_name} (Counts)", "d"),
        (axes[1], cm_norm, f"{method_name} (Row-normalized)", ".2f"),
    ]:
        im = ax.imshow(matrix, cmap="Blues")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(class_names)
        ax.set_yticklabels(class_names)
        ax.set_xlabel("Predicted label")
        ax.set_ylabel("True label")
        ax.set_title(title)
        for i in range(2):
            for j in range(2):
                value = matrix[i, j]
                text = format(int(value), fmt) if fmt == "d" else format(value, fmt)
                ax.text(j, i, text, ha="center", va="center", color="black")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    fig_path = os.path.join(output_dir, f"{method_name}_confusion_matrix.png")
    plt.savefig(fig_path, dpi=300)
    plt.close(fig)

    return {
        "cm_csv": cm_path,
        "cm_normalized_csv": cm_norm_path,
        "cm_png": fig_path,
    }
