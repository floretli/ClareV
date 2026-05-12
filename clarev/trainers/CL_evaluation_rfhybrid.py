import copy
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from clarev.models.rf_hybrid_fusion import RFFusionClassifier
from clarev.trainers.CL_evaluation import tensor_to_vfeature


def _collect_vfreq_labels(loader):
    x_list, y_list = [], []
    for _, batch_labels, batch_vfreq in loader:
        x_list.append(batch_vfreq.detach().cpu().numpy())
        y_list.append(batch_labels.detach().cpu().numpy())
    X = np.concatenate(x_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    return X, y


def _evaluate_loader_rfhybrid(
    classifier,
    loader,
    encoder,
    rf_model,
    criterion,
    device,
    num_vgene,
    emb_dim,
):
    classifier.eval()
    preds, pred_probs, labels = [], [], []
    total_loss = 0.0

    with torch.no_grad():
        for batch_ebd, batch_labels, batch_vfreq in loader:
            batch_ebd = batch_ebd.to(device)
            batch_labels = batch_labels.to(device)

            if encoder:
                batch_ebd = tensor_to_vfeature(batch_ebd, encoder)
            else:
                assert batch_ebd.size(1) == num_vgene
                assert batch_ebd.size(2) == emb_dim

            rf_prob_np = rf_model.predict_proba(batch_vfreq.detach().cpu().numpy())[:, 1]
            rf_prob = torch.tensor(rf_prob_np, dtype=torch.float32, device=device).unsqueeze(1)

            outputs = classifier(batch_ebd, rf_prob)
            loss = criterion(outputs, batch_labels)
            probs = torch.softmax(outputs, dim=1)
            predicted = torch.argmax(probs, dim=1)

            total_loss += loss.item() * len(batch_labels)
            preds.append(predicted.cpu().numpy())
            pred_probs.append(probs[:, 1].cpu().numpy())
            labels.append(batch_labels.cpu().numpy())

    total_loss /= len(loader.dataset)
    preds = np.concatenate(preds)
    pred_probs = np.concatenate(pred_probs)
    labels = np.concatenate(labels)

    try:
        auc = roc_auc_score(labels, pred_probs)
    except ValueError:
        auc = np.nan

    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds)
    return {
        "loss": total_loss,
        "auc": auc,
        "acc": acc,
        "f1": f1,
        "labels": labels,
        "preds": preds,
        "pred_probs": pred_probs,
    }


def vfeature_classification_rfhybrid(
    train_loader,
    test_loader,
    encoder,
    device,
    emb_dim=120,
    num_vgene=21,
    num_epochs=20,
    class_num=2,
    use_vfreq=True,
    return_details=False,
    val_loader=None,
    nested_mode=False,
    early_stopping_patience: Optional[int] = None,
    rf_n_estimators: int = 300,
    rf_seed: int = 1,
    clf_lr: float = 1e-4,
    clf_weight_decay: float = 1e-5,
    rffusion_agg_type: str = "flatten",
    rffusion_per_v_dim: int = 16,
    rffusion_bottleneck_dim: int = 128,
):
    if nested_mode and val_loader is None:
        raise ValueError("val_loader is required when nested_mode=True")

    X_train_vfreq, y_train = _collect_vfreq_labels(train_loader)
    rf_model = RandomForestClassifier(
        n_estimators=rf_n_estimators,
        random_state=rf_seed,
        n_jobs=-1,
        class_weight="balanced",
    )
    rf_model.fit(X_train_vfreq, y_train)

    classifier = RFFusionClassifier(
        v_num=num_vgene,
        emb_dim=emb_dim,
        class_num=class_num,
        agg_type=rffusion_agg_type,
        per_v_dim=rffusion_per_v_dim,
        bottleneck_dim=rffusion_bottleneck_dim,
    ).to(device)
    optimizer = torch.optim.AdamW(classifier.parameters(), lr=clf_lr, weight_decay=clf_weight_decay)
    criterion = nn.CrossEntropyLoss()

    if encoder:
        encoder = encoder.to(device)
        encoder.eval()
    else:
        encoder = None

    best_metrics = {"auc": 0.0, "acc": 0.0, "f1": 0.0, "epoch": 0, "loss": np.inf}
    best_state_dict = None
    best_details = None
    best_test_metrics_by_val = None
    last_epoch_test_metrics = None
    no_improve_epochs = 0
    epoch_logs = []

    for epoch in range(num_epochs):
        train_loss = 0.0
        train_correct = 0
        train_pred_probs = []
        train_labels_all = []
        classifier.train()

        for batch_ebd, batch_labels, batch_vfreq in train_loader:
            batch_ebd = batch_ebd.to(device)
            batch_labels = batch_labels.to(device)

            if encoder:
                batch_ebd = tensor_to_vfeature(batch_ebd, encoder)
            else:
                assert batch_ebd.size(1) == num_vgene
                assert batch_ebd.size(2) == emb_dim

            rf_prob_np = rf_model.predict_proba(batch_vfreq.detach().cpu().numpy())[:, 1]
            rf_prob = torch.tensor(rf_prob_np, dtype=torch.float32, device=device).unsqueeze(1)

            optimizer.zero_grad()
            outputs = classifier(batch_ebd, rf_prob)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()

            predicted = torch.argmax(outputs.data, dim=1)
            train_probs = torch.softmax(outputs, dim=1)[:, 1]
            train_correct += (predicted == batch_labels).sum().item()
            train_loss += loss.item() * len(batch_labels)
            train_pred_probs.append(train_probs.detach().cpu().numpy())
            train_labels_all.append(batch_labels.detach().cpu().numpy())

        train_loss /= len(train_loader.dataset)
        train_accuracy = train_correct / len(train_loader.dataset)
        train_labels_all = np.concatenate(train_labels_all)
        train_pred_probs = np.concatenate(train_pred_probs)
        try:
            train_auc = roc_auc_score(train_labels_all, train_pred_probs)
        except ValueError:
            train_auc = np.nan

        eval_loader = val_loader if nested_mode else test_loader
        eval_metrics = _evaluate_loader_rfhybrid(
            classifier=classifier,
            loader=eval_loader,
            encoder=encoder,
            rf_model=rf_model,
            criterion=criterion,
            device=device,
            num_vgene=num_vgene,
            emb_dim=emb_dim,
        )
        eval_auc = eval_metrics["auc"]
        improved = (not np.isnan(eval_auc)) and (eval_auc > best_metrics["auc"])

        if improved:
            best_metrics.update(
                {
                    "epoch": epoch + 1,
                    "auc": eval_metrics["auc"],
                    "acc": eval_metrics["acc"],
                    "f1": eval_metrics["f1"],
                    "loss": eval_metrics["loss"],
                }
            )
            best_state_dict = copy.deepcopy(classifier.state_dict())
            no_improve_epochs = 0
            if return_details and not nested_mode:
                best_details = {
                    "labels": eval_metrics["labels"].copy(),
                    "preds": eval_metrics["preds"].copy(),
                    "pred_probs": eval_metrics["pred_probs"].copy(),
                }
        else:
            no_improve_epochs += 1

        if nested_mode:
            test_metrics_epoch = _evaluate_loader_rfhybrid(
                classifier=classifier,
                loader=test_loader,
                encoder=encoder,
                rf_model=rf_model,
                criterion=criterion,
                device=device,
                num_vgene=num_vgene,
                emb_dim=emb_dim,
            )
            last_epoch_test_metrics = test_metrics_epoch
            if improved:
                best_test_metrics_by_val = {
                    "loss": test_metrics_epoch["loss"],
                    "auc": test_metrics_epoch["auc"],
                    "acc": test_metrics_epoch["acc"],
                    "f1": test_metrics_epoch["f1"],
                }
            print(
                (
                    "Epoch [{}/{}] Train Loss: {:.4f}, Train ACC: {:.4f}, Train AUC: {:.4f} | "
                    "Val Loss: {:.4f}, Val AUC: {:.4f}, Val ACC: {:.4f} | "
                    "Test Loss: {:.4f}, Test AUC: {:.4f}, Test ACC: {:.4f} | "
                    "Best Val AUC: {:.4f} (epoch {})"
                ).format(
                    epoch + 1,
                    num_epochs,
                    train_loss,
                    train_accuracy,
                    train_auc,
                    eval_metrics["loss"],
                    eval_metrics["auc"],
                    eval_metrics["acc"],
                    test_metrics_epoch["loss"],
                    test_metrics_epoch["auc"],
                    test_metrics_epoch["acc"],
                    best_metrics["auc"],
                    best_metrics["epoch"],
                ),
                flush=True,
            )
            epoch_logs.append(
                {
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "train_acc": train_accuracy,
                    "train_auc": train_auc,
                    "val_loss": eval_metrics["loss"],
                    "val_auc": eval_metrics["auc"],
                    "val_acc": eval_metrics["acc"],
                    "test_loss": test_metrics_epoch["loss"],
                    "test_auc": test_metrics_epoch["auc"],
                    "test_acc": test_metrics_epoch["acc"],
                    "best_val_auc": best_metrics["auc"],
                    "best_val_epoch": best_metrics["epoch"],
                }
            )
        else:
            print(
                (
                    "Epoch [{}/{}] Train Loss: {:.4f}, Train ACC: {:.4f}, Train AUC: {:.4f} | "
                    "Val Loss: {:.4f}, Val AUC: {:.4f}, Val ACC: {:.4f} | "
                    "Best Val AUC: {:.4f} (epoch {})"
                ).format(
                    epoch + 1,
                    num_epochs,
                    train_loss,
                    train_accuracy,
                    train_auc,
                    eval_metrics["loss"],
                    eval_metrics["auc"],
                    eval_metrics["acc"],
                    best_metrics["auc"],
                    best_metrics["epoch"],
                ),
                flush=True,
            )
            epoch_logs.append(
                {
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "train_acc": train_accuracy,
                    "train_auc": train_auc,
                    "val_loss": eval_metrics["loss"],
                    "val_auc": eval_metrics["auc"],
                    "val_acc": eval_metrics["acc"],
                    "test_loss": np.nan,
                    "test_auc": np.nan,
                    "test_acc": np.nan,
                    "best_val_auc": best_metrics["auc"],
                    "best_val_epoch": best_metrics["epoch"],
                }
            )

        if (
            early_stopping_patience is not None
            and early_stopping_patience > 0
            and no_improve_epochs >= early_stopping_patience
        ):
            print(
                (
                    "Early stop triggered at epoch {} | "
                    "No Val AUC improvement for {} epochs"
                ).format(epoch + 1, early_stopping_patience),
                flush=True,
            )
            break

    if nested_mode:
        if best_state_dict is not None:
            classifier.load_state_dict(best_state_dict)
        test_metrics = _evaluate_loader_rfhybrid(
            classifier=classifier,
            loader=test_loader,
            encoder=encoder,
            rf_model=rf_model,
            criterion=criterion,
            device=device,
            num_vgene=num_vgene,
            emb_dim=emb_dim,
        )
        best_metrics["loss"] = test_metrics["loss"]
        best_metrics["auc"] = test_metrics["auc"]
        best_metrics["acc"] = test_metrics["acc"]
        best_metrics["f1"] = test_metrics["f1"]
        print(
            (
                "Final Test Loss: {:.4f}, Test AUC: {:.4f}, Test ACC: {:.4f}, Test F1: {:.4f} | "
                "Selected by Best Val AUC epoch {}"
            ).format(
                test_metrics["loss"],
                test_metrics["auc"],
                test_metrics["acc"],
                test_metrics["f1"],
                best_metrics["epoch"],
            ),
            flush=True,
        )
        if best_test_metrics_by_val is not None:
            print(
                (
                    "Test at selected epoch {} -> Loss: {:.4f}, AUC: {:.4f}, ACC: {:.4f}, F1: {:.4f}"
                ).format(
                    best_metrics["epoch"],
                    best_test_metrics_by_val["loss"],
                    best_test_metrics_by_val["auc"],
                    best_test_metrics_by_val["acc"],
                    best_test_metrics_by_val["f1"],
                ),
                flush=True,
            )
        if last_epoch_test_metrics is not None:
            print(
                (
                    "Test at last epoch {} -> Loss: {:.4f}, AUC: {:.4f}, ACC: {:.4f}, F1: {:.4f}"
                ).format(
                    num_epochs,
                    last_epoch_test_metrics["loss"],
                    last_epoch_test_metrics["auc"],
                    last_epoch_test_metrics["acc"],
                    last_epoch_test_metrics["f1"],
                ),
                flush=True,
            )
        if return_details:
            best_details = {
                "labels": test_metrics["labels"].copy(),
                "preds": test_metrics["preds"].copy(),
                "pred_probs": test_metrics["pred_probs"].copy(),
            }

    best_metrics["epoch_logs"] = epoch_logs
    if return_details:
        if best_details is None and not nested_mode:
            final_eval = _evaluate_loader_rfhybrid(
                classifier=classifier,
                loader=test_loader,
                encoder=encoder,
                rf_model=rf_model,
                criterion=criterion,
                device=device,
                num_vgene=num_vgene,
                emb_dim=emb_dim,
            )
            best_details = {
                "labels": final_eval["labels"].copy(),
                "preds": final_eval["preds"].copy(),
                "pred_probs": final_eval["pred_probs"].copy(),
            }
        best_details["epoch_logs"] = epoch_logs
        return best_metrics, best_details

    return best_metrics
