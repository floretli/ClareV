import torch
from loguru import logger
from tqdm import tqdm
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
import os
import numpy as np
import pandas as pd


class CLF_Trainer:

    @staticmethod
    def train(model, device, loader, optimizer, args, calc_loss, scheduler=None):
        """
        Train the model for one epoch.

        :param model: The model to train
        :param device: The device to use for training (e.g., 'cuda' or 'cpu')
        :param loader: The DataLoader for the training dataset
        :param optimizer: The optimizer to use during training
        :param args: A namespace containing various training settings (e.g., learning rate, batch size)
        :param calc_loss: A function to calculate the loss given model predictions and ground truth labels
        :param scheduler: The learning rate scheduler, if any (default: None)
        :return: The average training loss for this epoch
        """

        model.train()

        loss_accum = 0
        t = tqdm(loader, desc="Train")

        for step, batch in enumerate(t):

            batch = batch.to(device)
            if batch.x.shape[0] == 1 or batch.batch[-1] == 0:
                pass
            else:
                optimizer.zero_grad()
                pred_list = model(batch)

                loss = calc_loss(pred_list, batch)

                loss.backward()
                if args.grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()

                if scheduler:
                    scheduler.step()

                detached_loss = loss.item()
                loss_accum += detached_loss
                t.set_description(f"Train (loss = {detached_loss:.4f}, smoothed = {loss_accum / (step + 1):.4f})")

        logger.info("Average training loss: {:.4f}".format(loss_accum / (step + 1)))
        return loss_accum / (step + 1)

    @staticmethod
    def loss_CE(pred, batch):
        loss = F.cross_entropy(pred, batch.y)  ## for classification
        return loss

    @staticmethod
    def train_epochs(model, device, loaders, optimizer, calc_loss, run_name, run_id, args,
                     scheduler=None, eval_metric="auc", metrics=["acc", "auc"] ):

        train_loader, train_loader_eval, valid_loader, test_loader = loaders[run_id]

        best_val, final_test = 0, 0

        for epoch in range(1, args.epochs + 1):
            logger.info(f"=====Epoch {epoch}=====")
            loss = CLF_Trainer.train(model, device, train_loader, optimizer, args, calc_loss,
                              scheduler if args.scheduler != "plateau" else None)

            model.epoch_callback(epoch)
            logger.info(
                f"train loss: {loss:.4f}, train/lr: {optimizer.param_groups[0]['lr']:.4f}, epoch: {epoch}")

            if epoch > args.start_eval:
                logger.info("Evaluating...")
                with torch.no_grad():
                    train_perf = CLF_Trainer.evaluate(model, device, train_loader_eval)
                    valid_perf = CLF_Trainer.evaluate(model, device, valid_loader)
                    if args.use_test_data:
                        test_perf = CLF_Trainer.evaluate(model, device, test_loader)
                    else:
                        test_perf = None

                train_metric, valid_metric = (
                    train_perf[eval_metric],
                    valid_perf[eval_metric],
                )
                test_metric = test_perf[eval_metric] if test_perf is not None else None

                logger.info(f"Running: {run_name} (runs {run_id})")
                for metric in metrics:
                    if test_perf is not None:
                        logger.info(
                            f"Run {run_id} - {metric} - train: {train_perf[metric]:.4f}, val: {valid_perf[metric]:.4f}, test: {test_perf[metric]:.4f}")
                    else:
                        logger.info(
                            f"Run {run_id} - {metric} - train: {train_perf[metric]:.4f}, val: {valid_perf[metric]:.4f}")

                # Save checkpoints
                state_dict = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch}
                state_dict["scheduler"] = scheduler.state_dict() if args.scheduler else None

                torch.save(state_dict, os.path.join(args.save_path, "fold-" + str(run_id), "last_model.pt"))

                if best_val < valid_metric:
                    best_val = valid_metric
                    if test_metric is not None:
                        final_test = test_metric

                    logger.info(f"best/valid/: {eval_metric} - runs {run_id}")
                    torch.save(state_dict, os.path.join(args.save_path, "fold-" + str(run_id), "best_model.pt"))
                    logger.info("[Best Model] Save model: {}",
                                os.path.join(args.save_path, "fold-" + str(run_id), "best_model.pt"))
                    logger.info("final test: {:.4f}".format(final_test))

        state_dict = torch.load(os.path.join(args.save_path, "fold-" + str(run_id), "best_model.pt"))
        model.load_state_dict(state_dict["model"])
        best_valid_perf, case_ids, true_labels, pred_probs= CLF_Trainer.evaluate(model, device, valid_loader, show_results=True)
        best_test_perf = CLF_Trainer.evaluate(model, device, test_loader) if args.use_test_data else None

        # Save the test predictions to a CSV file
        predictions_df = pd.DataFrame({"case_id": case_ids, "true_label": true_labels, "pred_prob": pred_probs})
        predictions_df.to_csv(os.path.join(args.save_path, "fold-" + str(run_id), "val_predictions.csv"), index=False)

        return best_valid_perf, best_test_perf

    @staticmethod
    @torch.no_grad()
    def evaluate(model, device, loader, show_results=False):
        """
        Evaluate the model on a given dataset.
        :return: A dictionary containing evaluation metrics (e.g., accuracy and AUC), predicted probabilities, and case_ids
        """
        model.eval()

        correct = 0
        pred_probs = []
        true_labels = []
        case_ids = []

        for step, batch in enumerate(tqdm(loader, desc="Eval")):
            batch = batch.to(device)
            pred = model(batch)
            pred_probs += F.softmax(pred, dim=1)[:, 1].cpu().tolist()
            true_labels += batch.y.cpu().tolist()
            case_ids += batch.name  # Assuming your batch contains case_id
            pred = pred.max(dim=1)[1]
            correct += pred.eq(batch.y).sum().item()

        if show_results:
            return {"acc": correct / len(loader.dataset),
                    "auc": roc_auc_score(true_labels, pred_probs)}, case_ids, true_labels, pred_probs
        else:
            return {"acc": correct / len(loader.dataset),
                    "auc": roc_auc_score(true_labels, pred_probs)}



    @staticmethod
    def eval_metrics(metrics, all_valid_results, all_test_results, args):
        vals, tests = {}, {}
        for metric in metrics:
            if metric not in vals:
                vals[metric] = []
                if args.use_test_data:
                    tests[metric] = []
            for valid_perf in all_valid_results:
                vals[metric].append(valid_perf[metric])
                if args.use_test_data:
                    for test_perf in all_test_results:
                        tests[metric].append(test_perf[metric])

        for metric in metrics:
            log_message = f"Average val {metric}: {np.mean(vals[metric]):.4f} ± {np.std(vals[metric]):.4f}"
            if args.use_test_data:
                log_message += f" # test {metric}: {np.mean(tests[metric]):.4f} ± {np.std(tests[metric]):.4f}"
            logger.info(log_message)
        # return vals, tests
