import torch
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import os
from pathlib import Path
import matplotlib.pyplot as plt

# Define training and evaluation functions
class CL_Trainer:

    def __init__(self, model, optimizer, save_dir, device="cuda" if torch.cuda.is_available() else "cpu", loss_type="triplet", margin=0.1, alpha=1e-4):
        self.model = model
        self.optimizer = optimizer
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.loss_type = loss_type
        self.margin = margin
        self.alpha = alpha
        self.scaler = GradScaler()
        self.model.to(self.device)
        self.best_model_path = os.path.join(save_dir, "best_model.pth")
        self.train_log_path = os.path.join(save_dir, "training_log.png")



    @staticmethod
    def compute_loss(out1, out2, out3, loss_type="triplet", margin=0.1, alpha=1e-4):
        if loss_type == "triplet":
            cos_sim_pos = torch.nn.functional.cosine_similarity(out1, out2) - alpha
            cos_sim_neg = torch.nn.functional.cosine_similarity(out1, out3) + alpha
            loss = torch.relu(margin - cos_sim_pos + cos_sim_neg).mean()
        elif loss_type == "cosine":
            cos_sim_pos = torch.nn.functional.cosine_similarity(out1, out2)
            cos_sim_neg = torch.nn.functional.cosine_similarity(out1, out3)
            loss = -cos_sim_pos + cos_sim_neg + 2 * alpha
            loss = loss.mean()
        else:
            raise ValueError("Invalid loss_type. Choose between 'triplet' and 'cosine'.")

        return loss
    def train(self, train_loader, val_loader=None, epochs=10):

        self.model.train()
        best_val_loss = float('inf')
        all_metrics = {}
        all_metrics['train_loss'] = []
        all_metrics['val_loss'] = []
        all_metrics['val_acc'] = []

        
        for epoch in range(epochs):
            total_loss = 0.0
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
            
            for pos1, pos2, neg in progress_bar:
                pos1, pos2, neg = pos1.to(self.device), pos2.to(self.device), neg.to(self.device)
                
                self.optimizer.zero_grad()
                with autocast():
                    out1, out2 = self.model(pos1, pos2)
                    _, out3 = self.model(pos1, neg)
                    loss = self.compute_loss(out1, out2, out3)
                
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                
                total_loss += loss.item()
                progress_bar.set_postfix({"loss": loss.item()})
            
            avg_train_loss = total_loss / len(train_loader)
            val_metrics = self.evaluate(self.model, val_loader) if val_loader else {}
            
            if val_metrics.get('loss', float('inf')) < best_val_loss:
                best_val_loss = val_metrics['loss']
                self.save_model(self.best_model_path)

            print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | "
                  f"Val Loss: {val_metrics.get('loss', 'N/A'):.4f} | "
                  f"Val Acc: {val_metrics.get('acc', 'N/A'):.2%}")
            
            all_metrics['train_loss'].append(avg_train_loss)
            all_metrics['val_loss'].append(val_metrics.get('loss', 'N/A'))
            all_metrics['val_acc'].append(val_metrics.get('acc', 'N/A'))

        return self.model, all_metrics

    def record_metrics(self, metrics):
        ## draw plot for trian, val loss, val acc with epoch
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(metrics['train_loss'], label='Train Loss')
        plt.plot(metrics['val_loss'], label='Val Loss')
        plt.ylim(0, 0.1)
        plt.xticks(range(0, len(metrics['train_loss'])+1))
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(metrics['val_acc'], label='Val Acc')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.ylim(0.5, 1)
        plt.xticks(range(0, len(metrics['val_acc'])+1))
        
        plt.legend()

        plt.savefig(self.train_log_path)

    def evaluate(self, model, val_loader):
        
        model.eval()
        total_loss, correct, total = 0.0, 0, 0
        
        with torch.no_grad():
            for pos1, pos2, neg in tqdm(val_loader, desc="Validating"):
                pos1, pos2, neg = pos1.to(self.device), pos2.to(self.device), neg.to(self.device)
                
                with autocast():
                    out1, out2 = model(pos1, pos2)
                    _, out3 = model(pos1, neg)
                    loss = self.compute_loss(out1, out2, out3)
                
                total_loss += loss.item()
                cos_sim_pos = torch.nn.functional.cosine_similarity(out1, out2)
                cos_sim_neg = torch.nn.functional.cosine_similarity(out1, out3)
                correct += (cos_sim_pos > cos_sim_neg).sum().item()
                total += pos1.size(0)
        
        return {
            'loss': total_loss / len(val_loader),
            'acc': correct / total
        }

    def inference_embedding(self, dataloader):
        ## inference embedding for a bag of TCRs

        self.model.eval()
        embeddings = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Predicting"):
                if isinstance(batch, (list, tuple)):
                    inputs = batch[0].to(self.device)
                else:
                    inputs = batch.to(self.device)
                emb = self.model(inputs)
                embeddings.append(emb.cpu())
        
        return torch.cat(embeddings, dim=0)

    def save_model(self, path):
        Path(os.path.dirname(path)).mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), path)


    # @staticmethod
    # def train(model, train_loader, optimizer, val_loader=None, device = self.device, loss_type="triplet", margin=0.1):
    #     model.train()
    #     model.to(device)
    #     total_loss = 0.0
    #     scaler = GradScaler()  # for automatic mixed precision training
    #     progress_bar = tqdm(train_loader, desc="Training")

    #     for pos_sample1, pos_sample2, neg_sample in progress_bar:
    #         pos_sample1, pos_sample2, neg_sample = pos_sample1.to(device), pos_sample2.to(device), neg_sample.to(device)
    #         optimizer.zero_grad()

    #         with autocast():  # for automatic mixed precision training
    #             out1, out2 = model(pos_sample1, pos_sample2)
    #             _, out3 = model(pos_sample1, neg_sample)
    #             loss = CL_Trainer.compute_loss(out1, out2, out3, loss_type=loss_type, margin=margin)

    #         scaler.scale(loss).backward()
    #         scaler.step(optimizer)
    #         scaler.update()

    #         total_loss += loss.item()
    #         progress_bar.set_postfix({"loss": loss.item()})
    #     return total_loss / len(train_loader)

    # @staticmethod
    # def evaluate(model, dataloader, device = "cuda" if torch.cuda.is_available() else "cpu"):
    #     model.eval()
    #     model.to(device)
    #     with torch.no_grad():
    #         correct = 0
    #         total = 0
    #         progress_bar = tqdm(dataloader, desc="Evaluating")

    #         for pos_sample1, pos_sample2, neg_sample in progress_bar:
    #             pos_sample1, pos_sample2, neg_sample = pos_sample1.to(device), pos_sample2.to(device), neg_sample.to(
    #                 device)
    #             out1, out2 = model(pos_sample1, pos_sample2)
    #             _, out3 = model(pos_sample1, neg_sample)

    #             cos_sim_pos = torch.nn.functional.cosine_similarity(out1, out2)
    #             cos_sim_neg = torch.nn.functional.cosine_similarity(out1, out3)

    #             correct += torch.sum(cos_sim_pos > cos_sim_neg).item()
    #             total += pos_sample1.size(0)
    #             progress_bar.set_postfix({"accuracy": correct / total})

    #     return correct / total