import torch
import os
import numpy as np
import pandas as pd

from clarev.data_process import bagdata_to_vfreq
from clarev.data_loaders.bag_data_loader import ClusterBagDataset, RepertoireDataset
from clarev.models.contrastive_model import ContrastiveModel
from clarev.utils.utils import save_pk, load_pk
from clarev.trainers.CL_evaluation import vfeature_classification

from sklearn.model_selection import KFold
from torch.utils.data import Subset
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.neural_network import MLPClassifier


n_features = 36
class CNN(nn.Module):
    def __init__(self, input_dim):
        super(CNN, self).__init__()
        self.conv = nn.Conv1d(1, 8, kernel_size=3)
        self.fc = nn.Linear(8*(input_dim-2), 1)
        
    def forward(self, x):
        x = x.unsqueeze(1)  # [batch, 1, features]
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return torch.sigmoid(self.fc(x))

class TorchClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, model_class, input_dim, epochs=10, lr=0.0001):
        self.model_class = model_class
        self.input_dim = input_dim
        self.epochs = epochs
        self.lr = lr
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def fit(self, X, y):
        X = torch.FloatTensor(X).to(self.device)
        y = torch.FloatTensor(y).to(self.device)
        
        self.model = self.model_class(self.input_dim).to(self.device)
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.BCELoss()
        
        dataset = TensorDataset(X, y)
        loader = DataLoader(dataset, batch_size=32, shuffle=True)
        
        for epoch in range(self.epochs):
            for batch_X, batch_y in loader:
                optimizer.zero_grad()
                outputs = self.model(batch_X).squeeze()
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
        return self
    
    def predict_proba(self, X):
        X = torch.FloatTensor(X).to(self.device)
        with torch.no_grad():
            proba = self.model(X).cpu().numpy()
        return np.hstack([1-proba, proba])
    
def bagdata_to_vfreq(superbags, v_gene_order, smp_list, norm=True):
    v_freq = np.zeros((len(smp_list), len(v_gene_order)))
    for i, smp in enumerate(superbags):
        for v_idx in range(len(v_gene_order)):
            v_freq[i][v_idx] = len(smp[v_idx])

    if norm:
        v_freq = v_freq / v_freq.sum(axis=1, keepdims=True)
    # v_freq_df = pd.DataFrame(v_freq, columns=v_gene_order, index=smp_list)
    return v_freq

def classificaiton_by_vfreq(v_freq, smp_labels, kf, result_dir = './results/'):

    methods = {
        'LogisticRegression': LogisticRegression(random_state=0, class_weight='balanced'),
        'RandomForest': RandomForestClassifier(random_state=0),
        'SVM_linear': SVC(kernel='linear', probability=True, random_state=0),
        'SVM_rbf': SVC(kernel='rbf', probability=True, random_state=0),
        'CNN': TorchClassifier(CNN, n_features, epochs=10),
        'MLP': MLPClassifier(
        hidden_layer_sizes=(128,),
        max_iter=100,
        random_state=0,
    )
    }


    results = {
        method_name: {
            'fold': [],
            'acc': [],
            'auc': [],
            'f1': []
        } for method_name in methods.keys()
    }

    for fold_id, (train_idx, test_idx) in enumerate(kf.split(v_freq)):
        print(f"\n=== Processing fold {fold_id} ===")

        
        X_train, X_test = v_freq[train_idx], v_freq[test_idx]
        y_train, y_test = smp_labels[train_idx], smp_labels[test_idx]

        # print(f"Positive samples in test: {np.sum(y_test)}")
        # print(f"Negative samples in test: {len(y_test) - np.sum(y_test)}")

        for method_name, clf in methods.items():
            model = clf.fit(X_train, y_train)
            
            y_prob = model.predict_proba(X_test)
            y_pred = y_prob.argmax(axis=1)

            acc = accuracy_score(y_test, y_pred)
            auc = roc_auc_score(y_test, y_prob[:, 1])
            f1 = f1_score(y_test, y_pred)
            
            results[method_name]['fold'].append(fold_id)
            results[method_name]['acc'].append(acc)
            results[method_name]['auc'].append(auc)
            results[method_name]['f1'].append(f1)

    for method_name, data in results.items():
        df = pd.DataFrame(data)
        
        stats = pd.DataFrame({
            'fold': ['Mean', 'Std'],
            'acc': [np.mean(df.acc), np.std(df.acc)],
            'auc': [np.mean(df.auc), np.std(df.auc)],
            'f1': [np.mean(df.f1), np.std(df.f1)],
        })
        
        final_df = pd.concat([df, stats], ignore_index=True)
        print(f"\n=== {method_name} ===")
        print(final_df)

        filename = f"{method_name}_results.csv"
        filename = os.path.join(result_dir, filename)
        final_df.to_csv(filename, index=False, float_format="%.4f")
        print(f"Saved {filename}")

def save_metrics_to_csv(metrics_dict, method_name, output_dir="./results/"):

    records = []
    for fold, metrics in metrics_dict.items():
        record = {'fold': fold}
        record.update(metrics)
        records.append(record)
    

    df = pd.DataFrame(records)
    
    stats = pd.DataFrame({
        'fold': ['Mean', 'Std'],
        'acc': [df['acc'].mean(), df['acc'].std()],
        'auc': [df['auc'].mean(), df['auc'].std()],
        'f1': [df['f1'].mean(), df['f1'].std()],
    })
    
    final_df = pd.concat([df, stats], ignore_index=True)
    print(f"\n=== {method_name} ===")
    print(final_df)
    
    filename = os.path.join(output_dir, f"{method_name}_results.csv")
    final_df.to_csv(filename, index=False, float_format="%.4f")
    print(f"Saved {filename}")
    return final_df


def main():
    ## =============== parameters =================

    ## input data dirs
    bag_data_dir = './data/processed_data/Emerson2017_vgene/'

    ## parameters for training and testing cl models
    trained_model_dir = './trained_models/Emerson2017_vgene_weight/'
    num_epochs = 30
    batch_size = 16
    test_batch_size = 1
    vgene_emb_dim = 120
    tcr_emb_dim = 120
    subbag_size = 50
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    result_dir = './results/cmv_classification_vgene_weight/'
    os.makedirs(result_dir, exist_ok=True)
    
    n_splits = 5
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    

    ## =========  load existing bag data  ===========
    superbags = load_pk(os.path.join(bag_data_dir, "superbags.pk"))
    v_gene_order = load_pk(os.path.join(bag_data_dir, "v_gene_order.pk"))
    smp_list = load_pk(os.path.join(bag_data_dir, "smp_list.pk"))
    smp_labels = load_pk(os.path.join(bag_data_dir, "smp_labels.pk"))
    weights = load_pk(os.path.join(bag_data_dir, "weights.pk"))
    
    print(f"loaded {len(superbags)} samples with {len(v_gene_order)} v genes from the input data directory")

    num_vgene = len(v_gene_order)
    v_freq = bagdata_to_vfreq(superbags, v_gene_order, smp_list, norm=True)  ## v_freq is a numpy array
    folds = kf.split(v_freq)

    ## =============== ML classification by vfreq features =================
    classificaiton_by_vfreq(v_freq, smp_labels, kf, result_dir = result_dir)

    ## =============== load cl model =================
    best_model = ContrastiveModel(input_dim=tcr_emb_dim, output_dim=vgene_emb_dim).to(device)
    best_model.load_state_dict(torch.load(os.path.join(trained_model_dir, 'best_model.pth'), map_location=device))

    whole_dataset = RepertoireDataset(superbags, labels = smp_labels, v_freq_mtx=v_freq, weight= weights, subbag_size = subbag_size, )

    ## =============== classification by vbag + vfreq features =================
    all_metrics = {}
    for fold_id, (train_idx, test_idx) in enumerate(folds):
        print(f"fold {fold_id}")
        train_dataset = Subset(whole_dataset, train_idx)
        test_dataset = Subset(whole_dataset, test_idx)

        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_dataloader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)

        fold_metrics = vfeature_classification(train_dataloader, test_dataloader, best_model, device, emb_dim = vgene_emb_dim, 
                                               num_vgene = num_vgene, num_epochs = num_epochs, class_num = 2, use_vfreq=True)
        print(fold_metrics)
        all_metrics[f'fold{fold_id}'] = {
            'acc': fold_metrics['acc'],
            'auc': fold_metrics['auc'],
            'f1': fold_metrics['f1']
        }
    save_metrics_to_csv(all_metrics, method_name="Vbag_Vfreq_feature_MLP", output_dir=result_dir)

    ## =============== classification by vbag features only  =================

    all_metrics = {}
    for fold_id, (train_idx, test_idx) in enumerate(folds):
        print(f"fold {fold_id}")
        train_dataset = Subset(whole_dataset, train_idx)
        test_dataset = Subset(whole_dataset, test_idx)

        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_dataloader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)

        fold_metrics = vfeature_classification(train_dataloader, test_dataloader, best_model, device, emb_dim = vgene_emb_dim, 
                                               num_vgene = num_vgene, num_epochs = num_epochs, class_num = 2, use_vfreq=False)
        print(fold_metrics)
        all_metrics[f'fold{fold_id}'] = {
            'acc': fold_metrics['acc'],
            'auc': fold_metrics['auc'],
            'f1': fold_metrics['f1']
        }
    save_metrics_to_csv(all_metrics, method_name="Vbag_feature_MLP", output_dir=result_dir)



if __name__ == "__main__":
    main()