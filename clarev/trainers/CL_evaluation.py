import torch.nn as nn
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
import numpy as np

class Embedding_classifier(nn.Module):
    def __init__(self, input_dim = 120, hidden_dim =128, output_dim = 2, dropout_prob = 0.2):
        super(Embedding_classifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x):
        x = self.dropout(torch.relu(self.fc1(x)))
        x = self.dropout(torch.relu(self.fc2(x)))
        x = self.fc3(x)
        return x

class Repertoire_classifier(nn.Module):
    def __init__(self, input_dim=21*120, hidden_dim=128, output_dim=2, dropout_prob=0.3):
        super(Repertoire_classifier, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x):
        x = self.flatten(x)  # [batch_size, 21*120]
        x = self.dropout(torch.relu(self.fc1(x)))
        x = self.fc3(x)  # output [batch_size, output_dim]
        return x

## dual model ===========
class DualPathClassifier(nn.Module):
    def __init__(self, v_num = 21, emb_dim =120, class_num = 2):
        super().__init__()
        
        self.main_path = Repertoire_classifier(input_dim= emb_dim*v_num, hidden_dim=128, output_dim=128)
        self.v_num = v_num

        self.freq_path = nn.Sequential(
            nn.Linear(v_num, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 64)
        )
        
        self.fusion = nn.Sequential(
            nn.Linear(128 + 64, 128),  # main_path output 128 + freq_path output 64
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, class_num)
        )
    
    def forward(self, x, freq):
        main_feat = self.main_path(x)  # [batch, 2]
        freq_feat = self.freq_path(freq)  # [batch, 64]
        combined = torch.cat([main_feat, freq_feat], dim=1)
        
        return self.fusion(combined)


def create_classfication_data(embeddings, labels, split_ratio = 0.8, batch_size = 16):  ## embeddings: [ torch.tensor ], labels : list

    train_size = int(split_ratio * len(embeddings))
    train_ebd = embeddings[:train_size]
    test_ebd = embeddings[train_size:]
    train_labels = labels[:train_size]
    test_labels = labels[train_size:]

    # convert to tensor
    train_ebd_tensor = torch.stack(train_ebd, dim=0)
    train_labels_tensor = torch.tensor(train_labels)
    test_ebd_tensor = torch.stack(test_ebd, dim=0)
    test_labels_tensor = torch.tensor(test_labels)

    train_c_dataset = TensorDataset(train_ebd_tensor, train_labels_tensor)
    train_c_dataloader = DataLoader(train_c_dataset, batch_size=batch_size, shuffle=True)

    test_c_dataset = TensorDataset(test_ebd_tensor, test_labels_tensor)
    test_c_dataloader = DataLoader(test_c_dataset, batch_size=batch_size, shuffle=False)

    return train_c_dataloader, test_c_dataloader

def vfeature_classification(train_loader, test_loader, encoder, device, emb_dim = 120, num_vgene = 21, num_epochs = 20, class_num = 2, use_vfreq=False):
    ## use embedding to classfy
    if use_vfreq:
        classifier = DualPathClassifier( v_num = num_vgene, emb_dim =emb_dim, class_num = class_num).to(device)
        optimizer = torch.optim.AdamW([
                    {'params': classifier.main_path.parameters(), 'lr': 1e-4},
                    {'params': classifier.freq_path.parameters(), 'lr': 1e-4},
                    {'params': classifier.fusion.parameters(), 'lr': 1e-4}
                    ], weight_decay=1e-5)
    else:
        # print("use V embedding only")
        classifier = Repertoire_classifier(emb_dim*num_vgene, 128, class_num, 0.2).to(device)
        optimizer = torch.optim.Adam(classifier.parameters(), lr=1e-4)

    criterion = nn.CrossEntropyLoss()

    if encoder:
        encoder = encoder.to(device)
        encoder.eval()
    else:
        encoder = None

    best_metrics = {
        'auc': 0.0,
        'acc': 0.0,
        'f1': 0.0,
        'epoch': 0,
        'loss': np.inf
    }

    for epoch in range(num_epochs):
        train_loss = 0.0
        train_correct = 0
        classifier.train()

        for batch_ebd, batch_labels, batch_vfreq in train_loader:
            batch_ebd = batch_ebd.to(device)
            batch_labels = batch_labels.to(device)
            batch_vfreq = batch_vfreq.to(device)

            if encoder:
                batch_ebd = tensor_to_vfeature(batch_ebd, encoder)
            else:
                assert batch_ebd.size(1) == num_vgene, "batch_ebd size is not equal to num_vgene"
                assert batch_ebd.size(2) == emb_dim,  "batch_ebd size is not equal to emb_dim"
                batch_ebd = batch_ebd  ## .view(-1, num_vgene, emb_dim)

            optimizer.zero_grad()
            if use_vfreq:
                outputs = classifier(batch_ebd, batch_vfreq)
            else:
                outputs = classifier(batch_ebd)

            loss = criterion(outputs, batch_labels)
            
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(outputs.data, 1)
            correct = (predicted == batch_labels).sum().item()
            train_correct += correct
            train_loss += loss.item() * len(batch_labels)
        ## length of train_loader.dataset
        train_loss /= len(train_loader.dataset)
        train_accuracy = train_correct / len(train_loader.dataset)

        ## validation
        classifier.eval()
        preds = []
        labels = []
        test_loss = 0.0
        with torch.no_grad():
            for batch_ebd, batch_labels, batch_vfreq in test_loader:
                batch_ebd = batch_ebd.to(device)
                batch_labels = batch_labels.to(device)
                batch_vfreq = batch_vfreq.to(device)

                if encoder:
                    batch_ebd = tensor_to_vfeature(batch_ebd, encoder)
                else:
                    assert batch_ebd.size(1) == num_vgene, "batch_ebd size is not equal to num_vgene"
                    assert batch_ebd.size(2) == emb_dim,  "batch_ebd size is not equal to emb_dim"
                    batch_ebd = batch_ebd

                if use_vfreq:
                    outputs = classifier(batch_ebd, batch_vfreq)
                else:
                    outputs = classifier(batch_ebd)

                loss = criterion(outputs, batch_labels)
                _, predicted = torch.max(outputs.data, 1)

                test_loss += loss.item() * len(batch_labels)

                preds.append(predicted.cpu().numpy())
                labels.append(batch_labels.cpu().numpy())

            test_loss /= len(test_loader.dataset)
            preds = np.concatenate(preds)
            labels = np.concatenate(labels)
            test_auc = roc_auc_score(labels, preds)
            test_accuracy = accuracy_score(labels, preds)
            test_f1 = f1_score(labels, preds)

            # if test_loss < best_metrics['loss']:
            if test_auc > best_metrics['auc']:
                best_metrics['loss'] = test_loss
                best_metrics['acc'] = test_accuracy
                best_metrics['auc'] = test_auc
                best_metrics['f1'] = test_f1
                best_metrics['epoch'] = epoch + 1

        print(
            'Epoch [{}/{}], Train Loss: {:.4f}, Train Accuracy: {:.4f}, Test Loss: {:.4f}'.format(
                epoch + 1, num_epochs, train_loss, train_accuracy, test_loss))

        # print(
        #     'Epoch [{}/{}], Train Loss: {:.4f}, Test Loss: {:.4f}'.format(
        #         epoch + 1, num_epochs, train_loss, test_loss))
    return best_metrics

def loader_to_vfeature(dataloader, model):  ## data_tensor: tensor with size=(N, ebd_dim), model: ContrastiveModel
    model.eval()
    embeddings = []
    labels = []
    with torch.no_grad():
        for batch_data, batch_y, vfreq in dataloader:  ## ([batch size, v num, bag size, 120])
            batch_data = batch_data.to(model.fc1[0].weight.device)
            batch_out = model(batch_data)  ##  ([batch size, v num, bag size, 120])
            embeddings.append(batch_out)
            labels.append(batch_y)
    embeddings = torch.cat(embeddings, dim=0).cpu().numpy()
    labels = torch.cat(labels, dim=0).cpu().numpy()
    return embeddings, labels, vfreq


def tensor_to_vfeature(batch_data_x, model):
    model.eval()
    with torch.no_grad():
        batch_data_x = batch_data_x.to(model.fc1[0].weight.device)
        batch_out = model(batch_data_x)  ##  ([batch size, v num, bag size, 120])
    return batch_out




if __name__ == '__main__':

    ## test embedding classification
    embeddings = [torch.randn(120) + 0.5 for i in range(50)] + [torch.randn(120) for i in range(50)]
    labels = [1] * 50 + [0] * 50
    print("Classifier training")
    # evaluate_ebd_classification(embeddings, labels)
