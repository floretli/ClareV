import torch.nn as nn
import torch
from torch.utils.data import TensorDataset, DataLoader

class Embedding_classifier(nn.Module):
    def __init__(self, input_dim = 96, hidden_dim =128, output_dim = 2, dropout_prob = 0.2):
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

def evaluate_ebd_classification(embeddings, labels, input_dim = 96, num_epochs = 10):
    ## use embedding to classfy
    classifier = Embedding_classifier(input_dim, 128, 2, 0.5)
    optimizer = torch.optim.Adam(classifier.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    train_c_dataloader, test_c_dataloader = create_classfication_data(embeddings, labels)

    for epoch in range(num_epochs):
        train_loss = 0.0
        train_correct = 0
        for batch_ebd, batch_labels in train_c_dataloader:
            outputs = classifier(batch_ebd)

            loss = criterion(outputs, batch_labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(outputs.data, 1)
            correct = (predicted == batch_labels).sum().item()
            train_correct += correct
            train_loss += loss.item() * len(batch_labels)

        ## length of train_c_dataloader.dataset
        train_loss /= len(train_c_dataloader.dataset)
        train_accuracy = train_correct / len(train_c_dataloader.dataset)

        with torch.no_grad():
            test_loss = 0.0
            test_correct = 0
            for batch_ebd, batch_labels in test_c_dataloader:
                outputs = classifier(batch_ebd)

                loss = criterion(outputs, batch_labels)
                _, predicted = torch.max(outputs.data, 1)
                correct = (predicted == batch_labels).sum().item()

                test_loss += loss.item() * len(batch_labels)
                test_correct += correct

            test_loss /= len(test_c_dataloader.dataset)
            test_accuracy = test_correct / len(test_c_dataloader.dataset)

        print(
            'Epoch [{}/{}], Train Loss: {:.4f}, Train Accuracy: {:.4f}, Test Loss: {:.4f}, Test Accuracy: {:.4f}'.format(
                epoch + 1, num_epochs, train_loss, train_accuracy, test_loss, test_accuracy))

if __name__ == '__main__':

    ## test embedding classification
    embeddings = [torch.randn(96) + 0.5 for i in range(50)] + [torch.randn(96) for i in range(50)]
    labels = [1] * 50 + [0] * 50
    print("Classifier training")
    evaluate_ebd_classification(embeddings, labels)
