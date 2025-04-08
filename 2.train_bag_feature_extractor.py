import torch
import os

from clarev.data_loaders.bag_data_loader import ClusterBagDataset
from clarev.trainers.CL_trainer import CL_Trainer
from clarev.models.contrastive_model import ContrastiveModel
from clarev.utils.utils import load_pk

from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split


def train_cl():

    ## =============== parameters =================

    ## input data dirs
    bag_data_dir = './data/processed_data/Emerson2017_vgene/'

    ## parameters for training and testing cl models
    trained_model_dir = './trained_models/Emerson2017_vgene_weight/'
    data_random_seed = 42
    num_epochs = 30
    batch_size = 32
    test_batch_size = 1
    vgene_emb_dim = 120
    tcr_emb_dim = 120
    subbag_size = 50
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    

    os.makedirs(trained_model_dir, exist_ok=True)

    ## =========  load existing bag data  ===========
    superbags = load_pk(os.path.join(bag_data_dir, "superbags.pk"))
    v_gene_order = load_pk(os.path.join(bag_data_dir, "v_gene_order.pk"))
    smp_list = load_pk(os.path.join(bag_data_dir, "smp_list.pk"))
    smp_labels = load_pk(os.path.join(bag_data_dir, "smp_labels.pk"))
    weights = load_pk(os.path.join(bag_data_dir, "weights.pk"))
    
    print(f"loaded {len(superbags)} samples with {len(v_gene_order)} v genes from the input data directory")
    

    ## =============== split data =================
    train_bags, test_bags, train_weights, test_weights = train_test_split(superbags, weights, test_size=0.1, random_state=data_random_seed)
    train_bags, eval_bags, train_weights, eval_weights = train_test_split(train_bags, train_weights, test_size=0.11, random_state=data_random_seed)

    print(f"train dataset size: {len(train_bags)}")
    print(f"eval dataset size: {len(eval_bags)}")
    print(f"test dataset size: {len(test_bags)}")
    print(f"train weights size: {len(train_weights)}")
    print(f"eval weights size: {len(eval_weights)}")
    print(f"test weights size: {len(test_weights)}")


    # Define dataloaders for train and test sets
    train_dataset = ClusterBagDataset(train_bags, weight = train_weights, subbag_size = subbag_size, data_repeat = 3)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    eval_dataset = ClusterBagDataset(eval_bags, weight = eval_weights, subbag_size = subbag_size)
    eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)

    test_dataset = ClusterBagDataset(test_bags, weight = test_weights, subbag_size = subbag_size)
    test_dataloader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)


    ## ============= train clareV CL model ==============
    print("Training CL model...")
    model = ContrastiveModel(input_dim=tcr_emb_dim, output_dim=vgene_emb_dim )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    trainer = CL_Trainer(model, optimizer, device=device, save_dir=trained_model_dir)

    final_model, train_metrics = trainer.train(train_dataloader, eval_dataloader, epochs=num_epochs)
    trainer.record_metrics(train_metrics) ## record training metrics as plot

    best_model = ContrastiveModel(input_dim=tcr_emb_dim, output_dim=vgene_emb_dim).to(device)
    best_model.load_state_dict(torch.load(os.path.join(trained_model_dir, 'best_model.pth'), map_location=device))
    test_metrics = trainer.evaluate(best_model, test_dataloader)
    test_accuracy = test_metrics['acc']
    test_loss = test_metrics['loss']
    print("evaluate test dataset")
    print('Test Accuracy: ', test_accuracy)
    print('Test Loss: ', test_loss)
    print('')


    ## save test metrics
    with open(os.path.join(trained_model_dir, 'test_metrics.txt'), 'w') as f:
        f.write(f'Test Accuracy: {test_accuracy}\n')
        f.write(f'Test Loss: {test_loss}\n')
        f.close()


if __name__ == "__main__":
    train_cl()