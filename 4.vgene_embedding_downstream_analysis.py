import torch
import os
import numpy as np
import pickle

from clarev.data_loaders.bag_data_loader import RepertoireAggregateDataset, RepertoireDataset
from clarev.models.contrastive_model import ContrastiveModel
from clarev.utils.utils import save_pk, load_pk
from clarev.trainers.CL_evaluation import vfeature_classification, loader_to_vfeature
from clarev.data_process import bagdata_to_vfreq
from torch.utils.data import DataLoader
from umap import UMAP
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans


def agg_dataloader_to_vfeature(agg_dataloader):
    all_embeddings = []
    for batch_ebd, batch_labels, _ in agg_dataloader:
        all_embeddings.append(batch_ebd.detach().cpu().numpy())
    vembeddings = np.concatenate(all_embeddings, axis=0)
    return vembeddings

def map_vgene_to_vfam(vgene_order):
    vgene_to_vfam = {}
    for vgene in vgene_order:
        if vgene.startswith("TCRBV") or vgene.startswith("TCRAV"):
            vfam = vgene.split("-")[0]
        else:
            vfam = vgene
        vgene_to_vfam[vgene] = vfam
    return vgene_to_vfam

def main():
    ## =============== parameters =================

    ## input data dirs
    bag_data_dir = './data/processed_data/Emerson2017_vgene/'
    trained_model_dir = './trained_models/Emerson2017_vgene_weight/'
    num_epochs = 30
    batch_size = 16
    test_batch_size = 1
    vgene_emb_dim = 120
    tcr_emb_dim = 120
    subbag_size = 50
    cutoff_smp = False # False ##  10 ## 10
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    result_dir = './results/figs/vgene_analysis'
    os.makedirs(result_dir, exist_ok=True)
    os.makedirs(trained_model_dir, exist_ok=True)

    ## =========  load existing bag data  ===========
    superbags = load_pk(os.path.join(bag_data_dir, "superbags.pk"))
    v_gene_order = load_pk(os.path.join(bag_data_dir, "v_gene_order.pk"))
    smp_list = load_pk(os.path.join(bag_data_dir, "smp_list.pk"))
    smp_labels = load_pk(os.path.join(bag_data_dir, "smp_labels.pk"))
    weights = load_pk(os.path.join(bag_data_dir, "weights.pk"))
    
    print(f"loaded {len(superbags)} samples with {len(v_gene_order)} v genes from the input data directory")

    num_vgene = len(v_gene_order)
    v_freq = bagdata_to_vfreq(superbags, v_gene_order, smp_list, norm=True)  ## v_freq is a numpy array

    # =============== use clareV model =================
    print("loading the best model and bag data... ")
    best_model = ContrastiveModel(input_dim=tcr_emb_dim, output_dim=vgene_emb_dim).to(device)
    best_model.load_state_dict(torch.load(os.path.join(trained_model_dir, 'best_model.pth'), map_location=device))

    whole_dataset = RepertoireDataset(superbags, labels = smp_labels, v_freq_mtx=v_freq, weight= weights, subbag_size = subbag_size)
    whole_dataloader = DataLoader(whole_dataset, batch_size=batch_size, shuffle=False)
    best_model.to(device)

    print("processing v embeddings by clareV...")
    vembeddings, labels, _ = loader_to_vfeature(whole_dataloader, best_model)
    print(f"vembeddings shape: {vembeddings.shape}, labels shape: {labels.shape}") ## vembeddings shape: (760, 21, 120)

    # ## ================ use pooling ====================
    # agg_dataset = RepertoireAggregateDataset(superbags, labels = smp_labels, v_freq_mtx=v_freq, agg_type = "mean")
    # agg_dataloader = DataLoader(agg_dataset, batch_size=batch_size, shuffle=False)
    # print("processing v embeddings by max pooling...")
    # vembeddings = agg_dataloader_to_vfeature(agg_dataloader)
    # print(f"vembeddings shape: {vembeddings.shape}") ## vembeddings shape: (760, 21, 120)


    if cutoff_smp:
        np.random.seed(0)
        idx = np.random.permutation(len(smp_list))
        smp_list = np.array(smp_list)[idx]
        smp_labels = np.array(smp_labels)[idx]
        vembeddings = vembeddings[idx]

        smp_list = smp_list[:cutoff_smp]
        smp_labels = smp_labels[:cutoff_smp]
        vembeddings = vembeddings[:cutoff_smp]
        scatter_size = 5
    else:
        scatter_size = 2

    vembeddings = vembeddings.reshape(-1, 120)


    emb_v_labels = np.tile(v_gene_order, len(smp_list))
    emb_smp_labels = np.repeat(smp_list, len(v_gene_order))
    emb_cmv_labels = np.repeat(smp_labels, len(v_gene_order))


    ##  ======== UMAP visualization ==========
    print(f"emb_v_labels shape: {emb_v_labels.shape}")
    print(f"vembeddings shape: {vembeddings.shape}")
    umap = UMAP(n_components=2, n_neighbors=100, min_dist=0.2, metric= 'cosine')
    vembeddings_2d = umap.fit_transform(vembeddings)

    ## ===== label by V gene
    plt.figure(figsize=(12, 10))
    sns.scatterplot(x=vembeddings_2d[:, 0], y=vembeddings_2d[:, 1], hue=emb_v_labels, 
                    palette=sns.color_palette("hsv", len(v_gene_order)), s=scatter_size, alpha=0.8, legend="full")
    plt.legend(
        bbox_to_anchor=(1.05, 1),  
        loc='upper left',           
        borderaxespad=0.,           
        title="V Genes",
        fontsize=10,
        ncol=1                     
    )
    plt.tight_layout()
    # plt.title('UMAP projection of the V gene embeddings', fontsize=24)
    plt.savefig(os.path.join(result_dir, 'v_gene_embeddings_vtype.png'), dpi=600)


    ## ===== label by CMV status
    plt.figure(figsize=(10, 10))
    sns.scatterplot(x=vembeddings_2d[:, 0], y=vembeddings_2d[:, 1], hue=emb_cmv_labels,
                    s=scatter_size, alpha=0.8)
    # plt.title('UMAP projection of the V gene embeddings', fontsize=24)
    plt.legend(
        bbox_to_anchor=(1.05, 1),
        loc='upper left',        
        borderaxespad=0.,        
        title="CMV label",
        fontsize=10,
        ncol=1                   
    )
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, 'v_gene_embeddings_cmv.png'), dpi=600)



    ##  ========= the clarev learned V embeding with family==========
    vgene2vfam = map_vgene_to_vfam(v_gene_order)
    v_fam_order = [vgene2vfam[vgene] for vgene in v_gene_order]
    v_fam_order = sorted(list(set(v_fam_order))) ## uniq
    fam2vgene = {}
    for vgene, vfam in vgene2vfam.items():
        fam2vgene.setdefault(vfam, set()).add(vgene)
    multi_family = {fam: (len(vgenes) > 1) for fam, vgenes in fam2vgene.items()}

    colors = dict(zip(v_fam_order, sns.color_palette("hsv", len(v_fam_order))))

    emb_vfam_labels = np.array([vgene2vfam[vgene] for vgene in emb_v_labels])

    row_colors = [colors[vgene2vfam[vgene]] for vgene in v_gene_order]
    vembeddings = vembeddings.reshape(len(smp_list), num_vgene, vgene_emb_dim)
    vembeddings = np.mean(vembeddings, axis=0) ## mean at sample level
    # print(f"mean vembeddings shape: {vembeddings.shape}") ## (36, 120)

    plt.figure(figsize=(14, 10))
    ## label by V gene
    g = sns.clustermap(vembeddings, cmap='YlGn', metric='cosine', row_cluster=True, col_cluster=False, row_colors=row_colors)
    v_gene_order = np.array(v_gene_order)
    row_order = v_gene_order[g.dendrogram_row.reordered_ind]
    g.ax_heatmap.set_yticklabels(row_order)
    plt.setp(g.ax_heatmap.get_xticklabels(), rotation=90)
    plt.setp(g.ax_heatmap.get_yticklabels(), rotation=0)
    plt.savefig(os.path.join(result_dir, 'v_gene_embeddings_similarity.png'), bbox_inches='tight',dpi=600)



main()