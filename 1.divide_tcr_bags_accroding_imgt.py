import torch
import os
import numpy as np

from clarev.encoders.tcr2vec_encoder import load_tcr2vec
from clarev.data_process import read_filelist_from_dir, filelist_to_vgenelist, filelist_to_superbags
from clarev.utils.utils import save_pk, load_pk, merge_listpk


def generate_bags():

    ## =============== parameters =================

    ## load TCR2vec model
    path_to_TCR2vec = './pretrained_models/TCR2vec_120'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    emb_model = load_tcr2vec(path_to_TCR2vec, device)

    ## data dirs
    repertoire_data_dir_pos = './data/Emerson2017/CMVpos'
    repertoire_data_dir_neg = './data/Emerson2017/CMVneg'
    bag_data_dir = './data/processed_data/Emerson2017_vgene/'
    process_by_batch = True
    process_batch_size = 100
    tcr_emb_dim = 120
    save_bag_data = True ## True or False
    
    ## =============== process data =================
    if save_bag_data:
        os.makedirs(bag_data_dir, exist_ok=True)
    
    ## test from filelist to superbags
    pos_file_list, pos_smp_list = read_filelist_from_dir(repertoire_data_dir_pos, format = ".tsv", suffix=".tsv")
    neg_file_list, neg_smp_list = read_filelist_from_dir(repertoire_data_dir_neg, format = ".tsv", suffix=".tsv")
    
    smp_labels = np.array([0] * len(pos_smp_list) + [1] * len(neg_smp_list))
    smp_list = neg_smp_list + pos_smp_list 
    file_list = neg_file_list + pos_file_list

    ## test cut 
    smp_list = smp_list
    file_list = file_list

    num_smp = len(smp_list)

    if process_by_batch & (num_smp > process_batch_size):
        print(f" {len(smp_list)} samples found in the input dataset, processing by batch...")
        v_gene_order = filelist_to_vgenelist(file_list, file_type="tsv",v_col="vMaxResolved")
        superbags = []
        weights = []
        for i in range(0, num_smp, process_batch_size):
            batch_files = file_list[i:i+process_batch_size]
            batch_smps = smp_list[i:i+process_batch_size]

            print(f"processing samples {i+1} to {i+process_batch_size}...")
            batch_superbags, batch_weights = filelist_to_superbags(batch_files, v_gene_order, emb_model, encoder_batch_size=2048, 
                                                    file_type="tsv", tcr_col="full_seq", v_col="vMaxResolved", tcr_emb_dim=tcr_emb_dim)
            save_pk(os.path.join(bag_data_dir, f"superbags_part{i//process_batch_size+1}.pk"), batch_superbags)
            save_pk(os.path.join(bag_data_dir, f"weights_part{i//process_batch_size+1}.pk"), batch_weights)
            superbags += batch_superbags
            weights += batch_weights
        merge_listpk(bag_data_dir)
    else:
        print(f" {len(smp_list)} samples found in the input dataset")
        print("Processing data to v gene package...")
        v_gene_order = filelist_to_vgenelist(file_list, file_type="tsv",v_col="vMaxResolved")
        superbags, weights = filelist_to_superbags(file_list, v_gene_order, emb_model, encoder_batch_size=2048, 
                                        file_type="tsv", tcr_col="full_seq", v_col="vMaxResolved", tcr_emb_dim=tcr_emb_dim)
        

    print("superbags shape (num of repertoire, num of v gene categories, bag mtx size in the first v gene) = ", 
          len(superbags), len(superbags[0]), superbags[0][0].shape)
    print("v gene order = ", v_gene_order)

    if save_bag_data:
        save_pk(os.path.join(bag_data_dir, "superbags.pk"), superbags)
        save_pk(os.path.join(bag_data_dir, "v_gene_order.pk"), v_gene_order)
        save_pk(os.path.join(bag_data_dir, "smp_list.pk"), smp_list)
        save_pk(os.path.join(bag_data_dir, "smp_labels.pk"), smp_labels)
        save_pk(os.path.join(bag_data_dir, "weights.pk"), weights)

if __name__ == "__main__":
    generate_bags()