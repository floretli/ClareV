import os
import pandas as pd
import torch
from typing import List, Tuple
from .encoders.tcr2vec_encoder import seqlist2ebd, load_tcr2vec
from collections import defaultdict
import numpy as np

def read_filelist(filepath_txt, suffix="_filt_full.tsv"):

    if isinstance(filepath_txt, list):  ## read files from multiple txt file
        total_filelist = []
        for f in filepath_txt:
            filelist = open(f, "r").read().split("\n")
            filelist.remove("")
            total_filelist += filelist
    else:    ## read files from single txt file
        total_filelist = open(filepath_txt, "r").read().split("\n")
        total_filelist.remove("")

    for filepath in total_filelist:
        if not (os.path.isfile(filepath)):
            total_filelist.remove(filepath)
            print(f"Warning: file {filepath} does not exist. Removing from list.")

    ## get basename of each file
    smp_list = [os.path.basename(f).replace(suffix, '') for f in total_filelist]
    return total_filelist, smp_list

def read_filelist_from_dir(directory, format = ".tsv", suffix="_filt_full.tsv"):
    # Initialize empty lists for the file paths and case ids
    file_list = []
    sample_list = []

    # Iterate over all files in the given directory
    for filename in os.listdir(directory):
        # Check if the file has the given suffix
        if filename.endswith(format):
            # Add the full file path to the file list
            file_list.append(os.path.join(directory, filename))
            # Remove the suffix to get the case id, and add it to the case list
            sample_id = filename[:-len(suffix)]
            sample_list.append(sample_id)

    return file_list, sample_list

def filt_v_gene_list(v_gene_order, v_gene_counts, total_files, min_files_threshold):
    filtered = []
    for v in v_gene_order:
        if v == "-":
            continue
        count = v_gene_counts.get(v, 0)

        if isinstance(min_files_threshold, float):
            threshold = int(min_files_threshold * total_files)
        else:
            threshold = min_files_threshold
        if count >= threshold:
            filtered.append(v)
    return filtered

def filelist_to_vgenelist(file_list: List[str],
                     file_type: str = "tsv",
                     v_col: str = "vMaxResolved",
                     ) -> Tuple[List[List[torch.Tensor]], List[str]]:
    """
    Convert CSV/TSV files containing TCR data into superbags format for ClusterBagDataset.
    
    Args:
        file_list: List of paths to CSV files or TSV files
        tcr_col: Column name containing TCR sequences
        v_col: Column name containing V gene information

    Returns:
        Tuple containing:
        - v_gene_order: Fixed-order list of all unique V genes across files
    """

    if file_type == "tsv":
        sep = "\t"
    elif file_type == "csv":
        sep = ","
    # First pass: Collect all unique V genes across all files
    v_gene_appears = defaultdict(int)
    all_v_genes = set()
    total_files = len(file_list)
    for file in file_list:
        df = pd.read_csv(file, sep=sep)
        all_v_genes.update(df[v_col].unique())
        for v in df[v_col].unique():
            v_gene_appears[v] += 1
    

    # Create ordered V gene list and lookup index
    v_gene_order = sorted(all_v_genes)
    print(f"Found {len(v_gene_order)} unique V genes across all files")

    ## filter out the v genes that appear in less than 50% files
    v_gene_order = filt_v_gene_list(v_gene_order, v_gene_appears, total_files, min_files_threshold = 1.0 )
    print(f"Filtered to {len(v_gene_order)} unique V genes")
    return v_gene_order

def filelist_to_superbags(file_list: List[str],
                     v_gene_order: List[str],
                     emb_model: torch.nn.Module,
                     file_type: str = "tsv",
                     tcr_col: str = "full_seq",
                     v_col: str = "vMaxResolved",
                     freq_col: str = "count (templates/reads)",
                     encoder_batch_size: int = 2048,
                     tcr_emb_dim: int = 120,
                     ) -> Tuple[List[List[torch.Tensor]], List[str]]:
    """
    Convert CSV/TSV files containing TCR data into superbags format for ClusterBagDataset.
    
    Args:
        file_list: List of paths to CSV files or TSV files
        tcr_col: Column name containing TCR sequences
        v_col: Column name containing V gene information
        emb_model: TCR2vec model for encoding TCR sequences
        encoder_batch_size: Batch size for encoding TCR sequences
        tcr_emb_dim: Dimension of TCR embeddings

    Returns:
        Tuple containing:
        - superbags: List of lists of tensors (one list per file)
        - v_gene_order: Fixed-order list of all unique V genes across files
    """

    if file_type == "tsv":
        sep = "\t"
    elif file_type == "csv":
        sep = ","

    # Second pass: Process files with batch encoding
    superbags = []
    weights = []
    for file in file_list:
        df = pd.read_csv(file, sep=sep)
        
        # Batch process all TCRs in the file
        all_tcrs = df[tcr_col].tolist()
        all_freqs = df[freq_col].tolist()
        embeddings = seqlist2ebd(all_tcrs, emb_model, batch_size=encoder_batch_size, keep_pbar=False)
        # ## mimic the data
        # embeddings = np.zeros((len(all_tcrs), tcr_emb_dim))
        
        
        # Create index mapping for V genes
        df[v_col] = df[v_col].astype('category')
        groups = df.groupby(v_col, observed=True).groups
        v_indices = {
            v: groups[v].to_numpy()
            for v in groups
        }
        # Create tensors for each V gene in fixed order
        file_bags = []
        file_weights = []
        for v_gene in v_gene_order:
            if v_gene in v_indices:
                indices = v_indices[v_gene]
                tensor = embeddings[indices]
                freqs = np.array(all_freqs)[indices]
            else:
                tensor = torch.empty((0, tcr_emb_dim))
                freqs = np.array([])
            file_bags.append(tensor)
            file_weights.append(freqs)
        
        superbags.append(file_bags)
        weights.append(file_weights)
    
    return superbags, weights

def bagdata_to_vfreq(superbags, v_gene_order, smp_list, norm=True):
    v_freq = np.zeros((len(smp_list), len(v_gene_order)))
    for i, smp in enumerate(superbags):
        for v_idx in range(len(v_gene_order)):
            v_freq[i][v_idx] = len(smp[v_idx])

    if norm:
        v_freq = v_freq / v_freq.sum(axis=1, keepdims=True)
    # v_freq_df = pd.DataFrame(v_freq, columns=v_gene_order, index=smp_list)
    return v_freq


if __name__ == "__main__":
    ## load the trained TCR2vec model
    path_to_TCR2vec = '../pretrained_models/TCR2vec_120'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    emb_model = load_tcr2vec(path_to_TCR2vec, device)

    # convert list of seqs to numpy array
    seq_list = ['NAGVTQTPKFQVLKTGQSMTLQCAQDMNHNSMYWYRQDPYSASEGTTDKGEVPNGYNVSRLNKREFSLRLESAAPSQTSVYFCASSEALGTGNTIYFGEGSWLTVV',
                'NAGVTQTPKFQVLKTGQSMTLQCAQDMNHNSMYWYRQDPGMGLLLIYYSASEGTTDKGEVPNGYNVSRLNKREFSLRLESAAPSQTSVYFCASSEALGTGNTIYFGEGSWLTVV',
                'NAGVTQTPKFQVLKTGQSMTLQCAQDMNHNSMYWYRQDPGMGLRLIYYSRLNKREFSLRLESAAPSQTSVYFCASSEALGTGNTIYFGEGSWLTVV']
    embmtx = seqlist2ebd(seq_list, emb_model)
    print("example seq list = ", seq_list)
    print("embedding mtx shape = ", embmtx.shape)

    ## test from filelist to superbags
    demo_data_dir = '../data/example_dataset/'
    example_list, smp_list = read_filelist_from_dir(demo_data_dir, format = ".tsv", suffix=".tsv")

    superbags, v_gene_order = filelist_to_superbags(example_list, emb_model, file_type="tsv", tcr_emb_dim=120)
    print("example file list = ", example_list)
    print("example sample list = ", smp_list)
    print("superbags shape (num of repertoire, num of v gene categories, bag mtx size in the first v gene) = ", 
          len(superbags), len(superbags[0]), superbags[0][0].shape)
    print("v gene order = ", v_gene_order)

