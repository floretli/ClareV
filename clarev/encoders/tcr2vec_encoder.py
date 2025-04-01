import torch
from .tcr2vec.model import TCR2vec
from .tcr2vec.dataset import TCRLabeledDset
from .tcr2vec.utils import get_emb
from torch.utils.data import DataLoader

import numpy as np
import os

def check_model_exist(path_to_TCR2vec = '../../pretrained_models/TCR2vec_120'):

    model_file_path = os.path.join(path_to_TCR2vec, 'pytorch_model.bin')
    args_file_path = os.path.join(path_to_TCR2vec, 'args.json')
    config_file_path = os.path.join(path_to_TCR2vec, 'config.json')

    # Check if model and json files exist
    if not os.path.exists(model_file_path) or not os.path.exists(args_file_path) or not os.path.exists(config_file_path):
        import gdown
        import zipfile

        pretrained_path = os.path.dirname(path_to_TCR2vec)
        download_url = 'https://drive.google.com/uc?export=download&id=1Nj0VHpJFTUDx4X7IPQ0OGXKlGVCrwRZl'
        zip_path = os.path.join(pretrained_path, 'tcr2vec_120.zip')

        os.makedirs(pretrained_path, exist_ok=True)

        print(f"Downloading model from {download_url} to {zip_path}")
        gdown.download(download_url, zip_path, quiet=False)

        print(f"Extracting {zip_path} to {pretrained_path}")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(pretrained_path)
        os.remove(zip_path)
        print(f"TCR encoder model TCR2vec extracted successfully.")
    else:
        print(f"loading TCR2vec encoder...")


## load the trained TCR2vec model
def load_tcr2vec(path_to_TCR2vec = '../../pretrained_models/TCR2vec_120', device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
    check_model_exist(path_to_TCR2vec)
    emb_model = TCR2vec(path_to_TCR2vec)
    emb_model = emb_model.to(device)
    return emb_model

def seqlist2ebd(seq_list, emb_model, emb_size = 120, batch_size=2048, keep_pbar = True):  ## input: a list of TCR seqs ['CAAAGGIYEQYF', 'CAAAPGINEQFF' ... ], output: the mtx of 96-dim embedding

    if len(seq_list) == 0 :
        return np.zeros((1, emb_size),dtype='float32')

    dset = TCRLabeledDset(seq_list, only_tcr=True) #input a list of TCRs
    loader = DataLoader(dset, batch_size=batch_size, collate_fn=dset.collate_fn, shuffle=False)
    emb = get_emb(emb_model, loader, detach=True, keep_pbar = keep_pbar) #B x emb_size

    return emb


if __name__ == "__main__":
    ## load the trained TCR2vec model
    path_to_TCR2vec = '../../pretrained_models/TCR2vec_120'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    emb_model = load_tcr2vec(path_to_TCR2vec, device)

    # convert list of seqs to numpy array
    seq_list = ['NAGVTQTPKFQVLKTGQSMTLQCAQDMNHNSMYWYRQDPYSASEGTTDKGEVPNGYNVSRLNKREFSLRLESAAPSQTSVYFCASSEALGTGNTIYFGEGSWLTVV',
                'NAGVTQTPKFQVLKTGQSMTLQCAQDMNHNSMYWYRQDPGMGLLLIYYSASEGTTDKGEVPNGYNVSRLNKREFSLRLESAAPSQTSVYFCASSEALGTGNTIYFGEGSWLTVV',
                'NAGVTQTPKFQVLKTGQSMTLQCAQDMNHNSMYWYRQDPGMGLRLIYYSRLNKREFSLRLESAAPSQTSVYFCASSEALGTGNTIYFGEGSWLTVV']
    embmtx = seqlist2ebd(seq_list, emb_model)
    print("example seq list = ", seq_list)
    print("embedding mtx shape = ", embmtx.shape)
    print("embedding mtx = ", embmtx)
