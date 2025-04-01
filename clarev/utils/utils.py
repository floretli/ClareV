import pickle
import os
import glob
from collections import defaultdict

def save_pk(file_savepath, data):
    with open(file_savepath, "wb") as fp:
        pickle.dump(data, fp)

def load_pk(filename):
    with open(filename, "rb") as fp:
        data_dict = pickle.load(fp)
    return data_dict

def read_filelist(filepath_txt):

    if isinstance(filepath_txt, list):
        total_filelist = []
        for f in filepath_txt:
            filelist = open(f, "r").read().split("\n")
            filelist.remove("")
            total_filelist += filelist
    else:
        total_filelist = open(filepath_txt, "r").read().split("\n")
        total_filelist.remove("")

    for filepath in total_filelist:
        if not (os.path.isfile(filepath)):
            total_filelist.remove(filepath)
            print(f"Warning: file {filepath} does not exist. Removing from list.")
    return total_filelist

def class_sampling(group, n, seed=123):
    return group.sample(n=n, random_state=seed)

def class_balance(df, label_col="label"):
    class_size = df[label_col].value_counts().to_list()
    min_size = min(class_size)
    df = df.groupby(label_col).apply(class_sampling, min_size)
    return df

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def merge_listpk(directory):
    files = glob.glob(os.path.join(directory, '*.pk'))
    if len(files) <= 1:
        return

    prefix_dict = defaultdict(list)
    for file in files:
        prefix = os.path.basename(file).split('_part')[0]
        prefix_dict[prefix].append(file)

    for prefix, files in prefix_dict.items():
        if len(files) <= 1:
            continue

        print(f"Merging files for prefix: {prefix}")
        merged_list = []
        for file in files:
            with open(file, "rb") as fp:
                data = pickle.load(fp)
                print(f"Loaded data from file: {file}, sample size: {len(data)}")
                merged_list.extend(data)
        print("Merged sample size: ", len(merged_list))
        print(f"Saving merged list for prefix: {prefix}")
        save_pk(os.path.join(directory, f"{prefix}.pk"), merged_list)
        
        # ## remove file
        # for file in files:
        #     os.remove(file)
    # return merged_list

if __name__ == "__main__":
    read_filelist("/home/grads/miaozhhuo2/projects/TCRseq_data/datalist/MDA_T.list")
