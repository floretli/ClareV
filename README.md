# ClareV
A Contrastive Learning Framework for Context-Aware V Gene Representations in TCR Repertoires.

## Project Introduction
This repository contains the package and experimental code associated with the publication. 

## Installation
Follow these steps to get started:

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/floretli/ClareV.git
   cd ClareV
   ```

2. **Create the Conda Environment:**
   ```bash
   conda create -n clarev_env python=3.8
   conda activate clarev_env
   ```

3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Dataset
The filtered repertoire data and the data used for training the TCR bag feature extractor can be downloaded from [Zenodo](https://zenodo.org/records/15173766). After downloading, unzip the files and place them into the `./data/` directory.

## Script Descriptions
- **1.divide_tcr_bags_accroding_imgt.py**  
  This script processes the repertoire data by dividing TCR bags according to the IMGT classification definitions.

- **2.train_bag_feature_extractor.py**  
  This script trains the TCR bag feature extractor using the processed training data.

- **3.cmv_classification_method_comparison.py**  
  This script compares different CMV classification methods.

- **4.vgene_embedding_downstream_analysis.py**  
  This script performs downstream analyses on V gene embeddings, including the umap visualization of V gene embedding and constructing a phylogenetic tree.

## Additional Notes
For any issues or questions, please feel free to open an issue in this repository.
Email: miaozhhuo2-c@my.cityu.edu.hk

