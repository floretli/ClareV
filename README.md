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

### External Dataset Processing
The external dataset should be processed following the TITAN project's TCR sequence concatenation method. This involves concatenating the CDR3 sequence with its corresponding VJ gene sequences. For more details, please refer to the TCR sequence concatenation [script](https://github.com/PaccMann/TITAN/blob/main/scripts/cdr3_to_full_seq.py) from [TITAN's project](https://github.com/PaccMann/TITAN).

After processing, the data should be saved in a TSV or CSV format. This file will be used as the input for the script `1.divide_tcr_bags_accroding_imgt.py` to convert the data into the TCR groupings required by the ClareV framework.
Within the `filelist_to_superbags` function, ensure that the following parameters match the input data format and column names:
- `file_type="tsv"`  Specifies the format of the input file.  full_seq for TCR2vec's full-length mode, aminoAcid for TCR2vec's CDR3 mode.
- `tcr_col="full_seq"` Refers to the column holding the TCR sequence data.
- `v_col="vMaxResolved"` Indicates the column containing the V gene information. vFamilyName for V family level experiment, vMaxResolved for V gene level experiment.


```
aminoAcid	frequencyCount (%)	count (templates/reads)	nucleotide	vFamilyName	vMaxResolved	dFamilyName	dGeneName	jFamilyName	jMaxResolved	cdr3Length	full_seq
CASSSSQGRDSPLHF	0.0248436674828364	16559	CTGTCGGCTGCTCCCTCCCAGACATCTGTGTACTTCTGTGCCAGCAGTTCCTCACAGGGCCGGGATTCACCCCTCCACTTTGGGAAC	TCRBV06	TCRBV06-05*01	TCRBD01	TCRBD01-01	TCRBJ01	TCRBJ01-06*01	45	NAGVTQTPKFQVLKTGQSMTLQCAQDMNHEYMSWYRQDPGMGLRLIHYSVGAGITDQGEVPNGYNVSRSTTEDFPLRLLSAAPSQTSVYFCASSSSQGRDSPLHF
CASSQRGQGYYGYTF	0.0070694704498535	4712	AACGCCTTGGAGCTGGACGACTCGGCCCTGTATCTCTGTGCCAGCAGCCAGCGCGGACAGGGCTACTATGGCTACACCTTCGGTTCG	TCRBV05	TCRBV05-04*01	TCRBD01	TCRBD01-01	TCRBJ01	TCRBJ01-02*01	45	ETGVTQSPTHLIKTRGQQVTLRCSSQSGHNTVSWYQQALGQGPQFIFQYYREEENGRGNFPPRFSGLQFPNYSSELNVNALELDDSALYLCASSQRGQGYYGYTF
CASSRTGDSYEQYF	0.0057611983292524	3840	...
```
- Each sample file must include a "frequencyCount (%)" or a "count (templates/reads)	nucleotide" column to ensure correct parsing and processing during TCR bag construction and subsequent analyses.
- The full_seq column should contain the concatenated sequence of the original CDR3 and its corresponding VJ gene sequence.
- Our ClareV framework utilizes the [TCR2vec tool](https://github.com/jiangdada1221/TCR2vec) to encode TCR sequences based on the full_seq column.

## TCR Encoder
ClareV leverages TCR clonotype-level embedding through deep learning to learn V gene-level representations. The built-in method for TCR embedding is the TCR2vec tool, which uses the full-length encoding mode by default. This model is based on a transformer-based pre-trained model of TCR sequences.

During the first run, ClareV will automatically download the pre-trained model from [Google Drive](https://drive.google.com/file/d/1SML_YjiK6WwIgXD-4jIRcy1vWE-46PUp/view?usp=sharing) and save it to the `./pretrained_models` directory. If the download fails or if you prefer to use a different encoding mode (e.g., CDR3 encoding), please manually download the appropriate model from the [TCR2vec project]((https://github.com/jiangdada1221/TCR2vec)) and update the `path_to_TCR2vec` variable in the script 1.divide_tcr_bags_accroding_imgt.py to point to the correct directory.

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

