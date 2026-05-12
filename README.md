# ClareV
A Contrastive Learning Framework for Context-Aware V Gene Representations in TCR Repertoires.

## Project Introduction
This repository contains the package and experimental code associated with the publication.

## Installation

Use the repository root (`ClareV/`) as your working directory.

1. **Clone and enter repository**
   ```bash
   git clone https://github.com/floretli/ClareV.git
   cd ClareV
   ```

2. **Create environment**
   ```bash
   conda create -n clarev_env python=3.8
   conda activate clarev_env
   ```

3. **Install PyTorch (choose command for your CUDA/CPU)**
   - Follow the official selector: [PyTorch installation guide](https://pytorch.org/get-started/locally/).

4. **Install remaining dependencies**
   ```bash
   pip install -r requirements-no-torch.txt
   pip install -e .
   ```

Notes:
- `gdown` is required (used to download pretrained encoder files).
- `tensorflow` is optional and only needed for legacy paths in `clarev/encoders/tcr2vec/`.
- `scikit-learn-intelex` is optional (performance acceleration only).

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
- Each sample file must include either `frequencyCount (%)` or `count (templates/reads)` for abundance parsing.
- The full_seq column should contain the concatenated sequence of the original CDR3 and its corresponding VJ gene sequence.
- Our ClareV framework utilizes the [TCR2vec tool](https://github.com/jiangdada1221/TCR2vec) to encode TCR sequences based on the full_seq column.

## TCR Encoder
ClareV leverages TCR clonotype-level embedding through deep learning to learn V gene-level representations. The built-in method for TCR embedding is the TCR2vec tool, which uses the full-length encoding mode by default. This model is based on a transformer-based pre-trained model of TCR sequences.

During the first run, ClareV will automatically download the pre-trained model from [Google Drive](https://drive.google.com/file/d/1SML_YjiK6WwIgXD-4jIRcy1vWE-46PUp/view?usp=sharing) and save it to the `./pretrained_models` directory. If the download fails or if you prefer to use a different encoding mode (e.g., CDR3 encoding), please manually download the appropriate model from the [TCR2vec project](https://github.com/jiangdada1221/TCR2vec) and update the `path_to_TCR2vec` variable in `1.divide_tcr_bags_accroding_imgt.py` to point to the correct directory.

## Repository Layout

After downloading the data and the pretrained extractors, the relevant
top-level directories follow this convention:

```
data/
├── processed_data/
│   ├── Emerson2017_vfam/        # bag-level inputs (V family resolution)
│   ├── Emerson2017_vgene/       # bag-level inputs (V gene resolution)
│   └── <your_dataset>_vfam/     # add your own datasets here
└── splits/
    ├── emerson2017_nested_kfold5.npz   # repertoire-level nested 5-fold split
    └── <your_dataset>_nested_kfold5.npz

trained_models/
├── Emerson2017_vfam_weight/
│   └── fold_wise/fold_{0..4}/best_model.pth
├── Emerson2017_vgene_weight/
│   └── fold_wise/fold_{0..4}/best_model.pth
└── <your_dataset>_vfam_weight/
    └── fold_wise/fold_{0..4}/best_model.pth
```

- **`data/splits/<dataset>_nested_kfold5.npz`** — a single `.npz` file
  storing the repertoire-level 5-fold nested split for that dataset
  (`train_splits`, `val_splits`, `test_splits` — one array of indices
  per outer fold). The split is derived purely from repertoire IDs.
- **`trained_models/<dataset>_<resolution>_weight/fold_wise/`** — five
  fold-specific contrastive bag-feature extractors, one per outer
  fold of the nested protocol. Each `fold_X/best_model.pth` was
  trained only on that fold's outer train+val pool, so the outer test
  fold is never seen during contrastive training.

## Script Descriptions

The numbered top-level scripts cover the full pipeline. Run them from
the repository root.

- **`1.divide_tcr_bags_accroding_imgt.py`** — process repertoire TSVs
  with TCR2vec encoding and group clones by V annotation, producing
  the `data/processed_data/<dataset>/` pickle set
  (`superbags.pk`, `weights.pk`, `smp_list.pk`, `smp_labels.pk`,
  `v_gene_order.pk`).
- **`2.0.train_wholedata_bag_feature_extractor.py`** — train **one** shared
  contrastive V-bag feature extractor on an 80/10/10 repertoire split
  (used for downstream analysis).
- **`2.1.train_foldwise_bag_feature_extractor.py`** — paper-canonical
  foldwise variant: trains five fold-matched extractors so each
  downstream outer fold has its own extractor that never saw the
  outer test repertoires.
- **`3.0.clarev_classification.py`** — repertoire-level
  classification with the ClareV embedding (and the RFFusion variant
  that combines the embedding with V-usage features).
- **`3.1.v-usage_baseline_classification.py`** — V-usage-only baselines
  (Random Forest, Logistic Regression, SVM linear/RBF, MLP, CNN).
- **`4.vgene_embedding_downstream_analysis.py`** — UMAP, V-family
  organization, and hierarchical clustering analyses on the learned
  V-gene representations.

## Reproducible Commands

### Step 1 — bag inputs (only needed for new repertoire data)

```bash
python 1.divide_tcr_bags_accroding_imgt.py
```

The Zenodo bundle already contains pre-built `data/processed_data/`
inputs for the cohorts used in the paper, so this step can be skipped
when reproducing those experiments.

### Step 2 — contrastive V-bag feature extractor

The paper protocol uses the foldwise variant. Pre-trained checkpoints
that match the paper configuration are shipped under
`trained_models/<dataset>_<resolution>_weight/fold_wise/`. Step 2.1 can
also be re-run from scratch.

V-family resolution example:
```bash
python 2.1.train_foldwise_bag_feature_extractor.py \
  --bag-data-dir data/processed_data/Emerson2017_vfam \
  --trained-model-dir trained_models/Emerson2017_vfam_weight/fold_wise \
  --split-file data/splits/emerson2017_nested_kfold5.npz
```

V-gene resolution example:
```bash
python 2.1.train_foldwise_bag_feature_extractor.py \
  --bag-data-dir data/processed_data/Emerson2017_vgene \
  --trained-model-dir trained_models/Emerson2017_vgene_weight/fold_wise \
  --split-file data/splits/emerson2017_nested_kfold5.npz
```

If the five `best_model.pth` checkpoints are already present, skip
this step and go directly to Step 3 — `3.0` reads the same directory.

### Step 3 — downstream classification

ClareV with RFFusion (default fusion backend) on Emerson V-family:
```bash
python 3.0.clarev_classification.py \
  --bag-data-dir data/processed_data/Emerson2017_vfam \
  --trained-model-dir trained_models/Emerson2017_vfam_weight/fold_wise \
  --split-file data/splits/emerson2017_nested_kfold5.npz \
  --extractor-source foldwise
```

The fusion backend defaults to RFFusion (`--vfreq-fusion-backend rf`).
Pass `--vfreq-fusion-backend mlp` to fall back to the legacy MLP fusion
variant.

V-usage-only baselines on the same split:
```bash
python 3.1.v-usage_baseline_classification.py \
  --bag-data-dir data/processed_data/Emerson2017_vfam \
  --split-file data/splits/emerson2017_nested_kfold5.npz
```

### Step 4 — embedding-space analyses

```bash
python 4.vgene_embedding_downstream_analysis.py
```

Outputs UMAP projections, V-family alignment metrics, and a
population-level hierarchical clustering of the learned V-gene
representations.

## Additional Notes
For any issues or questions, please feel free to open an issue in this repository.
Email: miaozhhuo2-c@my.cityu.edu.hk
