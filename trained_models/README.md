# Trained Models

## `{Dataset}_weight/best_model.pth`

Trained on a larger proportion of the repertoires in each dataset.
Used for downstream analyses in the paper (UMAP visualization, clustering experiments, etc.).

## `{Dataset}_weight/fold_wise/fold_*/best_model.pth`

Each model corresponds strictly to one fold defined in `data/splits/`.
The held-out test split for that fold was never seen during training.
Used in the nested k-fold cross-validation experiments to evaluate the repertoire classification performance of the learned V embeddings.
