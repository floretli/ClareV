from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
import torch
import torch.nn as nn
import torch.optim as optim
import os
# from loguru import logger
import random
import numpy as np

def create_classifier(args, model_cls, loaders, edge_index, feature_dim, num_tasks):

    device = torch.device("cuda") if torch.cuda.is_available() and args.devices else torch.device("cpu")
    train_size = len(loaders[0][0].dataset)

    if args.resume is not None:
        args.save_path = args.resume
    else:
        os.makedirs(args.save_path, exist_ok=True)

    ## show args
    # logger.info(args)

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if device == torch.cuda.is_available():
            torch.cuda.manual_seed(args.seed)

    if model_cls.need_deg():
        deg = torch.bincount(edge_index[1], minlength=args.num_nodes)
        args.deg = deg

    node_encoder_cls = lambda: nn.Linear(feature_dim, args.gnn_emb_dim)
    def edge_encoder_cls(_):
        def zero(_):
            return 0
        return zero

    model = model_cls(
        num_tasks=num_tasks,
        args=args,
        node_encoder=node_encoder_cls(),
        edge_encoder_cls=edge_encoder_cls
    ).to(device)

    optimizer = optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    if args.scheduler == "plateau":
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=20,
            min_lr=0.0001,
            verbose=False
        )

    elif args.scheduler == "cosine":
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=args.epochs * train_size // args.batch_size,
        )
    elif args.scheduler is None:
        scheduler = None
    else:
        raise NotImplementedError

    return model, optimizer, scheduler, device, args