import sys
import os
import argparse
import math
import numpy as np
import random

current_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.dirname(current_dir)
sys.path.append(repo_root)

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger, CSVLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor

from puget.decoder_data import get_hic_dataloader
from puget.decoder_model import Transformer2DRegressor
from puget.utils import load_config

def set_reproducibility(seed, ignore_if_cuda=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    if not (ignore_if_cuda and torch.cuda.is_available()):
        try:
            torch.use_deterministic_algorithms(True)
        except RuntimeError as e:
            print(f"Warning: {e}. Some operations might not be made deterministic.")

def train_model(cfg):
    set_reproducibility(cfg.seed)
    
    os.makedirs(cfg.logger_dir, exist_ok=True)
    os.makedirs(cfg.model_dir, exist_ok=True)

    logger_save_dir = os.path.join(cfg.logger_dir, cfg.logger_name)
    model_save_dir = os.path.join(cfg.model_dir, cfg.logger_name)
    
    os.makedirs(logger_save_dir, exist_ok=True)
    os.makedirs(model_save_dir, exist_ok=True)

    use_wandb = getattr(cfg, "use_wandb", False)
    wandb_project = getattr(cfg, "wandb_project", "Puget")

    if use_wandb:
        print(f"Logging to WandB (Project: {wandb_project})")
        logger = WandbLogger(
            project=wandb_project,
            save_dir=logger_save_dir,
            name=cfg.logger_name,
            config=cfg
        )
    else:
        print("WandB disabled. Logging to local CSV files.")
        logger = CSVLogger(
            save_dir=cfg.logger_dir, 
            name=cfg.logger_name
        )

    es = EarlyStopping(monitor="val_loss", patience=cfg.patience, mode="min")
    checkpoint_callback = ModelCheckpoint(
        dirpath=model_save_dir,
        monitor="val_loss",
        save_top_k=1,
        mode="min"
    )
    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    train_loader = get_hic_dataloader(
        x_npy_path=cfg.train_x,
        y_npy_path=cfg.train_y,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        prefetch_factor=2 if cfg.num_workers > 0 else None,
        persistent_workers=(cfg.num_workers > 0),
        pin_memory=True,
        mmap_mode="r",
    )

    val_loader = get_hic_dataloader(
        x_npy_path=cfg.val_x,
        y_npy_path=cfg.val_y,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        prefetch_factor=2 if cfg.num_workers > 0 else None,
        persistent_workers=(cfg.num_workers > 0),
        pin_memory=True,
        mmap_mode="r"
    )

    x_hdr = np.load(cfg.train_x, mmap_mode="r")
    input_dim = x_hdr.shape[-1]
    n_rows = x_hdr.shape[-3]
    n_cols = x_hdr.shape[-2]
    del x_hdr
    
    output_dim = 1
    print("input_dim: ", input_dim, "output_dim: ", output_dim)
    
    steps_per_train_epoch = math.ceil(len(train_loader.dataset) / cfg.batch_size)
    
    model = Transformer2DRegressor(
        input_dim=input_dim,
        output_dim=output_dim,
        steps_per_train_epoch=steps_per_train_epoch,
        n_rows=n_rows,
        n_cols=n_cols,
        proj_dim=cfg.proj_dim,
        mlp_ratio=cfg.mlp_ratio,
        num_heads=cfg.num_heads,
        num_layers=cfg.num_layers,
        pool_method=cfg.pool_method,
        dropout_p=cfg.dropout_p,
        lr=float(cfg.lr),
        warmup_epochs=cfg.warmup_epochs,
        decay_epochs=cfg.decay_epochs,
        min_lr=float(cfg.min_lr),
        weight_decay=float(cfg.weight_decay)
    )

    trainer = pl.Trainer(
        accelerator="auto",
        devices="auto",
        callbacks=[es, checkpoint_callback, lr_monitor],
        max_epochs=cfg.max_epochs,
        benchmark=False,
        profiler='simple',
        precision=cfg.precision,
        logger=logger,
        val_check_interval=cfg.val_check_interval,
        log_every_n_steps=1,
        gradient_clip_val=1.0,
        gradient_clip_algorithm="norm"
    )

    trainer.fit(model, train_loader, val_loader)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", 
        type=str, 
        default=os.path.join(repo_root, "configs/training/hicfoundation_train.yaml"),
        help="Path to YAML config file"
    )
    args = parser.parse_args()

    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Config file not found at {args.config}")
    
    cfg = load_config(args.config)
    train_model(cfg)

if __name__ == "__main__":
    main()
