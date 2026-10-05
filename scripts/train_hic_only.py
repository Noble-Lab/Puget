from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from pathlib import Path

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
for variable in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMBA_NUM_THREADS",
):
    os.environ.setdefault(variable, "1")

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger, WandbLogger

from puget.hic_only_data import build_hicfoundation_loader, load_biosample_table


def set_seed(seed: int, deterministic: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = bool(deterministic)
    torch.backends.cudnn.benchmark = not bool(deterministic)
    torch.use_deterministic_algorithms(bool(deterministic), warn_only=True)


def load_config(path: str, *, include_test: bool = True) -> dict:
    with open(path) as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Config {path} is not a mapping")
    for group in ("data", "encoder"):
        for key, value in config[group].items():
            if not include_test and key.startswith("test_"):
                continue
            if key in {"hic_root", "biosamples_csv", "train_bedpe", "valid_bedpe",
                       "test_bedpe", "train_label", "valid_label", "test_label",
                       "checkpoint_path"} and isinstance(value, str) and not os.path.isabs(value):
                config[group][key] = str((ROOT / value).resolve())
    return config


def ensure_encoder_checkpoint(encoder_cfg: dict) -> Path:
    checkpoint = Path(encoder_cfg["checkpoint_path"])
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    return checkpoint


def make_model(encoder_cfg: dict, decoder_cfg: dict):
    from puget.hic_only_model import FrozenHiCFoundationRegressor

    return FrozenHiCFoundationRegressor(
        encoder_ckpt_path=str(encoder_cfg["checkpoint_path"]),
        model_name=str(encoder_cfg["model_name"]),
        input_size=int(encoder_cfg["input_size"]),
        patch_size=int(encoder_cfg["patch_size"]),
        embed_dim=int(encoder_cfg["embed_dim"]),
        grid_rows=int(encoder_cfg["grid_rows"]),
        grid_cols=int(encoder_cfg["grid_cols"]),
        output_dim=int(decoder_cfg.get("output_dim", 1)),
        proj_dim=int(decoder_cfg.get("proj_dim", 512)),
        mlp_ratio=float(decoder_cfg.get("mlp_ratio", 4.0)),
        num_heads=int(decoder_cfg.get("num_heads", 16)),
        num_layers=int(decoder_cfg.get("num_layers", 4)),
        pool_method=str(decoder_cfg.get("pool_method", "cls")),
        dropout=float(decoder_cfg.get("dropout", 0.1)),
        encoder_fp32=bool(encoder_cfg.get("force_fp32", False)),
        output_activation=str(decoder_cfg.get("output_activation", "softplus")),
    )


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--gpu", type=int, default=None,
                        help="physical GPU index for this single-GPU job (sets CUDA_VISIBLE_DEVICES)")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None,
                        help="override training.lr from the config")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="override output_dir from the config")
    parser.add_argument("--resume", default=None, help="Lightning checkpoint to resume from")
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument(
        "--output-activation", choices=("softplus", "linear"), default=None,
        help="override decoder.output_activation (default: softplus)",
    )
    parser.add_argument(
        "--no-wandb", action="store_true", help="use local CSV logging only"
    )
    parser.add_argument("--fast", action="store_true", help="Eight train and four validation batches")
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Validate config, split isolation, and required files without loading Hi-C PKLs",
    )
    return parser.parse_args(argv)


def count_bedpe_rows(path: str) -> int:
    with open(path) as handle:
        return sum(1 for line in handle if line.strip())


def validate_experiment(config: dict) -> tuple[int, int, str]:
    data = config["data"]
    encoder = config["encoder"]
    decoder = config["decoder"]
    training = config["training"]
    split_cfg = config["split"]
    if str(data.get("input_semantics", "")).lower() != "raw_none":
        raise ValueError("HiCFoundation requires raw/NONE contacts")
    if decoder.get("output_activation") != "softplus" or int(decoder["output_dim"]) != 1:
        raise ValueError("The raw Hi-C baseline requires one Softplus output per Hi-C window")
    if not bool(encoder.get("freeze", True)):
        raise ValueError("The HiCFoundation encoder must remain frozen")
    size = int(data["size"])
    if size not in (192, 512) or int(data["window_bp"]) != size * 1024:
        raise ValueError("Expected 192 or 512 exact 1,024 bp bins")
    patch = int(encoder["patch_size"])
    grid = size // patch
    if int(encoder["input_size"]) != size or size % patch or (int(encoder["grid_rows"]), int(encoder["grid_cols"])) != (grid, grid):
        raise ValueError("Decoder must receive the complete encoder patch grid")
    if str(training["optimizer"]).lower() != "adamw" or float(training["lr"]) != 5e-5:
        raise ValueError("The raw Hi-C baseline uses AdamW at 5e-5")
    if int(training["batch_size"]) != 128 or str(training["precision"]) != "16-mixed":
        raise ValueError("The raw Hi-C baseline uses batch 128 and 16-mixed precision")
    table = load_biosample_table(data["biosamples_csv"])
    train_names = [name for row, _, name in table if row < 14]
    if len(train_names) != 14 or {"GM12878", "K562"} & set(train_names):
        raise ValueError("Biosample manifest must contain 14 training biosamples")
    for split, names, group in (
        ("train", train_names, "traincell"),
        ("valid", train_names, "traincell"),
    ):
        requested = split_cfg["validation_biosamples" if split == "valid" else f"{split}_biosamples"]
        if requested != names or data[f"{split}_label_group"] != group:
            raise ValueError(f"{split}: biosample or label-group order differs")
        bedpe = Path(data[f"{split}_bedpe"])
        label = Path(data[f"{split}_label"])
        n_genes = count_bedpe_rows(str(bedpe))
        values = np.load(label, mmap_mode="r")
        if values.shape != (len(names), n_genes) or values.dtype != np.float32:
            raise ValueError(f"{label}: expected {(len(names), n_genes)} float32 labels")
        missing = [str(Path(data["hic_root"]) / split / f"{acc}.pkl")
                   for row, acc, name in table if name in names
                   and not (Path(data["hic_root"]) / split / f"{acc}.pkl").is_file()]
        if missing:
            raise FileNotFoundError(f"Missing {len(missing)} {split} raw Hi-C windows; first: {missing[0]}")
    if not Path(encoder["checkpoint_path"]).is_file():
        raise FileNotFoundError(encoder["checkpoint_path"])
    return size * 1024, grid, str(config["run_name"])


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.gpu is not None:
        if args.gpu < 0:
            raise SystemExit("--gpu must be a non-negative GPU index")
        # CUDA is not initialized before this point, so this selects the device.
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    config = load_config(args.config, include_test=False)
    config_lr = float(config["training"]["lr"])
    config_activation = str(config["decoder"].get("output_activation", "softplus"))
    config_seed = int(config.get("seed", 42))
    if args.learning_rate is not None:
        config["training"]["lr"] = args.learning_rate
    data_cfg = dict(config["data"])
    encoder_cfg = dict(config["encoder"])
    decoder_cfg = dict(config["decoder"])
    training_cfg = dict(config["training"])
    decoder_cfg["output_activation"] = str(
        args.output_activation or decoder_cfg.get("output_activation", "softplus"))
    if args.num_workers is not None:
        training_cfg["num_workers"] = int(args.num_workers)
    if args.no_wandb:
        training_cfg["use_wandb"] = False
    seed = int(config.get("seed", 42) if args.seed is None else args.seed)
    if args.fast:
        training_cfg.update(max_epochs=1, use_wandb=False, val_check_interval=1.0)

    config["decoder"] = decoder_cfg
    config["training"] = training_cfg

    window_bp, expected_grid, name = validate_experiment(config)
    print(
        f"Preflight passed: window={window_bp:,} bp, matrix={data_cfg['size']}x{data_cfg['size']}, "
        f"grid={expected_grid}x{expected_grid}, train/validation biosamples=14, "
        f"lr={float(training_cfg['lr']):g}, "
        f"precision={training_cfg['precision']}, "
        f"activation={decoder_cfg['output_activation']}"
    )
    if args.preflight_only:
        return 0

    encoder_cfg["checkpoint_path"] = str(ensure_encoder_checkpoint(encoder_cfg))

    input_size = int(encoder_cfg["input_size"])
    patch_size = int(encoder_cfg["patch_size"])
    if not bool(encoder_cfg.get("freeze", True)):
        raise ValueError("This experiment requires a frozen HiCFoundation encoder")

    torch.set_num_threads(int(os.environ["OMP_NUM_THREADS"]))
    torch.set_float32_matmul_precision(str(training_cfg.get("float32_matmul_precision", "high")))
    set_seed(seed, bool(training_cfg.get("deterministic", False)))
    workers = int(training_cfg.get("num_workers", 0))
    available_cpus = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None
    if available_cpus is not None:
        print(
            f"CPU affinity: {available_cpus} CPUs; DataLoader workers={workers}; "
            f"threads/process={torch.get_num_threads()}"
        )
        if workers >= available_cpus:
            print(
                "WARNING: num_workers should normally leave at least one affinity CPU "
                "for the training process"
            )

    output_activation = str(decoder_cfg["output_activation"])
    learning_rate = float(training_cfg["lr"])
    # Suffixes appear only for CLI overrides, so they never replace the default run.
    if output_activation != config_activation:
        name = f"{name}_{output_activation}"
    if learning_rate != config_lr:
        name = f"{name}_lr{learning_rate:.0e}"
    if seed != config_seed:
        name = f"{name}_seed{seed}"
    if args.fast:
        name = f"fastdev_{name}"
    output_dir = args.output_dir or Path(config["output_dir"])
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    (output_dir / "logs").mkdir(parents=True, exist_ok=True)
    resolved_config = dict(config)
    resolved_config["seed"] = seed
    resolved_config["decoder"] = decoder_cfg
    resolved_config["training"] = training_cfg
    with (output_dir / f"{name}_config.json").open("w") as handle:
        json.dump(resolved_config, handle, indent=2, sort_keys=True)

    logger_config = {
        "input_semantics": "RAW",
        "window_bp": window_bp,
        "matrix_bins": input_size,
        "patch_size": patch_size,
        "token_grid": [int(encoder_cfg["grid_rows"]), int(encoder_cfg["grid_cols"])],
        "encoder_frozen": True,
        "encoder_fp32": bool(encoder_cfg.get("force_fp32", False)),
        "embedding_cache": False,
        "encoder": encoder_cfg,
        "decoder": decoder_cfg,
        "training": training_cfg,
        "split": config["split"],
    }
    if bool(training_cfg.get("use_wandb", True)):
        logger = WandbLogger(
            project=str(training_cfg.get("wandb_project", "PugetHiCBaselines")),
            save_dir=str(output_dir / "logs"),
            name=name,
            config=logger_config,
        )
    else:
        logger = CSVLogger(save_dir=str(output_dir / "logs"), name=name)

    batch_size = int(training_cfg.get("batch_size", 4))
    accumulate = int(training_cfg.get("accumulate_grad_batches", 32))
    train_loader, _ = build_hicfoundation_loader(
        data_cfg=data_cfg,
        training_cfg=training_cfg,
        split="train",
        biosamples=config["split"]["train_biosamples"],
        shuffle=True,
        batch_size=batch_size,
    )
    valid_loader, _ = build_hicfoundation_loader(
        data_cfg=data_cfg,
        training_cfg=training_cfg,
        split="valid",
        biosamples=config["split"]["validation_biosamples"],
        shuffle=False,
        batch_size=batch_size,
    )
    steps_per_epoch = max(1, math.ceil(len(train_loader) / accumulate))

    model = make_model(encoder_cfg, decoder_cfg)
    from puget.hic_only_model import HiCFoundationLit, parameter_count

    total_parameters, trainable_parameters = parameter_count(model)
    print(
        f"HiCFoundation + decoder: {total_parameters:,} parameters "
        f"({trainable_parameters:,} trainable); complete {expected_grid}x{expected_grid} grid; "
        f"batch={batch_size} x {accumulate} = {batch_size * accumulate} effective; "
        f"optimizer steps/epoch={steps_per_epoch}"
    )
    lit_model = HiCFoundationLit(
        model,
        lr=float(training_cfg["lr"]),
        min_lr=float(training_cfg["min_lr"]),
        weight_decay=float(training_cfg["weight_decay"]),
        warmup_epochs=int(training_cfg["warmup_epochs"]),
        decay_epochs=int(training_cfg["decay_epochs"]),
        steps_per_epoch=steps_per_epoch,
        optimizer=str(training_cfg.get("optimizer", "adam")),
    )

    checkpoint = ModelCheckpoint(
        dirpath=str(output_dir),
        filename=f"{name}_best",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        save_last=False,
    )
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        logger=logger,
        callbacks=[
            EarlyStopping(
                monitor="val_loss", mode="min", patience=int(training_cfg.get("patience", 40))
            ),
            checkpoint,
            LearningRateMonitor(logging_interval="step"),
        ],
        max_epochs=int(training_cfg["max_epochs"]),
        precision=str(training_cfg.get("precision", "bf16-mixed")),
        val_check_interval=training_cfg.get("val_check_interval", 0.5),
        accumulate_grad_batches=accumulate,
        gradient_clip_val=float(training_cfg.get("gradient_clip_val", 1.0)),
        gradient_clip_algorithm="norm",
        log_every_n_steps=int(training_cfg.get("log_every_n_steps", 20)),
        benchmark=not bool(training_cfg.get("deterministic", False)),
        limit_train_batches=8 if args.fast else 1.0,
        limit_val_batches=4 if args.fast else 1.0,
    )
    trainer.fit(lit_model, train_loader, valid_loader, ckpt_path=args.resume)
    if not checkpoint.best_model_path:
        raise RuntimeError("No best checkpoint was created")
    summary = {
        "run_name": name,
        "config": str(Path(args.config).resolve()),
        "best_checkpoint": str(checkpoint.best_model_path),
        "best_val_loss": float(checkpoint.best_model_score),
    }
    summary_path = output_dir / f"{name}_validation.json"
    with summary_path.open("w") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    print(f"Best checkpoint: {checkpoint.best_model_path}; "
          f"validation MSE: {summary['best_val_loss']:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
