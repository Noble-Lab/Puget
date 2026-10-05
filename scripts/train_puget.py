import argparse
import hashlib
import json
import math
import os
import random
import shutil
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
# Honour the per-job thread budget whether launched via run_all.sh or directly.
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "8")

import pytorch_lightning as pl
import torch

torch.set_num_threads(int(os.environ["OMP_NUM_THREADS"]))
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger, WandbLogger

from scripts._puget_config import load_config
from puget.puget_data import build_split_loader, load_biosample_table, resolve_subset
from puget.puget_model import BiasModel, VariantLit


def hvg_gene_mask(labels, label_rows, top_frac=0.25):
    labels = np.asarray(labels)                     # (n_biosamples_total, n_genes)
    var = np.nanvar(labels[list(label_rows), :], axis=0)
    mask = np.zeros(var.shape[0], dtype=bool)
    keep = np.where(np.isfinite(var) & (var > 0))[0]
    if keep.size == 0:
        return mask
    k = max(1, int(np.ceil(top_frac * keep.size)))
    mask[keep[np.argsort(var[keep])[::-1][:k]]] = True
    return mask


def set_seed(seed: int, deterministic: bool = False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = bool(deterministic)
    torch.backends.cudnn.benchmark = not bool(deterministic)
    if deterministic:
        try:
            torch.use_deterministic_algorithms(True)
        except RuntimeError as e:
            print(f"[warn] {e}; continuing without full determinism")


def _abspath(root, p):
    return p if os.path.isabs(p) else os.path.join(root, p)


def resolve_inputs(cfg, *, include_test: bool = True):
    keys = ["train_bedpe", "val_bedpe", "train_seq", "val_seq",
            "train_label", "val_label", "hic_root", "biosamples_csv",
            "output_dir"]
    if include_test:
        keys.extend(("test_bedpe", "test_seq", "test_label", "pred_dir"))
    for key in keys:
        setattr(cfg, key, _abspath(ROOT, getattr(cfg, key)))


def preflight(cfg):
    if (cfg.hic_semantics != "oe_scale" or cfg.hic_transform != "log"
            or cfg.window_height != cfg.n_cols or cfg.window_width != cfg.n_cols):
        raise ValueError("Puget requires square O/E SCALE Hi-C windows with log transform")
    BiasModel(seq_input_dim=cfg.embed_dim, n_cols=cfg.n_cols,
              output_activation=cfg.output_activation, **cfg.model)
    table = load_biosample_table(cfg.biosamples_csv)
    missing = []
    for split, prefix, cells in (("train", "train", cfg.train_biosamples),
                                 ("valid", "val", cfg.val_biosamples)):
        names, _, accessions = resolve_subset(cells, table)
        expected_names = cfg.train_biosamples if split == "train" else cfg.val_biosamples
        expected = len(expected_names)
        if names != list(expected_names) or expected != 14:
            raise ValueError(f"{split}: wrong biosample order or count")
        group = cfg.get(f"{split}_label_group", "")
        if group != "traincell":
            raise ValueError(f"{split}: wrong label group {group!r}")
        for key in (f"{prefix}_bedpe", f"{prefix}_seq", f"{prefix}_label"):
            if not os.path.isfile(getattr(cfg, key)):
                missing.append(getattr(cfg, key))
        if not any(not os.path.isfile(getattr(cfg, key)) for key in
                   (f"{prefix}_bedpe", f"{prefix}_seq", f"{prefix}_label")):
            with open(getattr(cfg, f"{prefix}_bedpe")) as handle:
                n_genes = sum(bool(line.strip()) for line in handle)
            labels = np.load(getattr(cfg, f"{prefix}_label"), mmap_mode="r")
            sequence = np.load(getattr(cfg, f"{prefix}_seq"), mmap_mode="r")
            if labels.shape != (expected, n_genes) or labels.dtype != np.float32:
                raise ValueError(f"{split}: wrong label shape or dtype: {labels.shape} {labels.dtype}")
            if sequence.shape != (n_genes, int(cfg.n_cols), int(cfg.embed_dim)) or sequence.dtype != np.float16:
                raise ValueError(f"{split}: wrong sequence shape or dtype: {sequence.shape} {sequence.dtype}")
        for acc in accessions:
            path = os.path.join(cfg.hic_root, split, f"{acc}.pkl")
            if not os.path.isfile(path) or os.path.getsize(path) == 0:
                missing.append(path)
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} Puget inputs; first: {missing[0]}")
    print(f"Preflight passed: {cfg.run_name}; 14 train/validation biosamples")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--gpu", type=int, default=None,
                    help="physical GPU index for this single-GPU job (sets CUDA_VISIBLE_DEVICES)")
    ap.add_argument("--seed", type=int, default=None,
                    help="override cfg.seed; the run name gets a _seed<N> suffix so "
                         "seed replicates never collide in logs/models")
    ap.add_argument("--fast", action="store_true",
                    help="a few batches, one epoch, no wandb -- exercises the whole "
                         "training and validation path in minutes")
    ap.add_argument(
        "--output-activation", choices=("softplus", "linear"), default=None,
        help="override config output activation (config defaults to softplus)")
    ap.add_argument("--no-wandb", action="store_true",
                    help="use a local CSV logger instead of Weights & Biases")
    ap.add_argument("--preflight-only", action="store_true")
    ap.add_argument("--output-dir", help="override output_dir from the config")
    args = ap.parse_args(argv)
    if args.gpu is not None:
        if args.gpu < 0:
            raise SystemExit("--gpu must be a non-negative GPU index")
        # CUDA is not initialized before this point, so this selects the device.
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    cfg = load_config(args.config)
    resolve_inputs(cfg, include_test=False)
    if args.output_dir:
        cfg.output_dir = os.path.abspath(args.output_dir)
    config_activation = str(cfg.get("output_activation", "softplus"))
    config_seed = int(cfg.seed)
    cfg.output_activation = str(args.output_activation or config_activation)
    if cfg.output_activation not in ("softplus", "linear"):
        raise ValueError("output_activation must be softplus or linear")
    if args.seed is not None:
        cfg.seed = int(args.seed)
    if args.no_wandb:
        cfg.use_wandb = False
    if args.fast:
        cfg.max_epochs = 1
        cfg.use_wandb = False
        cfg.val_check_interval = 1.0
    # Suffixes appear only for CLI overrides, so they never replace the default run.
    if cfg.output_activation != config_activation:
        cfg.run_name = f"{cfg.run_name}_{cfg.output_activation}"
    if int(cfg.seed) != config_seed:
        cfg.run_name = f"{cfg.run_name}_seed{int(cfg.seed)}"
    if args.fast:
        cfg.run_name = f"fastdev_{cfg.run_name}"
    preflight(cfg)
    if args.preflight_only:
        return
    if str(cfg.get("accelerator", "gpu")) == "gpu" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; Puget paper reproduction requires a GPU")
    torch.set_float32_matmul_precision(str(cfg.get("float32_matmul_precision", "high")))
    set_seed(int(cfg.seed), bool(cfg.get("deterministic", False)))

    output_dir = _abspath(ROOT, cfg.output_dir)
    logger_dir = os.path.join(output_dir, "logs")
    os.makedirs(logger_dir, exist_ok=True)
    run = str(cfg.run_name)
    stable_best = os.path.join(output_dir, f"{run}_best.ckpt")
    if args.output_dir and os.path.exists(stable_best):
        raise FileExistsError(f"Refusing to overwrite {stable_best}")
    with open(os.path.join(output_dir, f"{run}_config.json"), "w") as f:
        json.dump(cfg.to_dict(), f, indent=2, sort_keys=True)

    if bool(cfg.get("use_wandb", False)):
        logger = WandbLogger(project=str(cfg.get("wandb_project", "PugetVariants")),
                             save_dir=logger_dir, name=run,
                             config=cfg.to_dict())
    else:
        logger = CSVLogger(save_dir=logger_dir, name=run)

    table = load_biosample_table(cfg.biosamples_csv)
    train_loader, _ = build_split_loader(cfg, "train", cfg.train_bedpe, cfg.train_seq,
                                         cfg.train_label, cfg.train_biosamples, table, True)
    val_loader, _ = build_split_loader(cfg, "valid", cfg.val_bedpe, cfg.val_seq,
                                       cfg.val_label, cfg.val_biosamples, table, False)

    steps_per_epoch = max(1, math.ceil(
        len(train_loader.dataset) / (int(cfg.batch_size) * int(cfg.accumulate_grad_batches))))
    print(f"train examples = {len(train_loader.dataset)}, steps_per_epoch = {steps_per_epoch}")

    model = BiasModel(seq_input_dim=cfg.embed_dim, n_cols=cfg.n_cols,
                      output_activation=cfg.output_activation, **cfg.model)
    print(f"[{run}] pointwise O/E Hi-C bias; "
          f"params={sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.2f}M")
    if bool(cfg.get("compile_model", False)):
        compile_mode = str(cfg.get("compile_mode", "default"))
        # nn.Module.compile() is in-place and leaves state_dict keys unchanged,
        # so the best checkpoint remains loadable by the ordinary eager model.
        model.compile(mode=compile_mode)
        print(f"torch.compile enabled (mode={compile_mode})")

    lit = VariantLit(model, lr=float(cfg.lr), min_lr=float(cfg.min_lr),
                     weight_decay=float(cfg.weight_decay),
                     warmup_epochs=int(cfg.warmup_epochs), decay_epochs=int(cfg.decay_epochs),
                     steps_per_train_epoch=steps_per_epoch)
    lit.val_gene_mask = hvg_gene_mask(val_loader.dataset.labels, val_loader.dataset.label_rows)

    ckpt_cb = ModelCheckpoint(dirpath=output_dir, filename=f"{run}_best", monitor="val_loss", save_top_k=1, mode="min",
                              save_last=False)
    callbacks = [EarlyStopping(monitor="val_loss", patience=int(cfg.patience), mode="min"),
                 ckpt_cb, LearningRateMonitor(logging_interval="step")]
    runtime = {
        "torch_version": torch.__version__, "lightning_version": pl.__version__,
        "cuda_version": torch.version.cuda, "cudnn_version": torch.backends.cudnn.version(),
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "torch_num_threads": torch.get_num_threads(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpu_name": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
        "cpu_rng_before_fit_sha256": hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest(),
        "source_sha256": {},
    }
    for source in (__file__, os.path.join(ROOT, "puget", "puget_model.py"),
                   os.path.join(ROOT, "puget", "puget_data.py")):
        with open(source, "rb") as handle:
            runtime["source_sha256"][os.path.relpath(source, ROOT)] = hashlib.sha256(handle.read()).hexdigest()
    with open(os.path.join(output_dir, f"{run}_runtime.json"), "w") as handle:
        json.dump(runtime, handle, indent=2, sort_keys=True)
    trainer = pl.Trainer(
        accelerator=str(cfg.get("accelerator", "gpu")), devices=int(cfg.get("devices", 1)),
        strategy="auto", logger=logger,
        callbacks=callbacks,
        max_epochs=int(cfg.max_epochs), precision=str(cfg.precision),
        val_check_interval=cfg.val_check_interval,
        gradient_clip_val=float(cfg.get("gradient_clip_val", 1.0)),
        gradient_clip_algorithm="norm",
        accumulate_grad_batches=int(cfg.accumulate_grad_batches),
        log_every_n_steps=int(cfg.get("log_every_n_steps", 10)),
        benchmark=not bool(cfg.get("deterministic", False)),
        limit_train_batches=8 if args.fast else 1.0,
        limit_val_batches=4 if args.fast else 1.0)

    trainer.fit(lit, train_loader, val_loader)
    best = ckpt_cb.best_model_path
    print(f"Best checkpoint: {best}")
    if not best:
        raise RuntimeError("no best checkpoint -- did val_loss log?")
    if os.path.abspath(best) != os.path.abspath(stable_best):
        shutil.copy2(best, stable_best)
        best = stable_best
    summary = {
        "config": os.path.abspath(args.config),
        "run_name": run,
        "best_checkpoint": str(best),
        "best_val_loss": float(ckpt_cb.best_model_score),
    }
    with open(os.path.join(output_dir, f"{run}_validation.json"), "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    print(f"Validation MSE: {summary['best_val_loss']:.6f}; checkpoint: {best}")


if __name__ == "__main__":
    main()
