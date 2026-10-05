from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(variable, "1")

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from scripts.build_ig_baseline import DEFAULT_CONFIG, N_BINS, read_training_config, sha256
from puget.biosamples import load_biosample_table
from puget.puget_data import SeqHiCDataset
from puget.puget_model import BiasModel, VariantLit

DEFAULT_CHECKPOINT = ROOT / "data" / "trained_models" / "borzoi_split" / "puget_alphagenome_best.ckpt"
DEFAULT_PREDICTIONS = ROOT / "outputs" / "borzoi_split_inference" / "puget_alphagenome"
DEFAULT_OUTPUT = ROOT / "outputs" / "borzoi_split_ablation" / "puget_alphagenome"
MODES = ("real", "zero", "diag_shuffle", "promoter_anchored")
PROMOTER = slice(254, 258)
PAPER_SEED = 42
# Bootstrap seeds are fixed, so --seed changes only the diagonal shuffles.
BOOTSTRAP_SEED = 42
N_GENES = 1690
N_BIOSAMPLES = 16
N_BOOT = 2000
HVG_FRACS = (0.10, 0.25, 0.50, 1.00)


def atomic_save_npy(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.partial")
    with temporary.open("wb") as handle:
        np.save(handle, np.asarray(value, dtype=np.float32))
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_save_npz(path: Path, **values) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.partial")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **values)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_write(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.partial")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def stable_seed(base_seed: int, biosample: str, gene_index: int) -> int:
    payload = f"{int(base_seed)}:{biosample}:{int(gene_index)}".encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, "little") & 0x7FFFFFFF


def diag_shuffle_symmetric(matrix: np.ndarray, seed: int) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.shape != (N_BINS, N_BINS):
        raise ValueError(f"Expected {(N_BINS, N_BINS)}, got {matrix.shape}")
    if not np.allclose(matrix, matrix.T, rtol=0, atol=1e-6):
        raise ValueError("Diagonal shuffle requires a symmetric transformed input")
    output = matrix.copy()
    rng = np.random.default_rng(int(seed))
    for distance in range(1, N_BINS):
        row = np.arange(N_BINS - distance)
        col = row + distance
        values = matrix[row, col].copy()
        rng.shuffle(values)
        output[row, col] = values
        output[col, row] = values
    return output


def promoter_anchored_symmetric(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.shape != (N_BINS, N_BINS):
        raise ValueError(f"Expected {(N_BINS, N_BINS)}, got {matrix.shape}")
    if not np.allclose(matrix, matrix.T, rtol=0, atol=1e-6):
        raise ValueError("Promoter-contact ablation requires a symmetric input")
    output = np.zeros_like(matrix)
    output[PROMOTER, :] = matrix[PROMOTER, :]
    output[:, PROMOTER] = matrix[:, PROMOTER]
    return output


def self_test(seed: int) -> None:
    rng = np.random.default_rng(7)
    matrix = rng.normal(size=(N_BINS, N_BINS)).astype(np.float32)
    matrix = (matrix + matrix.T) / 2
    example_seed = stable_seed(seed, "test-cell", 123)
    shuffled_a = diag_shuffle_symmetric(matrix, example_seed)
    shuffled_b = diag_shuffle_symmetric(matrix, example_seed)
    if not np.array_equal(shuffled_a, shuffled_b):
        raise AssertionError("Stable diagonal shuffle is not exactly reproducible")
    if not np.array_equal(shuffled_a, shuffled_a.T):
        raise AssertionError("Diagonal shuffle lost symmetry")
    if not np.array_equal(np.diag(shuffled_a), np.diag(matrix)):
        raise AssertionError("Diagonal shuffle changed the main diagonal")
    for distance in range(1, N_BINS):
        row = np.arange(N_BINS - distance)
        if not np.array_equal(
            np.sort(shuffled_a[row, row + distance]),
            np.sort(matrix[row, row + distance]),
        ):
            raise AssertionError(f"Diagonal {distance} value multiset changed")

    anchored = promoter_anchored_symmetric(matrix)
    keep = np.zeros((N_BINS, N_BINS), dtype=bool)
    keep[PROMOTER, :] = True
    keep[:, PROMOTER] = True
    if not np.array_equal(anchored, anchored.T):
        raise AssertionError("Promoter-contact ablation lost symmetry")
    if not np.array_equal(anchored[keep], matrix[keep]) or np.any(anchored[~keep]):
        raise AssertionError("Promoter-contact keep mask is incorrect")


def configure_determinism(seed: int, float32_matmul_precision: str) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_float32_matmul_precision(float32_matmul_precision)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def label_path(cfg, group: str) -> Path:
    name = "traincell_testgene.npy" if group == "train14" else "testcell_testgene.npy"
    return Path(cfg.test_label).with_name(name)


def validate_inputs(args):
    if not args.checkpoint.is_file() or args.checkpoint.stat().st_size == 0:
        raise FileNotFoundError(args.checkpoint)
    cfg = read_training_config(args.config)
    for path in (cfg.test_bedpe, cfg.test_seq, label_path(cfg, "train14"),
                 label_path(cfg, "test2"), cfg.biosamples_csv):
        if not Path(path).is_file() or Path(path).stat().st_size == 0:
            raise FileNotFoundError(path)
    with Path(cfg.test_bedpe).open() as handle:
        if sum(1 for _ in handle) != N_GENES:
            raise ValueError(f"Expected {N_GENES} Borzoi-split test genes")
    sequence = np.load(cfg.test_seq, mmap_mode="r")
    if sequence.shape != (N_GENES, N_BINS, int(cfg.embed_dim)) or sequence.dtype != np.float16:
        raise ValueError(f"Unexpected sequence embedding: {sequence.shape}, {sequence.dtype}")
    train_truth = np.load(label_path(cfg, "train14"), mmap_mode="r")
    test_truth = np.load(label_path(cfg, "test2"), mmap_mode="r")
    if train_truth.shape != (14, N_GENES) or test_truth.shape != (2, N_GENES):
        raise ValueError(f"Unexpected truth grids: {train_truth.shape}, {test_truth.shape}")

    table = load_biosample_table(cfg.biosamples_csv)
    names = [row[2] for row in table]
    expected_names = list(cfg.train_biosamples) + list(cfg.test_biosamples)
    if len(table) != N_BIOSAMPLES or names != expected_names:
        raise ValueError(f"Unexpected canonical biosample order: {names}")
    for _index, accession, _name in table:
        path = Path(cfg.hic_root) / "test" / f"{accession}.pkl"
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(path)
    return cfg, table, names


def load_real_predictions(args, cfg, names: list[str]) -> np.ndarray:
    run = str(cfg.run_name)
    grid = np.full((N_BIOSAMPLES, N_GENES), np.nan, dtype=np.float32)
    artifacts = (
        (args.predictions_dir / f"{run}_testgenes_train14_pred.npy",
         args.predictions_dir / f"{run}_testgenes_train14_biosamples.txt"),
        (args.predictions_dir / f"{run}_testpred.npy",
         args.predictions_dir / f"{run}_test_biosamples.txt"),
    )
    for prediction_path, names_path in artifacts:
        if not prediction_path.is_file() or not names_path.is_file():
            raise FileNotFoundError(
                f"Missing real-Hi-C predictions; run scripts/infer_puget.py first: {prediction_path}"
            )
        values = np.load(prediction_path)
        artifact_names = names_path.read_text(encoding="utf-8").splitlines()
        if values.shape != (len(artifact_names), N_GENES) or not np.isfinite(values).all():
            raise ValueError(f"Invalid real-prediction artifact: {prediction_path}")
        for row, name in enumerate(artifact_names):
            grid[names.index(name)] = values[row]
    if not np.isfinite(grid).all():
        raise ValueError("Real prediction grid is incomplete")
    return grid


class PerturbedDataset(Dataset):
    def __init__(self, base: Dataset, base_seed: int):
        self.base = base
        self.base_seed = int(base_seed)

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int):
        image, sequence, target, metadata = self.base[index]
        matrix = image[0].numpy()
        biosample = str(metadata[0])
        gene_index = int(metadata[6])
        shuffled = diag_shuffle_symmetric(
            matrix, stable_seed(self.base_seed, biosample, gene_index)
        )
        anchored = promoter_anchored_symmetric(matrix)
        return (
            torch.from_numpy(anchored).unsqueeze(0),
            torch.from_numpy(shuffled).unsqueeze(0),
            sequence,
            target,
            metadata,
        )


def collate_perturbed(batch):
    anchored, shuffled, sequences, targets, metadata = zip(*batch)
    return (
        torch.stack(anchored),
        torch.stack(shuffled),
        torch.stack(sequences),
        torch.stack(targets),
        list(metadata),
    )


def make_cell_loader(cfg, table, cell_index: int, args):
    _manifest_row, accession, name = table[cell_index]
    if name in cfg.train_biosamples:
        labels = label_path(cfg, "train14")
        label_row = list(cfg.train_biosamples).index(name)
    else:
        labels = label_path(cfg, "test2")
        label_row = list(cfg.test_biosamples).index(name)
    base = SeqHiCDataset(
        bedpe_path=cfg.test_bedpe,
        pkl_paths=[str(Path(cfg.hic_root) / "test" / f"{accession}.pkl")],
        label_rows=[label_row],
        biosample_names=[name],
        seq_emb_npy=cfg.test_seq,
        label_npy=str(labels),
        window_height=N_BINS,
        window_width=N_BINS,
        skip_missing_hic=True,
        hic_transform="log",
        seq_emb_mmap_mode="r",
    )
    if len(base) != N_GENES:
        raise ValueError(f"{name}: expected {N_GENES} usable Hi-C windows, got {len(base)}")
    loader = DataLoader(
        PerturbedDataset(base, args.seed),
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        prefetch_factor=1 if args.num_workers > 0 else None,
        collate_fn=collate_perturbed,
    )
    return loader, accession, name


def autocast_settings(cfg, device: torch.device) -> tuple[bool, torch.dtype]:
    precision = str(cfg.precision)
    enabled = device.type == "cuda" and "mixed" in precision
    dtype = torch.bfloat16 if "bf16" in precision else torch.float16
    return enabled, dtype


def load_model(args, cfg, device: torch.device):
    lit = VariantLit.load_from_checkpoint(
        str(args.checkpoint),
        model=BiasModel(seq_input_dim=cfg.embed_dim, n_cols=cfg.n_cols,
                        output_activation=cfg.output_activation, **cfg.model),
        strict=True,
        map_location="cpu",
    ).eval().to(device)
    for parameter in lit.parameters():
        parameter.requires_grad_(False)
    if lit.model.output_activation != "softplus":
        raise ValueError("Loaded model is not the paper Softplus variant")
    return lit


@torch.inference_mode()
def predict_zero(lit, cfg, device: torch.device, batch_size: int) -> np.ndarray:
    sequence = np.load(cfg.test_seq, mmap_mode="r")
    values = np.empty(N_GENES, dtype=np.float32)
    use_amp, amp_dtype = autocast_settings(cfg, device)
    for start in range(0, N_GENES, batch_size):
        stop = min(start + batch_size, N_GENES)
        seq = torch.from_numpy(np.array(sequence[start:stop], copy=True)).to(device)
        image = torch.zeros(stop - start, 1, N_BINS, N_BINS, dtype=torch.float32, device=device)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            output = lit(image, seq)
        values[start:stop] = output.float().cpu().numpy().reshape(-1)
    if not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("Invalid zero-Hi-C predictions")
    return values


@torch.inference_mode()
def predict_cell(lit, cfg, loader, device: torch.device, name: str) -> tuple[np.ndarray, np.ndarray]:
    anchored_values = np.full(N_GENES, np.nan, dtype=np.float32)
    shuffled_values = np.full(N_GENES, np.nan, dtype=np.float32)
    use_amp, amp_dtype = autocast_settings(cfg, device)
    for batch_index, (anchored, shuffled, sequence, _target, metadata) in enumerate(loader, start=1):
        sequence = sequence.to(device, non_blocking=True)
        for image, output_grid in ((anchored, anchored_values), (shuffled, shuffled_values)):
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                output = lit(image.to(device, non_blocking=True), sequence)
            for value, meta in zip(output.float().cpu().numpy().reshape(-1), metadata):
                if str(meta[0]) != name:
                    raise ValueError(f"Loader returned {meta[0]!r} while processing {name!r}")
                output_grid[int(meta[6])] = float(value)
        if batch_index % 20 == 0 or batch_index == len(loader):
            print(f"[{name}] batch {batch_index}/{len(loader)}", flush=True)
    for mode, values in (("promoter_anchored", anchored_values), ("diag_shuffle", shuffled_values)):
        if not np.isfinite(values).all() or np.any(values < 0):
            raise ValueError(f"Invalid {mode} predictions for {name}")
    return anchored_values, shuffled_values


def load_completed_cell(path: Path, cell_index: int, accession: str, name: str, seed: int):
    if not path.is_file():
        return None
    with np.load(path, allow_pickle=False) as archive:
        recorded = (
            int(archive["cell_index"]),
            str(archive["accession"]),
            str(archive["biosample"]),
            int(archive["base_seed"]),
        )
        expected = (cell_index, accession, name, seed)
        if recorded != expected:
            raise ValueError(f"Stale per-cell ablation artifact: {path}; {recorded} != {expected}")
        anchored = archive["promoter_anchored"].copy()
        shuffled = archive["diag_shuffle"].copy()
    if anchored.shape != (N_GENES,) or shuffled.shape != (N_GENES,):
        raise ValueError(f"Unexpected per-cell prediction shape: {path}")
    if not np.isfinite(anchored).all() or not np.isfinite(shuffled).all():
        raise ValueError(f"Nonfinite per-cell prediction: {path}")
    return anchored, shuffled


def corr_rows(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    output = np.full(x.shape[0], np.nan)
    for row in range(x.shape[0]):
        keep = np.isfinite(x[row]) & np.isfinite(y[row])
        if keep.sum() >= 2 and np.std(x[row, keep]) > 0 and np.std(y[row, keep]) > 0:
            output[row] = np.corrcoef(x[row, keep], y[row, keep])[0, 1]
    return output


def bootstrap_mean_ci(values: np.ndarray, seed: int) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    means = np.mean(rng.choice(values, size=(N_BOOT, len(values)), replace=True), axis=1)
    return float(values.mean()), *np.quantile(means, [0.025, 0.975]).tolist()


def write_metric_tables(grids: dict[str, np.ndarray], names: list[str], cfg, table_dir: Path) -> None:
    truth = np.vstack((
        np.load(label_path(cfg, "train14")),
        np.load(label_path(cfg, "test2")),
    )).astype(np.float64)
    if truth.shape != (N_BIOSAMPLES, N_GENES):
        raise ValueError(f"Unexpected combined truth shape: {truth.shape}")
    labels = {
        "real": "Real Hi-C",
        "zero": "Zero Hi-C",
        "diag_shuffle": "Diagonal shuffled",
        "promoter_anchored": "Promoter contacts",
    }
    observed_variance = np.var(truth, axis=0)
    eligible = np.flatnonzero(np.isfinite(observed_variance) & (observed_variance > 0))
    hvg_order = eligible[np.argsort(-observed_variance[eligible], kind="stable")]
    per_biosample_rows, summary_rows, per_gene_rows, hvg_rows = [], [], [], []
    for condition_i, condition in enumerate(MODES):
        across_genes = corr_rows(grids[condition], truth)
        for biosample, value in zip(names, across_genes):
            per_biosample_rows.append({
                "condition": condition, "label": labels[condition], "biosample": biosample,
                "pearson_across_genes": value,
            })
        mean, low, high = bootstrap_mean_ci(across_genes, BOOTSTRAP_SEED + condition_i)
        summary_rows.append({
            "condition": condition, "label": labels[condition],
            "n_biosamples": N_BIOSAMPLES, "n_genes": N_GENES,
            "mean": mean, "ci_low": low, "ci_high": high,
        })
        across_cells = corr_rows(grids[condition].T, truth.T)
        for gene_index, value in enumerate(across_cells):
            per_gene_rows.append({
                "condition": condition, "label": labels[condition],
                "gene_index": gene_index, "observed_variance": observed_variance[gene_index],
                "pearson_across_cell_lines": value,
            })
        for frac_i, frac in enumerate(HVG_FRACS):
            indices = hvg_order[:max(1, int(np.ceil(frac * len(hvg_order))))]
            values = across_cells[indices]
            mean, low, high = bootstrap_mean_ci(
                values, BOOTSTRAP_SEED + 100 * condition_i + frac_i
            )
            hvg_rows.append({
                "condition": condition, "label": labels[condition],
                "hvg_fraction": frac, "n_selected_genes": len(indices),
                "n_valid_genes": int(np.isfinite(values).sum()),
                "mean": mean, "ci_low": low, "ci_high": high,
            })
    table_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(per_biosample_rows).to_csv(
        table_dir / "across_gene_per_biosample.tsv", sep="\t", index=False
    )
    pd.DataFrame(summary_rows).to_csv(
        table_dir / "across_gene_summary.tsv", sep="\t", index=False
    )
    pd.DataFrame(per_gene_rows).to_csv(
        table_dir / "per_gene_across_cell_lines.tsv", sep="\t", index=False
    )
    pd.DataFrame(hvg_rows).to_csv(
        table_dir / "hvg_per_gene_across_cell_lines_summary.tsv", sep="\t", index=False
    )


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG,
                        help="Puget training YAML in configs/training")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--predictions-dir", type=Path, default=DEFAULT_PREDICTIONS,
                        help="directory with the real-Hi-C arrays from scripts/infer_puget.py")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--gpu", type=int, default=None,
                        help="physical GPU index for this job (sets CUDA_VISIBLE_DEVICES)")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=PAPER_SEED,
                        help=f"diagonal-shuffle base seed (paper: {PAPER_SEED})")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args(argv)
    if args.batch_size < 1 or args.num_workers < 0:
        parser.error("batch size must be positive and workers nonnegative")
    if args.gpu is not None:
        if args.gpu < 0:
            parser.error("--gpu must be a non-negative GPU index")
        # CUDA is not initialized before this point, so this selects the device.
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    for name in ("config", "checkpoint", "predictions_dir", "output"):
        setattr(args, name, getattr(args, name).resolve())
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    self_test(args.seed)
    cfg, table, names = validate_inputs(args)
    real = load_real_predictions(args, cfg, names)
    print("Transformation self-tests and Borzoi-split preflight passed.")
    print(f"Target: {N_BIOSAMPLES} biosamples x {N_GENES} test genes; seed={args.seed}")
    print(f"Output: {args.output}")
    if args.preflight_only:
        return 0
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; no inference was started")

    device = torch.device(args.device)
    configure_determinism(args.seed, str(cfg.get("float32_matmul_precision", "high")))
    lit = load_model(args, cfg, device)

    started = time.time()
    args.output.mkdir(parents=True, exist_ok=True)
    atomic_save_npy(args.output / "real_pred.npy", real)
    atomic_write(args.output / "biosamples.txt", "\n".join(names) + "\n")

    zero_path = args.output / "zero_pred.npy"
    if zero_path.is_file():
        zero = np.load(zero_path)
        if zero.shape != (N_BIOSAMPLES, N_GENES) or not np.isfinite(zero).all():
            raise ValueError(f"Invalid existing zero prediction grid: {zero_path}")
    else:
        zero_one = predict_zero(lit, cfg, device, args.batch_size)
        zero = np.broadcast_to(zero_one, (N_BIOSAMPLES, N_GENES)).copy()
        atomic_save_npy(zero_path, zero)
        print(f"[zero] inferred {N_GENES:,} sequence-only values and broadcast across "
              f"{N_BIOSAMPLES} biosamples")

    anchored = np.full((N_BIOSAMPLES, N_GENES), np.nan, dtype=np.float32)
    shuffled = np.full((N_BIOSAMPLES, N_GENES), np.nan, dtype=np.float32)
    cell_dir = args.output / "cells"
    for cell_index, (_manifest_row, accession, name) in enumerate(table):
        cell_path = cell_dir / f"{cell_index:02d}_{accession}.npz"
        completed = load_completed_cell(cell_path, cell_index, accession, name, args.seed)
        if completed is None:
            loader, loaded_accession, loaded_name = make_cell_loader(cfg, table, cell_index, args)
            if (loaded_accession, loaded_name) != (accession, name):
                raise RuntimeError("Cell loader identity changed")
            cell_anchored, cell_shuffled = predict_cell(lit, cfg, loader, device, name)
            atomic_save_npz(
                cell_path,
                cell_index=np.int64(cell_index),
                accession=np.asarray(accession),
                biosample=np.asarray(name),
                base_seed=np.int64(args.seed),
                promoter_anchored=cell_anchored,
                diag_shuffle=cell_shuffled,
            )
            del loader
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
        else:
            cell_anchored, cell_shuffled = completed
            print(f"[{name}] reused completed per-cell artifact", flush=True)
        anchored[cell_index] = cell_anchored
        shuffled[cell_index] = cell_shuffled

    grids = {
        "real": real,
        "zero": zero,
        "diag_shuffle": shuffled,
        "promoter_anchored": anchored,
    }
    for mode, grid in grids.items():
        if grid.shape != (N_BIOSAMPLES, N_GENES) or not np.isfinite(grid).all():
            raise ValueError(f"Incomplete {mode} grid")
        if mode in {"diag_shuffle", "promoter_anchored"}:
            atomic_save_npy(args.output / f"{mode}_pred.npy", grid)

    write_metric_tables(grids, names, cfg, args.output / "tables")
    metadata = {
        "completed": True,
        "description": "Deterministic Hi-C input ablations on Borzoi-split test genes",
        "modes": list(MODES),
        "prediction_layout": "biosample_x_test_gene_in_Borzoi_test_BEDPE_order",
        "shape": [N_BIOSAMPLES, N_GENES],
        "run_name": str(cfg.run_name),
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": sha256(args.checkpoint),
        "config": str(args.config),
        "config_sha256": sha256(args.config),
        "real_predictions_dir": str(args.predictions_dir),
        "test_bedpe": str(Path(cfg.test_bedpe).resolve()),
        "test_bedpe_sha256": sha256(cfg.test_bedpe),
        "test_sequence": str(Path(cfg.test_seq).resolve()),
        "hic_root": str(Path(cfg.hic_root).resolve()),
        "biosamples": names,
        "zero": {
            "space": "transformed model input log10(SCALE O/E + 1)",
            "value": 0.0,
            "optimization": "one prediction per gene broadcast across cells; sequence is cell-invariant",
        },
        "shuffle": {
            "kind": "symmetric within-positive-offset-diagonal permutation",
            "main_diagonal": "unchanged",
            "base_seed": args.seed,
            "per_example_seed": "blake2b(base_seed:biosample:gene_index), first 8 bytes little-endian masked to 31 bits",
            "space": "transformed model input log10(SCALE O/E + 1)",
            "numpy_version": np.__version__,
        },
        "promoter_anchored": {
            "row_slice_python": [PROMOTER.start, PROMOTER.stop],
            "zero_based_bins": list(range(PROMOTER.start, PROMOTER.stop)),
            "relative_interval_bp": [-2048, 2048],
            "rule": "retain pixel if row OR column is a promoter bin",
            "symmetric": True,
        },
        "metrics": {
            "bootstrap_resamples": N_BOOT,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "hvg_fractions": list(HVG_FRACS),
        },
        "determinism": {
            "torch_use_deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "allow_tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
            "allow_tf32_cudnn": torch.backends.cudnn.allow_tf32,
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "data_order": "canonical manifest order; shuffle=False",
            "model_mode": "eval/inference_mode",
            "scope": "bitwise repeatability is targeted on the same software and device stack",
        },
        "runtime": {
            "device": str(device),
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "elapsed_sec": time.time() - started,
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        },
        "output_sha256": {
            f"{mode}_pred.npy": sha256(args.output / f"{mode}_pred.npy") for mode in MODES
        },
    }
    atomic_write(args.output / "meta.json", json.dumps(metadata, indent=2, allow_nan=False) + "\n")
    print(f"Saved deterministic ablation predictions and metric tables to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
