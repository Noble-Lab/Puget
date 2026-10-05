from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
import pickle
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from scripts._puget_config import load_config
from scripts.train_puget import resolve_inputs
from puget.biosamples import load_biosample_table
from puget.puget_data import _apply_hic_transform, _dense_from_entry

DEFAULT_CONFIG = ROOT / "configs" / "training" / "borzoi_split_puget_alphagenome.yaml"
ATTRIBUTION_DIR = ROOT / "outputs" / "borzoi_split_attribution"
DEFAULT_BASELINE_DIR = ATTRIBUTION_DIR / "baseline_train14_distance_mean"
N_BINS = 512
WINDOW_BP = 524288
EMBED_DIM = 3072


def sha256(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_identity(path: Path | str) -> dict:
    path = Path(path).resolve()
    stat = path.stat()
    return {"path": str(path), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def read_json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def read_training_config(path: Path):
    if not path.is_file():
        raise FileNotFoundError(path)
    cfg = load_config(path)
    resolve_inputs(cfg)
    expected = {
        "hic_semantics": "oe_scale",
        "hic_transform": "log",
        "output_activation": "softplus",
        "n_cols": N_BINS,
        "window_height": N_BINS,
        "window_width": N_BINS,
        "embed_dim": EMBED_DIM,
    }
    for key, wanted in expected.items():
        if cfg.get(key) != wanted:
            raise ValueError(f"Expected {key}={wanted!r}, got {cfg.get(key)!r}")
    if len(cfg.train_biosamples) != 14 or len(set(cfg.train_biosamples)) != 14:
        raise ValueError("Expected exactly 14 unique training biosamples")
    if len(cfg.test_biosamples) != 2:
        raise ValueError("Expected the paper 14-training-cell plus 2-held-out-cell experiment")
    if set(cfg.train_biosamples) & set(cfg.test_biosamples):
        raise ValueError("Training and held-out biosamples overlap")
    return cfg


def gene_table(path: Path | str, strand: bool = False) -> pd.DataFrame:
    columns = [0, 1, 2, 6, 8] if strand else [0, 1, 2, 6]
    frame = pd.read_csv(path, sep="\t", header=None, usecols=columns)
    frame.columns = ["chrom", "window_start", "window_end", "gene_name", "strand"][: len(columns)]
    frame.insert(0, "gene_index", np.arange(len(frame), dtype=np.int64))
    frame["window_key"] = [
        f"{chrom}:{int(start)},{int(end)}"
        for chrom, start, end in zip(frame.chrom, frame.window_start, frame.window_end)
    ]
    if not ((frame.window_end - frame.window_start) == WINDOW_BP).all():
        raise ValueError(f"BEDPE contains a non-524,288-bp window: {path}")
    return frame


def hic_source(cfg, split: str, accession: str) -> Path:
    return Path(cfg.hic_root) / split / f"{accession}.pkl"


def transformed_map(entry) -> tuple[np.ndarray | None, str]:
    if entry is None:
        return None, "missing"
    row = np.asarray(entry["row"], dtype=np.int16)
    if bool(entry.get("is_empty", int(row.size == 0))):
        return None, "empty"
    if str(entry.get("source_semantics", "")).lower() != "oe_scale":
        raise ValueError("Expected O/E SCALE Hi-C windows")
    col = np.asarray(entry["col"], dtype=np.int16)
    data = np.asarray(entry["data"], dtype=np.float16)
    if not np.isfinite(data).all() or (data < 0).any():
        raise ValueError("Hi-C sparse values must be finite and nonnegative")
    if np.any(row < 0) or np.any(row >= N_BINS) or np.any(col < 0) or np.any(col >= N_BINS):
        raise ValueError("Hi-C sparse coordinate lies outside the 512x512 window")
    dense = np.nan_to_num(_dense_from_entry(N_BINS, N_BINS, row, col, data, entry))
    result = _apply_hic_transform(dense, "log")
    if not np.isfinite(result).all() or not np.allclose(result, result.T, rtol=0, atol=1e-6):
        raise ValueError("Expected a finite symmetric log10(SCALE O/E + 1) input")
    return result, "usable"


def validate_inputs(cfg) -> tuple[pd.DataFrame, list[tuple[int, str, str]]]:
    bed = gene_table(cfg.train_bedpe)
    manifest = load_biosample_table(cfg.biosamples_csv)
    by_name = {name: (int(row), accession) for row, accession, name in manifest}
    missing = [name for name in cfg.train_biosamples if name not in by_name]
    if missing:
        raise ValueError(f"Training biosamples missing from manifest: {missing}")
    cells = [(by_name[name][0], by_name[name][1], name) for name in cfg.train_biosamples]
    for _row, accession, _name in cells:
        source = hic_source(cfg, "train", accession)
        if not source.is_file() or source.stat().st_size == 0:
            raise FileNotFoundError(source)
    return bed, cells


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG,
                        help="Puget training YAML in configs/training")
    parser.add_argument("--output", type=Path, default=DEFAULT_BASELINE_DIR)
    args = parser.parse_args()
    args.config = args.config.resolve()
    args.output = args.output.resolve()
    return args


def main() -> int:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite completed baseline: {args.output}")
    partial = args.output.with_name(args.output.name + ".partial")
    if partial.exists():
        raise FileExistsError(f"Inspect interrupted baseline before rerunning: {partial}")

    cfg = read_training_config(args.config)
    genes, cells = validate_inputs(cfg)
    required = 2 * N_BINS * N_BINS * 4 + 5 * 2**20
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(args.output.parent).free < required:
        raise OSError("Insufficient space for baseline artifacts")
    partial.mkdir()

    distance = np.abs(np.arange(N_BINS)[:, None] - np.arange(N_BINS)[None, :])
    flat_distance = distance.ravel()
    distance_counts = np.bincount(flat_distance, minlength=N_BINS)
    profiles: list[np.ndarray] = []
    details: list[dict] = []
    dropped: list[dict] = []
    started = time.monotonic()

    for cell_number, (manifest_row, accession, name) in enumerate(cells, 1):
        source = hic_source(cfg, "train", accession)
        print(f"[{cell_number}/14] Loading {name}: {source}", flush=True)
        with source.open("rb") as handle:
            windows = pickle.load(handle)
        total = np.zeros(N_BINS, dtype=np.float64)
        counts = {"usable": 0, "missing": 0, "empty": 0}
        for gene in genes.itertuples(index=False):
            matrix, status = transformed_map(windows.get(gene.window_key))
            counts[status] += 1
            if matrix is None:
                dropped.append(
                    {
                        "gene_index": int(gene.gene_index),
                        "gene_name": gene.gene_name,
                        "biosample": name,
                        "reason": status,
                    }
                )
                continue
            total += np.bincount(
                flat_distance, weights=matrix.ravel(), minlength=N_BINS
            ) / distance_counts
            if (gene.gene_index + 1) % 2000 == 0:
                print(f"  {gene.gene_index + 1}/{len(genes)} training rows", flush=True)
        if counts["usable"] == 0:
            raise RuntimeError(f"No usable training windows for {name}")
        profiles.append(total / counts["usable"])
        details.append(
            {
                "name": name,
                "accession": accession,
                "manifest_row": manifest_row,
                "requested": len(genes),
                **counts,
                "source": file_identity(source),
            }
        )
        del windows
        gc.collect()
        print(f"  {counts}", flush=True)

    cell_profiles = np.stack(profiles)
    mean_profile = cell_profiles.mean(axis=0, dtype=np.float64).astype(np.float32)
    baseline = mean_profile[distance]
    if baseline.dtype != np.float32 or not np.isfinite(baseline).all() or (baseline < 0).any():
        raise FloatingPointError("Invalid computed baseline")
    if not np.array_equal(baseline, baseline.T):
        raise ValueError("Computed baseline is not exactly symmetric")

    np.save(partial / "baseline.npy", baseline)
    np.save(partial / "zero.npy", np.zeros_like(baseline))
    profile_table = pd.DataFrame(
        {
            "distance_bins": np.arange(N_BINS),
            "distance_bp": np.arange(N_BINS) * 1024,
            "mean_train14": mean_profile,
        }
    )
    for detail, values in zip(details, cell_profiles):
        profile_table[detail["name"]] = values
    profile_table.to_csv(partial / "distance_profiles.tsv", sep="\t", index=False)
    pd.DataFrame(
        dropped, columns=["gene_index", "gene_name", "biosample", "reason"]
    ).to_csv(partial / "dropped_training_pairs.tsv", sep="\t", index=False)

    write_json(
        partial / "meta.json",
        {
            "completed": True,
            "definition": (
                "untrimmed mean over transformed pixels at each absolute contact distance, "
                "then usable training-gene rows within cell, then equal mean over 14 training cells"
            ),
            "split": "train",
            "model_input_semantics": "log10(SCALE O/E + 1)",
            "implicit_sparse_zeros_included": True,
            "duplicate_gene_windows": "retain BEDPE gene-row weighting",
            "missing_empty_maps": "exclude within each cell and report every excluded pair",
            "n_training_gene_rows": len(genes),
            "n_training_biosamples": len(cells),
            "shape": [N_BINS, N_BINS],
            "dtype": "float32",
            "bin_bp": 1024,
            "baseline_sha256": sha256(partial / "baseline.npy"),
            "training_config": file_identity(args.config),
            "training_config_sha256": sha256(args.config),
            "train_bedpe": file_identity(cfg.train_bedpe),
            "train_bedpe_sha256": sha256(cfg.train_bedpe),
            "biosamples_manifest": file_identity(cfg.biosamples_csv),
            "biosamples_manifest_sha256": sha256(cfg.biosamples_csv),
            "biosamples": details,
            "producer": file_identity(__file__),
            "producer_sha256": sha256(__file__),
            "elapsed_sec": time.monotonic() - started,
        },
    )
    partial.rename(args.output)
    print(f"Saved {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
