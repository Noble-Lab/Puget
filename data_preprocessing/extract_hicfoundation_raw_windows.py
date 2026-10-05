# ---------------------------------------------------------------------------------
# The genome-to-submatrix and accession-wide input-count conventions build on
# the HiCFoundation paper utility.
# Source: https://github.com/Noble-Lab/HiCFoundation_paper/blob/main/utils/scan_array_diag.py
# License: Apache License 2.0
# This version selects BEDPE gene windows and area-rebins raw/NONE contacts.
# ---------------------------------------------------------------------------------

from __future__ import annotations

import argparse
import os
import pickle
import time
from pathlib import Path
from typing import Any

# Set limits before importing NumPy/SciPy, which load their thread pools.
_threads = os.environ.get("PUGET_HIC_THREADS", "1")
if not _threads.isdecimal() or int(_threads) < 1:
    raise ValueError("PUGET_HIC_THREADS must be a positive integer")
for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = _threads

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix, issparse


BEDPE_COLUMNS = (
    "chr1",
    "start1",
    "end1",
    "chr2",
    "start2",
    "end2",
    "name",
    "score",
    "strand1",
    "strand2",
)


def chromosome_name(value: object) -> str:
    value = str(value)
    return value if value.startswith("chr") else f"chr{value}"


def symmetric_csr(matrix: Any, source_name: str):
    if not issparse(matrix):
        raise TypeError(f"{source_name}: expected a scipy sparse matrix")
    matrix = matrix.tocoo(copy=True)
    matrix.sum_duplicates()
    row = np.asarray(matrix.row)
    col = np.asarray(matrix.col)
    data = np.asarray(matrix.data)
    if np.any(row > col):
        raise ValueError(f"{source_name}: expected upper-triangular raw contacts")
    if not np.isfinite(data).all() or np.any(data < 0):
        raise ValueError(f"{source_name}: raw contacts must be finite and non-negative")

    off_diagonal = row != col
    result = coo_matrix(
        (
            np.concatenate((data, data[off_diagonal])).astype(np.float32, copy=False),
            (
                np.concatenate((row, col[off_diagonal])),
                np.concatenate((col, row[off_diagonal])),
            ),
        ),
        shape=matrix.shape,
        dtype=np.float32,
    )
    result.sum_duplicates()
    return result.tocsr()


def load_raw_genome(path: Path, source_resolution: int) -> tuple[dict[str, Any], np.float64]:
    print(f"Loading raw/NONE source: {path}", flush=True)
    with path.open("rb") as handle:
        source = pickle.load(handle)
    if not isinstance(source, dict) or not source:
        raise TypeError(f"{path}: expected a non-empty chromosome dictionary")
    metadata_count = None
    if "chromosomes" in source:
        metadata = source.get("metadata")
        if not isinstance(metadata, dict):
            raise TypeError(f"{path}: missing genome metadata")
        if (metadata.get("matrix_type"), metadata.get("normalization")) != ("observed", "NONE"):
            raise ValueError(f"{path}: expected raw/NONE contacts")
        if int(metadata.get("source_resolution_bp", -1)) != source_resolution:
            raise ValueError(f"{path}: source resolution does not match --source-resolution")
        metadata_count = float(metadata.get("total_count", 0.0))
        if not np.isfinite(metadata_count) or metadata_count <= 0:
            raise ValueError(f"{path}: invalid metadata.total_count {metadata_count}")
        source = source["chromosomes"]
    if not isinstance(source, dict) or not source:
        raise TypeError(f"{path}: missing chromosome matrices")

    genome: dict[str, Any] = {}
    total_count = np.float64(0.0)
    # Pop entries as they are converted so source COO and reconstructed CSR
    # arrays are not retained together for the entire genome.
    for source_chromosome in list(source):
        matrix = source.pop(source_chromosome)
        if not issparse(matrix):
            raise TypeError(f"{path}:{source_chromosome}: expected a scipy sparse matrix")
        matrix = matrix.tocoo(copy=False)
        values = np.asarray(matrix.data)
        if not np.isfinite(values).all() or np.any(values < 0):
            raise ValueError(
                f"{path}:{source_chromosome}: raw contacts must be finite and non-negative"
            )
        total_count += values.sum(dtype=np.float64)
        normalized = chromosome_name(source_chromosome)
        if normalized in genome:
            raise ValueError(f"{path}: duplicate normalized chromosome {normalized}")
        genome[normalized] = symmetric_csr(matrix, f"{path}:{source_chromosome}")

    if not np.isfinite(total_count) or total_count <= 0:
        raise ValueError(f"{path}: invalid raw total_count {total_count}")
    if metadata_count is not None and not np.isclose(total_count, metadata_count, rtol=1e-6):
        raise ValueError(f"{path}: metadata.total_count does not match source matrices")
    print(f"Raw total_count: {total_count:.12g}", flush=True)
    return genome, total_count


def load_bedpe(path: Path, window_bp: int) -> pd.DataFrame:
    frame = pd.read_csv(path, sep="\t", header=None, names=BEDPE_COLUMNS)
    if frame.empty:
        raise ValueError(f"{path}: BEDPE is empty")
    for column in ("start1", "end1", "start2", "end2"):
        frame[column] = pd.to_numeric(frame[column], errors="raise").astype("int64")
    frame["chr1"] = frame["chr1"].map(chromosome_name)
    frame["chr2"] = frame["chr2"].map(chromosome_name)
    square = (
        (frame.chr1 == frame.chr2)
        & (frame.start1 == frame.start2)
        & (frame.end1 == frame.end2)
    )
    if not bool(square.all()):
        raise ValueError(f"{path}: every row must describe one square cis window")
    if not bool(((frame.end1 - frame.start1) == window_bp).all()):
        raise ValueError(f"{path}: every row must span exactly {window_bp} bp")
    if not bool(frame.strand1.isin(("+", "-")).all()):
        raise ValueError(f"{path}: strand1 must be '+' or '-'")
    return frame


def rebin_weights(
    window_start_bp: int,
    source_start_bin: int,
    source_bins: int,
    target_bins: int,
    source_resolution: int,
    target_resolution: int,
) -> np.ndarray:
    target_start = (
        window_start_bp
        + np.arange(target_bins, dtype=np.int64)[:, None] * target_resolution
    )
    target_end = target_start + target_resolution
    source_start = (
        source_start_bin
        + np.arange(source_bins, dtype=np.int64)[None, :]
    ) * source_resolution
    source_end = source_start + source_resolution
    overlap = np.maximum(
        0,
        np.minimum(target_end, source_end) - np.maximum(target_start, source_start),
    )
    result = overlap.astype(np.float32) / float(target_resolution)
    if not np.allclose(result.sum(axis=1), 1.0, atol=1e-6):
        raise ValueError("source bins do not completely cover the target window")
    return result


def upper_triangle_triplets(
    matrix: np.ndarray, value_dtype: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    matrix = ((matrix + matrix.T) * 0.5).astype(np.float32, copy=False)
    if not np.isfinite(matrix).all() or np.any(matrix < 0):
        raise ValueError("rebinned raw contact matrix contains invalid values")
    if value_dtype == "float16" and float(matrix.max(initial=0.0)) > np.finfo(np.float16).max:
        raise OverflowError("a raw contact exceeds float16 range; use --value-dtype float32")

    sparse = coo_matrix(matrix)
    keep = (sparse.data > 0) & (sparse.row <= sparse.col)
    dtype = np.float16 if value_dtype == "float16" else np.float32
    row = sparse.row[keep].astype(np.int16)
    col = sparse.col[keep].astype(np.int16)
    data = sparse.data[keep].astype(dtype)
    # Avoid explicit zeros if a very small positive value underflows in float16.
    positive = data > 0
    return row[positive], col[positive], data[positive]


def extract_split(
    genome: dict[str, Any],
    total_count: np.float64,
    bedpe_path: Path,
    output_path: Path,
    source_resolution: int,
    target_resolution: int,
    window_bp: int,
    value_dtype: str,
) -> None:
    target_bins = window_bp // target_resolution
    source_bins = (window_bp + source_resolution - 1) // source_resolution + 1
    bedpe = load_bedpe(bedpe_path, window_bp)
    output: dict[str, dict[str, object]] = {}
    weights_by_phase: dict[int, np.ndarray] = {}
    skipped_chromosome = skipped_bounds = duplicate_rows = 0
    started = time.time()

    for record in bedpe.itertuples(index=False):
        chromosome = record.chr1
        start_bp = int(record.start1)
        end_bp = int(record.end1)
        key = f"{chromosome}:{start_bp},{end_bp}"
        if key in output:
            duplicate_rows += 1
            continue
        source = genome.get(chromosome)
        if source is None:
            skipped_chromosome += 1
            continue
        source_start = start_bp // source_resolution
        if source_start < 0 or source_start + source_bins > source.shape[0]:
            skipped_bounds += 1
            continue

        phase = start_bp % source_resolution
        transform = weights_by_phase.get(phase)
        if transform is None:
            transform = rebin_weights(
                start_bp,
                source_start,
                source_bins,
                target_bins,
                source_resolution,
                target_resolution,
            )
            weights_by_phase[phase] = transform
        block = source[
            source_start : source_start + source_bins,
            source_start : source_start + source_bins,
        ].toarray()
        rebinned = transform @ block @ transform.T
        if record.strand1 == "-":
            rebinned = rebinned[::-1, ::-1].copy()
        row, col, data = upper_triangle_triplets(rebinned, value_dtype)
        output[key] = {
            "row": row,
            "col": col,
            "data": data,
            "is_empty": bool(data.size == 0),
            "storage": "triu_diag",
            "source_semantics": "raw_NONE",
            "total_count": np.float64(total_count),
        }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.tmp.{os.getpid()}")
    try:
        with temporary.open("wb") as handle:
            pickle.dump(output, handle, protocol=pickle.HIGHEST_PROTOCOL)
        temporary.replace(output_path)
    finally:
        if temporary.exists():
            temporary.unlink()
    print(
        f"Saved {len(output):,}/{len(bedpe):,} windows to {output_path} "
        f"(duplicate_rows={duplicate_rows:,}, skipped_chromosome={skipped_chromosome:,}, "
        f"skipped_bounds={skipped_bounds:,}, total_count={total_count:.12g}, "
        f"elapsed={time.time() - started:.1f}s)",
        flush=True,
    )


def split_specification(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("expected SPLIT=PATH")
    split, path = value.split("=", 1)
    if not split or not path:
        raise argparse.ArgumentTypeError("expected non-empty SPLIT=PATH")
    return split, Path(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--hic-pkl-path", type=Path)
    inputs.add_argument("--hic-pkl-dir", type=Path)
    parser.add_argument(
        "--bedpe",
        action="append",
        type=split_specification,
        required=True,
        metavar="SPLIT=PATH",
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--source-resolution", type=int, default=1000)
    parser.add_argument("--target-resolution", type=int, default=1024)
    parser.add_argument("--window-bp", type=int, default=196608)
    parser.add_argument("--value-dtype", choices=("float16", "float32"), default="float16")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.source_resolution <= 0 or args.target_resolution <= 0:
        raise ValueError("source and target resolutions must be positive")
    if args.window_bp <= 0 or args.window_bp % args.target_resolution:
        raise ValueError("window_bp must be positive and divisible by target_resolution")
    target_bins = args.window_bp // args.target_resolution
    if target_bins not in {192, 512}:
        raise ValueError(f"expected 192 or 512 target bins, found {target_bins}")
    if args.hic_pkl_path is not None:
        if not args.hic_pkl_path.is_file():
            raise FileNotFoundError(args.hic_pkl_path)
        input_paths = [args.hic_pkl_path]
    else:
        if not args.hic_pkl_dir.is_dir():
            raise NotADirectoryError(args.hic_pkl_dir)
        input_paths = sorted(path for path in args.hic_pkl_dir.glob("*.pkl") if path.is_file())
        if not input_paths:
            raise FileNotFoundError(f"No .pkl files in {args.hic_pkl_dir}")

    split_bedpes = dict(args.bedpe)
    if len(split_bedpes) != len(args.bedpe):
        raise ValueError("each split may be supplied only once")
    if set(split_bedpes) != {"train", "valid", "test"}:
        raise ValueError("exactly train, valid, and test BEDPE files are required")
    for path in split_bedpes.values():
        if not path.is_file():
            raise FileNotFoundError(path)

    for hic_pkl_path in input_paths:
        accession = hic_pkl_path.stem
        pending = [
            (split, path, args.output_root / split / f"{accession}.pkl")
            for split, path in split_bedpes.items()
            if args.force or not (args.output_root / split / f"{accession}.pkl").is_file()
        ]
        if not pending:
            print(f"All raw outputs already exist for {accession}", flush=True)
            continue

        genome, total_count = load_raw_genome(hic_pkl_path, args.source_resolution)
        for split, bedpe_path, output_path in pending:
            print(f"Extracting {accession}/{split}", flush=True)
            extract_split(
                genome=genome,
                total_count=total_count,
                bedpe_path=bedpe_path,
                output_path=output_path,
                source_resolution=args.source_resolution,
                target_resolution=args.target_resolution,
                window_bp=args.window_bp,
                value_dtype=args.value_dtype,
            )
        del genome
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
