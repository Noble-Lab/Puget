# ---------------------------------------------------------------------------------
# The genome-to-submatrix workflow builds on the HiCFoundation paper utility.
# Source: https://github.com/Noble-Lab/HiCFoundation_paper/blob/main/utils/scan_array_diag.py
# License: Apache License 2.0
# This version selects BEDPE gene windows and area-rebins O/E SCALE contacts.
# ---------------------------------------------------------------------------------

from __future__ import annotations

import argparse
import os
import pickle
import time
from pathlib import Path

# Set limits before importing NumPy/SciPy, which load their thread pools.
_threads = os.environ.get("PUGET_HIC_THREADS", "1")
if not _threads.isdecimal() or int(_threads) < 1:
    raise ValueError("PUGET_HIC_THREADS must be a positive integer")
for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = _threads

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix


SPLIT_COLUMNS = ("chr1", "start1", "end1", "chr2", "start2", "end2", "name", "score", "strand1", "strand2")


def chrom_name(value: object) -> str:
    value = str(value)
    return value if value.startswith("chr") else f"chr{value}"


def symmetric_csr(matrix: coo_matrix) -> coo_matrix:
    matrix = matrix.tocoo(copy=True)
    matrix.sum_duplicates()
    if np.any(matrix.row > matrix.col):
        raise ValueError("expected an upper-triangular source matrix")
    if not np.isfinite(matrix.data).all() or np.any(matrix.data < 0):
        raise ValueError("source O/E values must be finite and non-negative")
    off_diagonal = matrix.row != matrix.col
    result = coo_matrix(
        (
            np.concatenate((matrix.data, matrix.data[off_diagonal])).astype(np.float32, copy=False),
            (
                np.concatenate((matrix.row, matrix.col[off_diagonal])),
                np.concatenate((matrix.col, matrix.row[off_diagonal])),
            ),
        ),
        shape=matrix.shape,
    )
    result.sum_duplicates()
    return result.tocsr()


def load_genome(path: Path, source_resolution: int) -> dict[str, object]:
    print(f"Loading O/E source: {path}", flush=True)
    with path.open("rb") as handle:
        raw = pickle.load(handle)
    if not isinstance(raw, dict) or not raw:
        raise TypeError(f"{path}: expected a non-empty chromosome dictionary")
    if "chromosomes" in raw:
        metadata = raw.get("metadata")
        if not isinstance(metadata, dict):
            raise TypeError(f"{path}: missing genome metadata")
        if (metadata.get("matrix_type"), metadata.get("normalization")) != ("oe", "SCALE"):
            raise ValueError(f"{path}: expected O/E SCALE source")
        if int(metadata.get("source_resolution_bp", -1)) != source_resolution:
            raise ValueError(f"{path}: source resolution does not match --source-resolution")
        raw = raw["chromosomes"]
    if not isinstance(raw, dict) or not raw:
        raise TypeError(f"{path}: missing chromosome matrices")
    genome: dict[str, object] = {}
    for source_chrom in list(raw):
        matrix = raw.pop(source_chrom)
        if not isinstance(matrix, coo_matrix):
            raise TypeError(f"{path}:{source_chrom}: expected scipy.sparse.coo_matrix")
        normalized = chrom_name(source_chrom)
        if normalized in genome:
            raise ValueError(f"{path}: duplicate chromosome {normalized}")
        genome[normalized] = symmetric_csr(matrix)
    return genome


def load_bedpe(path: Path, window_bp: int) -> pd.DataFrame:
    frame = pd.read_csv(path, sep="\t", header=None, names=SPLIT_COLUMNS)
    if frame.empty:
        raise ValueError(f"{path}: BEDPE is empty")
    for column in ("start1", "end1", "start2", "end2"):
        frame[column] = pd.to_numeric(frame[column], errors="raise").astype("int64")
    frame["chr1"] = frame["chr1"].map(chrom_name)
    frame["chr2"] = frame["chr2"].map(chrom_name)
    square = (frame.chr1 == frame.chr2) & (frame.start1 == frame.start2) & (frame.end1 == frame.end2)
    if not bool(square.all()):
        raise ValueError(f"{path}: every row must be a cis square window")
    if not bool(((frame.end1 - frame.start1) == window_bp).all()):
        raise ValueError(f"{path}: every row must span exactly {window_bp} bp")
    if not frame.strand1.isin(("+", "-")).all():
        raise ValueError(f"{path}: strand1 must be '+' or '-'")
    return frame


def weights(start_bp: int, source_start_bin: int, source_bins: int, target_bins: int, source_resolution: int, target_resolution: int) -> np.ndarray:
    target_start = start_bp + np.arange(target_bins, dtype=np.int64)[:, None] * target_resolution
    target_end = target_start + target_resolution
    source_start = (source_start_bin + np.arange(source_bins, dtype=np.int64)[None, :]) * source_resolution
    source_end = source_start + source_resolution
    overlap = np.maximum(0, np.minimum(target_end, source_end) - np.maximum(target_start, source_start))
    result = overlap.astype(np.float32) / target_resolution
    if not np.allclose(result.sum(axis=1), 1.0, atol=1e-6):
        raise ValueError("source bins do not cover the target window")
    return result


def upper_triplets(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    matrix = ((matrix + matrix.T) * 0.5).astype(np.float32, copy=False)
    if not np.isfinite(matrix).all() or np.any(matrix < 0):
        raise ValueError("rebinned O/E matrix contains invalid values")
    row, col = np.triu_indices(matrix.shape[0])
    data = matrix[row, col]
    keep = data > 0
    row, col, data = row[keep], col[keep], data[keep].astype(np.float16)
    if data.size and not np.isfinite(data).all():
        raise OverflowError("rebinned O/E value exceeds float16 range")
    keep = data > 0  # discard values that underflowed during float16 conversion
    return row[keep].astype(np.int16), col[keep].astype(np.int16), data[keep]


def extract_split(genome: dict[str, object], bedpe_path: Path, output_path: Path, source_resolution: int, target_resolution: int, window_bp: int) -> None:
    target_bins = window_bp // target_resolution
    source_bins = (window_bp + source_resolution - 1) // source_resolution + 1
    bedpe = load_bedpe(bedpe_path, window_bp)
    output: dict[str, dict[str, object]] = {}
    cache: dict[int, np.ndarray] = {}
    skipped_chrom = skipped_bounds = duplicates = 0
    started = time.time()

    for record in bedpe.itertuples(index=False):
        chrom, start, end = record.chr1, int(record.start1), int(record.end1)
        key = f"{chrom}:{start},{end}"
        if key in output:
            duplicates += 1
            continue
        matrix = genome.get(chrom)
        if matrix is None:
            skipped_chrom += 1
            continue
        source_start = start // source_resolution
        if source_start < 0 or source_start + source_bins > matrix.shape[0]:
            skipped_bounds += 1
            continue
        phase = start % source_resolution
        transform = cache.get(phase)
        if transform is None:
            transform = weights(start, source_start, source_bins, target_bins, source_resolution, target_resolution)
            cache[phase] = transform
        source = matrix[source_start:source_start + source_bins, source_start:source_start + source_bins].toarray()
        rebinned = transform @ source @ transform.T
        if record.strand1 == "-":
            rebinned = rebinned[::-1, ::-1]
        row, col, data = upper_triplets(rebinned)
        output[key] = {
            "row": row,
            "col": col,
            "data": data,
            "is_empty": bool(data.size == 0),
            "storage": "triu_diag",
            "source_semantics": "oe_SCALE",
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
    print(f"Saved {len(output):,}/{len(bedpe):,} windows to {output_path} "
          f"(duplicates={duplicates:,}, skipped_chrom={skipped_chrom:,}, "
          f"skipped_bounds={skipped_bounds:,}, elapsed={time.time() - started:.1f}s)", flush=True)


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
    parser.add_argument("--bedpe", action="append", type=split_specification, required=True, metavar="SPLIT=PATH")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--source-resolution", type=int, default=1000)
    parser.add_argument("--target-resolution", type=int, default=1024)
    parser.add_argument("--window-bp", type=int, required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.source_resolution <= 0 or args.target_resolution <= 0:
        raise ValueError("source and target resolutions must be positive")
    if args.window_bp <= 0 or args.window_bp % args.target_resolution:
        raise ValueError("window_bp must be positive and divisible by target_resolution")
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
    if any(not path.is_file() for path in split_bedpes.values()):
        raise FileNotFoundError("one or more BEDPE paths do not exist")
    for hic_pkl_path in input_paths:
        accession = hic_pkl_path.stem
        pending = [(split, path, args.output_root / split / f"{accession}.pkl") for split, path in split_bedpes.items()
                   if args.force or not (args.output_root / split / f"{accession}.pkl").exists()]
        if not pending:
            print(f"All outputs already exist for {accession}", flush=True)
            continue
        genome = load_genome(hic_pkl_path, args.source_resolution)
        for split, bedpe_path, output_path in pending:
            extract_split(genome, bedpe_path, output_path, args.source_resolution, args.target_resolution, args.window_bp)
        del genome
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
