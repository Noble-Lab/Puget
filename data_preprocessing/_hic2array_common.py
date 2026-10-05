# ---------------------------------------------------------------------------------
# The .hic-to-chromosome-array workflow is adapted from HiCFoundation_paper.
# Source: https://github.com/Noble-Lab/HiCFoundation_paper/blob/main/utils/hic2array.py
# License: Apache License 2.0
# ---------------------------------------------------------------------------------

from __future__ import annotations

import os
import pickle
import math
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix


def chromosome_name(name: str) -> str:
    return name if name.startswith("chr") else f"chr{name}"


def keep_chromosome(name: str) -> bool:
    lower = name.lower()
    return not ("all" in lower or "un" in lower or "random" in lower or "alt" in lower)


def convert_hic(
    input_hic: Path,
    output_pkl: Path,
    resolution: int,
    matrix_type: str,
    normalization: str,
    force: bool = False,
) -> None:
    if resolution <= 0:
        raise ValueError("resolution must be positive")
    if not input_hic.is_file():
        raise FileNotFoundError(input_hic)
    if output_pkl.exists() and not force:
        raise FileExistsError(f"{output_pkl} already exists; pass --force to replace it")

    import hicstraw  # Only needed when converting a .hic file.

    hic = hicstraw.HiCFile(str(input_hic))
    if resolution not in hic.getResolutions():
        raise ValueError(f"{resolution} bp is unavailable; choices: {hic.getResolutions()}")

    chromosomes: dict[str, coo_matrix] = {}
    lengths: dict[str, int] = {}
    total_count = np.float64(0.0)
    for chrom in hic.getChromosomes():
        if not keep_chromosome(chrom.name):
            continue
        name = chromosome_name(chrom.name)
        if name in chromosomes:
            raise ValueError(f"duplicate chromosome name {name}")
        print(f"Reading {name} ({matrix_type}/{normalization})", flush=True)
        if matrix_type == "observed":
            records = hicstraw.straw(
                "observed", "NONE", str(input_hic), chrom.name, chrom.name,
                "BP", resolution,
            )
        else:
            # Match the O/E SCALE chromosome extraction used for the paper.
            zoom = hic.getMatrixZoomData(
                chrom.name, chrom.name, "oe", "SCALE", "BP", resolution
            )
            records = zoom.getRecords(0, chrom.length, 0, chrom.length)
        if not records:
            print(f"  No contacts for {name}; skipping", flush=True)
            continue
        rows = np.fromiter((int(r.binX) // resolution for r in records), dtype=np.int32, count=len(records))
        cols = np.fromiter((int(r.binY) // resolution for r in records), dtype=np.int32, count=len(records))
        values = np.fromiter((float(r.counts) for r in records), dtype=np.float32, count=len(records))
        del records
        if not np.isfinite(values).all() or np.any(values < 0):
            raise ValueError(f"{name}: contacts must be finite and non-negative")
        if np.any(rows < 0) or np.any(cols < 0):
            raise ValueError(f"{name}: negative contact coordinates")
        n_bins = math.ceil(chrom.length / resolution)
        if np.any(rows >= n_bins) or np.any(cols >= n_bins):
            raise ValueError(f"{name}: contact coordinates exceed chromosome length")
        if matrix_type == "observed":
            total_count += values.sum(dtype=np.float64)
        upper_rows = np.minimum(rows, cols)
        upper_cols = np.maximum(rows, cols)
        stored_bins = int(upper_cols.max()) + 1
        matrix = coo_matrix(
            (values, (upper_rows, upper_cols)),
            shape=(stored_bins, stored_bins),
            dtype=np.float32,
        )
        if matrix_type == "oe":
            matrix.sum_duplicates()
        chromosomes[name] = matrix
        lengths[name] = int(chrom.length)
        print(f"  {matrix.shape[0]:,} bins; {matrix.nnz:,} entries", flush=True)

    if not chromosomes:
        raise ValueError(f"{input_hic}: no cis contacts found")
    metadata: dict[str, object] = {
        "format_version": 1,
        "accession": input_hic.stem,
        "source_hic": input_hic.name,
        "source_resolution_bp": int(resolution),
        "matrix_type": matrix_type,
        "normalization": normalization,
        "chromosome_lengths_bp": lengths,
    }
    if matrix_type == "observed":
        if not np.isfinite(total_count) or total_count <= 0:
            raise ValueError(f"{input_hic}: invalid raw total_count {total_count}")
        metadata["total_count"] = np.float64(total_count)

    output_pkl.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_pkl.with_name(f".{output_pkl.name}.tmp.{os.getpid()}")
    try:
        with temporary.open("wb") as handle:
            pickle.dump({"metadata": metadata, "chromosomes": chromosomes}, handle,
                        protocol=pickle.HIGHEST_PROTOCOL)
        temporary.replace(output_pkl)
    finally:
        if temporary.exists():
            temporary.unlink()
    print(f"Saved {output_pkl} ({len(chromosomes)} chromosomes)", flush=True)
