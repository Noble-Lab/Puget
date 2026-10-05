from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import pickle
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(variable, "4")

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
import torch

from scripts.ablate_puget_hic import (
    DEFAULT_CHECKPOINT, DEFAULT_PREDICTIONS, load_model, load_real_predictions,
)
from scripts.build_ig_baseline import (
    DEFAULT_CONFIG, N_BINS, WINDOW_BP, read_training_config, sha256, transformed_map, write_json,
)
from puget.biosamples import load_biosample_table

DEFAULT_BIOSAMPLE = "GM12878"
DEFAULT_LOOPS = ROOT / "data" / "annotations" / "loop" / "ENCFF661SAZ_GM12878_loop.bedpe"
DEFAULT_OUTPUT = ROOT / "outputs" / "borzoi_split_interpretation" / "loop_ko"
N_GENES = 1690
BIN_BP = 1024
PROMOTER_START = 254
PROMOTER_END = 258
BASE_SEED = 42
RESULT_COLUMNS = [
    "instance_id", "cell_index", "biosample", "gene_index", "gene_name",
    "annotation_index", "loop_class", "prediction_original",
    "prediction_loop_KO", "prediction_control_KO",
]


# --------------------------------------------------------------------------- #
# Inputs
# --------------------------------------------------------------------------- #
def read_test_genes(cfg) -> pd.DataFrame:
    frame = pd.read_csv(
        cfg.test_bedpe, sep="\t", header=None, usecols=[0, 1, 2, 3, 4, 5, 6, 8, 9],
        names=[
            "chrom", "window_start", "window_end", "chrom2", "start2", "end2",
            "gene_name", "strand", "strand2",
        ],
    )
    valid = (
        frame.chrom.eq(frame.chrom2)
        & frame.window_start.eq(frame.start2)
        & frame.window_end.eq(frame.end2)
        & frame.strand.eq(frame.strand2)
        & (frame.window_end - frame.window_start).eq(WINDOW_BP)
        & frame.strand.isin(["+", "-"])
    )
    if not valid.all() or len(frame) != N_GENES:
        raise ValueError("Unexpected Borzoi-split test BEDPE")
    frame = frame.drop(columns=["chrom2", "start2", "end2", "strand2"])
    frame.insert(0, "gene_index", np.arange(len(frame), dtype=np.int64))
    frame["tss"] = frame.window_start + WINDOW_BP // 2
    # Same key as the Hi-C window pickles.
    frame["window_key"] = (
        frame.chrom.astype(str) + ":" + frame.window_start.astype(str) + ","
        + frame.window_end.astype(str)
    )
    manifest = pd.read_csv(Path(cfg.test_bedpe).with_name("Borzoi_gene_manifest.csv"))
    test = manifest.loc[manifest.split.eq("test")].sort_values("bed_row_index")
    if len(test) != len(frame):
        raise ValueError("Manifest and BEDPE test-row counts differ")
    checks = (
        np.array_equal(frame.gene_name.astype(str), test.gene_name.astype(str)),
        np.array_equal(frame.chrom.astype(str), test.chrom.astype(str)),
        np.array_equal(frame.strand.astype(str), test.strand.astype(str)),
        np.array_equal(frame.tss.to_numpy(), test.TSS.astype(np.int64).to_numpy()),
        np.array_equal(frame.window_start.to_numpy(), test.gene_window_start.astype(np.int64).to_numpy()),
        np.array_equal(frame.window_end.to_numpy(), test.gene_window_end.astype(np.int64).to_numpy()),
    )
    if not all(checks):
        raise ValueError("Gene manifest does not match BEDPE order/geometry")
    return frame


def read_loops(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(
        path, sep="\t", comment="#", header=None, usecols=range(6),
        names=["chrom1", "x1", "x2", "chrom2", "y1", "y2"],
    )
    frame.insert(0, "annotation_index", np.arange(len(frame), dtype=np.int64))
    frame = frame[frame.chrom1.eq(frame.chrom2)].copy()
    swap = frame.x1 > frame.y1
    first = frame.loc[swap, ["x1", "x2"]].to_numpy(copy=True)
    frame.loc[swap, ["x1", "x2"]] = frame.loc[swap, ["y1", "y2"]].to_numpy()
    frame.loc[swap, ["y1", "y2"]] = first
    valid = (frame.x1 < frame.x2) & (frame.y1 < frame.y2) & (frame.x1 <= frame.y1)
    if not valid.all():
        raise ValueError(f"Invalid loop geometry: {path}")
    return frame.drop(columns="chrom2")


# --------------------------------------------------------------------------- #
# Pair geometry
# --------------------------------------------------------------------------- #
def genomic_span_to_bins(start: int, end: int, window_start: int) -> tuple[int, int]:
    b0 = (int(start) - int(window_start)) // BIN_BP
    b1 = -(-(int(end) - int(window_start)) // BIN_BP)
    return b0, b1


def midpoint_bin(start: int, end: int, window_start: int) -> int:
    return ((int(start) + int(end)) - 2 * int(window_start)) // (2 * BIN_BP)


def orient_span(b0: int, b1: int, strand: str) -> tuple[int, int]:
    if strand == "+":
        return int(b0), int(b1)
    if strand == "-":
        return N_BINS - int(b1), N_BINS - int(b0)
    raise ValueError(f"Unexpected strand: {strand}")


def orient_bin(value: int, strand: str) -> int:
    if strand == "+":
        return int(value)
    if strand == "-":
        return N_BINS - 1 - int(value)
    raise ValueError(f"Unexpected strand: {strand}")


def stable_seed(base_seed: int, *parts: object) -> int:
    payload = ":".join([str(int(base_seed)), *(str(part) for part in parts)]).encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, "little") & 0x7FFFFFFF


def interval_overlap(a0, a1, b0, b1):
    return (a0 < b1) & (b0 < a1)


def rectangles_hit_any_loop(r0, r1, c0, c1, loop_spans: np.ndarray) -> np.ndarray:
    r0 = np.atleast_1d(r0).astype(np.int64)[:, None]
    r1 = np.atleast_1d(r1).astype(np.int64)[:, None]
    c0 = np.atleast_1d(c0).astype(np.int64)[:, None]
    c1 = np.atleast_1d(c1).astype(np.int64)[:, None]
    lr0, lr1, lc0, lc1 = (loop_spans[:, index][None, :] for index in range(4))
    direct = interval_overlap(r0, r1, lr0, lr1) & interval_overlap(c0, c1, lc0, lc1)
    crossed = interval_overlap(r0, r1, lc0, lc1) & interval_overlap(c0, c1, lr0, lr1)
    return (direct | crossed).any(axis=1)


def is_promoter_bin(value: int) -> bool:
    return PROMOTER_START <= int(value) < PROMOTER_END


def loop_spans_in_window(overlapping: pd.DataFrame, gene) -> np.ndarray:
    spans = []
    for loop in overlapping.itertuples(index=False):
        r0, r1 = genomic_span_to_bins(loop.x1, loop.x2, gene.window_start)
        c0, c1 = genomic_span_to_bins(loop.y1, loop.y2, gene.window_start)
        r0, r1 = max(0, r0), min(N_BINS, r1)
        c0, c1 = max(0, c0), min(N_BINS, c1)
        if r0 >= r1 or c0 >= c1:
            continue
        r0, r1 = orient_span(r0, r1, gene.strand)
        c0, c1 = orient_span(c0, c1, gene.strand)
        spans.append((r0, r1, c0, c1))
    return np.asarray(spans, dtype=np.int64) if spans else np.empty((0, 4), dtype=np.int64)


def promoter_mirror_control(r0, r1, c0, c1, r_promoter, c_promoter, loop_spans):
    if r_promoter and c_promoter:
        return None, "both_anchor_centers_promoter"
    if r_promoter:
        cr0, cr1 = r0, r1
        cc0, cc1 = N_BINS - c1, N_BINS - c0
        promoter_anchor = "anchor1"
    elif c_promoter:
        cr0, cr1 = N_BINS - r1, N_BINS - r0
        cc0, cc1 = c0, c1
        promoter_anchor = "anchor2"
    else:
        raise AssertionError("Promoter control requested for a non-promoter loop")
    if not (0 <= cr0 < cr1 <= N_BINS and 0 <= cc0 < cc1 <= N_BINS):
        return None, "mirrored_control_outside_window"
    if rectangles_hit_any_loop(cr0, cr1, cc0, cc1, loop_spans)[0]:
        return None, "mirrored_control_overlaps_loop"
    return {
        "control_kind": "promoter_tss_mirror",
        "promoter_anchor": promoter_anchor,
        "control_r0": cr0,
        "control_r1": cr1,
        "control_c0": cc0,
        "control_c1": cc1,
        "control_shift_bins": np.nan,
    }, None


def same_diagonal_control(r0, r1, c0, c1, r_mid, c_mid, loop_spans, seed):
    low = max(-r0, -c0)
    high = min(N_BINS - r1, N_BINS - c1)
    shifts = np.arange(low, high + 1, dtype=np.int64)
    shifts = shifts[shifts != 0]
    shifts = shifts[
        ~(
            ((PROMOTER_START <= r_mid + shifts) & (r_mid + shifts < PROMOTER_END))
            | ((PROMOTER_START <= c_mid + shifts) & (c_mid + shifts < PROMOTER_END))
        )
    ]
    if shifts.size == 0:
        return None, "no_in_window_same_diagonal_control"
    hit = rectangles_hit_any_loop(r0 + shifts, r1 + shifts, c0 + shifts, c1 + shifts, loop_spans)
    shifts = shifts[~hit]
    if shifts.size == 0:
        return None, "no_loop_free_same_diagonal_control"
    rng = np.random.default_rng(seed)
    shift = int(shifts[int(rng.integers(0, len(shifts)))])
    return {
        "control_kind": "same_diagonal_shift",
        "promoter_anchor": "none",
        "control_r0": r0 + shift,
        "control_r1": r1 + shift,
        "control_c0": c0 + shift,
        "control_c1": c1 + shift,
        "control_shift_bins": shift,
    }, None


def build_pairs(cell_index: int, biosample: str, genes: pd.DataFrame, loops: pd.DataFrame,
                base_seed: int) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    by_chrom = {chrom: part for chrom, part in loops.groupby("chrom1", sort=False)}
    kept, dropped = [], []
    n_full = 0
    for gene in genes.itertuples(index=False):
        chrom_loops = by_chrom.get(gene.chrom)
        if chrom_loops is None:
            continue
        ws, we = int(gene.window_start), int(gene.window_end)
        overlapping = chrom_loops[
            (chrom_loops.x2 > ws) & (chrom_loops.x1 < we)
            & (chrom_loops.y2 > ws) & (chrom_loops.y1 < we)
        ]
        full = overlapping[
            (overlapping.x1 >= ws) & (overlapping.x2 <= we)
            & (overlapping.y1 >= ws) & (overlapping.y2 <= we)
        ]
        if full.empty:
            continue
        all_loop_spans = loop_spans_in_window(overlapping, gene)
        n_full += len(full)
        for loop in full.itertuples(index=False):
            r0, r1 = orient_span(*genomic_span_to_bins(loop.x1, loop.x2, ws), gene.strand)
            c0, c1 = orient_span(*genomic_span_to_bins(loop.y1, loop.y2, ws), gene.strand)
            r_mid = orient_bin(midpoint_bin(loop.x1, loop.x2, ws), gene.strand)
            c_mid = orient_bin(midpoint_bin(loop.y1, loop.y2, ws), gene.strand)
            if not (
                0 <= r0 < r1 <= N_BINS and 0 <= c0 < c1 <= N_BINS
                and r0 <= r_mid < r1 and c0 <= c_mid < c1
            ):
                raise AssertionError("Fully contained loop mapped outside its model window")

            r_promoter, c_promoter = is_promoter_bin(r_mid), is_promoter_bin(c_mid)
            loop_class = "promoter_anchored" if r_promoter or c_promoter else "non_promoter"
            # Biological keys make the draw invariant to gene-table reordering.
            seed = stable_seed(
                base_seed, biosample, gene.window_key, gene.gene_name, gene.strand,
                int(loop.annotation_index),
            )
            if loop_class == "promoter_anchored":
                control, reason = promoter_mirror_control(
                    r0, r1, c0, c1, r_promoter, c_promoter, all_loop_spans
                )
            else:
                control, reason = same_diagonal_control(
                    r0, r1, c0, c1, r_mid, c_mid, all_loop_spans, seed
                )

            common = {
                "instance_id": (
                    f"{biosample}|{gene.window_key}|{gene.gene_name}|{gene.strand}|"
                    f"{int(loop.annotation_index)}"
                ),
                "cell_index": cell_index,
                "biosample": biosample,
                "gene_index": int(gene.gene_index),
                "gene_name": gene.gene_name,
                "chrom": gene.chrom,
                "window_start": ws,
                "window_end": we,
                "window_key": gene.window_key,
                "strand": gene.strand,
                "annotation_index": int(loop.annotation_index),
                "anchor1_start": int(loop.x1),
                "anchor1_end": int(loop.x2),
                "anchor2_start": int(loop.y1),
                "anchor2_end": int(loop.y2),
                "loop_class": loop_class,
                "loop_r0": r0,
                "loop_r1": r1,
                "loop_c0": c0,
                "loop_c1": c1,
                "anchor1_mid_bin": r_mid,
                "anchor2_mid_bin": c_mid,
                "loop_center_distance_bins": abs(c_mid - r_mid),
                "loop_center_distance_bp": abs(
                    (int(loop.y1) + int(loop.y2)) - (int(loop.x1) + int(loop.x2))
                ) // 2,
            }
            if control is None:
                dropped.append({**common, "drop_reason": reason})
            else:
                row = {**common, **control, "control_seed": seed}
                if (r1 - r0, c1 - c0) != (
                    row["control_r1"] - row["control_r0"], row["control_c1"] - row["control_c0"]
                ):
                    raise AssertionError("Control changed anchor dimensions")
                kept.append(row)

    kept_frame, dropped_frame = pd.DataFrame(kept), pd.DataFrame(dropped)
    if kept_frame.empty or kept_frame.instance_id.duplicated().any():
        raise ValueError(f"Invalid paired-instance table for {biosample}")
    counts = {
        "source_loop_calls": len(loops),
        "fully_contained_loop_instances": int(n_full),
        "valid_paired_instances": len(kept_frame),
        "dropped_no_valid_control": len(dropped_frame),
        "promoter_anchored_valid": int(kept_frame.loop_class.eq("promoter_anchored").sum()),
        "non_promoter_valid": int(kept_frame.loop_class.eq("non_promoter").sum()),
        "gene_rows_with_valid_pairs": int(kept_frame.gene_index.nunique()),
        "drop_reasons": {
            str(reason): int(n)
            for reason, n in dropped_frame.get("drop_reason", pd.Series(dtype=str)).value_counts().items()
        },
    }
    return kept_frame, dropped_frame, counts


# --------------------------------------------------------------------------- #
# Knockout inference
# --------------------------------------------------------------------------- #
def mask_rectangle(matrix: np.ndarray, r0: int, r1: int, c0: int, c1: int) -> None:
    matrix[int(r0):int(r1), int(c0):int(c1)] = 0.0
    matrix[int(c0):int(c1), int(r0):int(r1)] = 0.0


def usable_map(entry) -> np.ndarray:
    image, status = transformed_map(entry)
    if image is None:
        raise ValueError(f"{status} Hi-C entry for an annotated loop instance")
    return image


def configure_determinism(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_float32_matmul_precision("highest")


@torch.inference_mode()
def predict_batch(model, device, images, sequences, use_amp, amp_dtype) -> np.ndarray:
    image = torch.from_numpy(np.stack(images)).unsqueeze(1).to(device)
    sequence = torch.from_numpy(np.stack(sequences)).to(device)
    with torch.autocast(device.type, dtype=amp_dtype, enabled=use_amp):
        output = model(image, sequence)
    result = output.float().cpu().numpy().reshape(-1)
    if not np.isfinite(result).all():
        raise ValueError("Model emitted a nonfinite prediction")
    return result


def run_knockouts(pairs: pd.DataFrame, windows: dict, genes: pd.DataFrame, sequence: np.ndarray,
                  original: np.ndarray, model, cfg, device, batch_size: int,
                  verify_original: int) -> tuple[np.ndarray, np.ndarray, float | None]:
    use_amp = device.type == "cuda" and "16" in str(cfg.precision)
    amp_dtype = torch.bfloat16 if "bf16" in str(cfg.precision) else torch.float16

    # Unperturbed predictions for the first genes must reproduce infer_puget.py.
    verify_genes = pairs.gene_index.drop_duplicates().iloc[:verify_original]
    max_abs = None
    if len(verify_genes):
        observed = predict_batch(
            model, device,
            [usable_map(windows.get(genes.window_key.iloc[int(g)])) for g in verify_genes],
            [np.asarray(sequence[int(g)], dtype=np.float16) for g in verify_genes],
            use_amp, amp_dtype,
        )
        expected = original[verify_genes.to_numpy(dtype=np.int64)]
        max_abs = float(np.max(np.abs(observed - expected)))
        if not np.allclose(observed, expected, rtol=0.01, atol=0.03):
            raise ValueError(f"Real-prediction check failed: max_abs_diff={max_abs}")

    loop_predictions = np.full(len(pairs), np.nan, dtype=np.float32)
    control_predictions = np.full(len(pairs), np.nan, dtype=np.float32)
    pending_images, pending_sequences, pending_targets = [], [], []

    def flush() -> None:
        if not pending_images:
            return
        values = predict_batch(model, device, pending_images, pending_sequences, use_amp, amp_dtype)
        for value, (row_index, kind) in zip(values, pending_targets):
            (loop_predictions if kind == "loop" else control_predictions)[row_index] = value
        pending_images.clear()
        pending_sequences.clear()
        pending_targets.clear()

    started = time.time()
    completed = 0
    for gene_index, group in pairs.groupby("gene_index", sort=False):
        base = usable_map(windows.get(genes.window_key.iloc[int(gene_index)]))
        seq = np.asarray(sequence[int(gene_index)], dtype=np.float16)
        for row in group.itertuples(index=True):
            loop_ko, control_ko = base.copy(), base.copy()
            mask_rectangle(loop_ko, row.loop_r0, row.loop_r1, row.loop_c0, row.loop_c1)
            mask_rectangle(control_ko, row.control_r0, row.control_r1, row.control_c0, row.control_c1)
            if not np.array_equal(loop_ko, loop_ko.T) or not np.array_equal(control_ko, control_ko.T):
                raise AssertionError("Zero KO broke matrix symmetry")
            pending_images.extend([loop_ko, control_ko])
            pending_sequences.extend([seq, seq])
            pending_targets.extend([(int(row.Index), "loop"), (int(row.Index), "control")])
            if len(pending_images) >= batch_size:
                flush()
            completed += 1
            if completed % 1_000 == 0:
                elapsed = time.time() - started
                eta = elapsed / completed * (len(pairs) - completed)
                print(f"{completed:,}/{len(pairs):,} pairs; ETA {eta / 60:.1f} min", flush=True)
    flush()
    if not np.isfinite(loop_predictions).all() or not np.isfinite(control_predictions).all():
        raise ValueError("Incomplete KO predictions")
    return loop_predictions, control_predictions, max_abs


# --------------------------------------------------------------------------- #
# Paired tests
# --------------------------------------------------------------------------- #
def holm_adjust(p_values: list[float]) -> np.ndarray:
    values = np.asarray(p_values, dtype=float)
    order = np.argsort(values)
    adjusted = np.empty_like(values)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (len(values) - rank) * values[index])
        adjusted[index] = min(1.0, running)
    return adjusted


def significance_label(p_value: float) -> str:
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return "ns"


def summarize(result: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    result = result.copy()
    result["loop_delta"] = result.prediction_loop_KO - result.prediction_original
    result["control_delta"] = result.prediction_control_KO - result.prediction_original
    summary_rows, test_rows = [], []
    for loop_class in ("promoter_anchored", "non_promoter"):
        part = result.loc[result.loop_class.eq(loop_class)]
        for column, condition in (("loop_delta", "loop_KO"), ("control_delta", "control_KO")):
            values = part[column].to_numpy()
            summary_rows.append({
                "loop_class": loop_class,
                "condition": condition,
                "n": len(values),
                "mean_signed_delta": float(values.mean()),
                "median_signed_delta": float(np.median(values)),
                "sd_signed_delta": float(values.std(ddof=1)),
            })
        loop_values = part.loop_delta.to_numpy()
        control_values = part.control_delta.to_numpy()
        test = wilcoxon(loop_values, control_values, alternative="less",
                        zero_method="wilcox", method="approx")
        differences = loop_values - control_values
        test_rows.append({
            "loop_class": loop_class,
            "comparison": "loop_KO vs matched control_KO",
            "test": "paired one-sided Wilcoxon signed-rank: loop_KO < control_KO",
            "n_pairs": len(part),
            "n_nonzero_differences": int(np.count_nonzero(differences)),
            "wilcoxon_statistic": float(test.statistic),
            "z_statistic": float(getattr(test, "zstatistic", np.nan)),
            "p_value": float(test.pvalue),
            "mean_paired_difference": float(differences.mean()),
            "median_paired_difference": float(np.median(differences)),
        })
    summary = pd.DataFrame(summary_rows)
    tests = pd.DataFrame(test_rows)
    tests["p_value_holm"] = holm_adjust(tests.p_value.tolist())
    tests["significance"] = tests.p_value_holm.map(significance_label)
    return summary, tests


# --------------------------------------------------------------------------- #
def atomic_table(frame: pd.DataFrame, path: Path, compression=None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    frame.to_csv(temporary, sep="\t", index=False, compression=compression)
    os.replace(temporary, path)


def load_or_build_pairs(args, cfg, cell_index: int, genes: pd.DataFrame) -> pd.DataFrame:
    pairs_path = args.output / "pairs.tsv.gz"
    meta_path = args.output / "pairs_meta.json"
    provenance = {
        "biosample": args.biosample,
        "seed": args.seed,
        "loops": str(args.loops),
        "loops_sha256": sha256(args.loops),
        "test_bedpe_sha256": sha256(cfg.test_bedpe),
    }
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text())
        if meta.get("provenance") != provenance:
            raise RuntimeError(f"Existing pairs were built from different inputs: {meta_path}")
        print(f"Reusing {pairs_path}", flush=True)
        return pd.read_csv(pairs_path, sep="\t")
    loops = read_loops(args.loops)
    pairs, dropped, counts = build_pairs(cell_index, args.biosample, genes, loops, args.seed)
    atomic_table(pairs, pairs_path, compression="gzip")
    atomic_table(dropped, args.output / "dropped_pairs.tsv.gz", compression="gzip")
    write_json(meta_path, {
        "completed": True,
        "provenance": provenance,
        **counts,
        "bin_bp": BIN_BP,
        "promoter_center_bins": list(range(PROMOTER_START, PROMOTER_END)),
        "classification": "promoter-anchored iff either anchor midpoint is bin 254..257",
        "promoter_control": "keep promoter anchor fixed and TSS-mirror distal span",
        "non_promoter_control": "uniform seeded draw among valid same-diagonal common shifts",
        "control_exclusion": "paired control must not intersect any annotated loop in the window",
        "negative_strand": "orient spans and midpoints before classification/control selection",
        "seed_key": "seed,biosample,chrom:start,end,gene_name,strand,loop_annotation_index",
    })
    print(
        f"{args.biosample}: kept {counts['valid_paired_instances']:,}/"
        f"{counts['fully_contained_loop_instances']:,} loops "
        f"({counts['promoter_anchored_valid']:,} promoter-anchored, "
        f"{counts['non_promoter_valid']:,} non-promoter); "
        f"dropped {counts['dropped_no_valid_control']:,}",
        flush=True,
    )
    return pd.read_csv(pairs_path, sep="\t")


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--biosample", default=DEFAULT_BIOSAMPLE)
    parser.add_argument("--loops", type=Path, default=DEFAULT_LOOPS,
                        help="ENCODE loop BEDPE for --biosample")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG,
                        help="Puget training YAML in configs/training")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--predictions-dir", type=Path, default=DEFAULT_PREDICTIONS,
                        help="directory with the real-Hi-C arrays from scripts/infer_puget.py")
    parser.add_argument("--output", type=Path, default=None,
                        help="default: outputs/borzoi_split_interpretation/loop_ko/<biosample>")
    parser.add_argument("--gpu", type=int, default=None,
                        help="physical GPU index for this job (sets CUDA_VISIBLE_DEVICES)")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--batch-size", type=int, default=16, help="KO maps per forward pass")
    parser.add_argument("--verify-original", type=int, default=4,
                        help="genes whose unperturbed prediction is checked first")
    parser.add_argument("--seed", type=int, default=BASE_SEED)
    parser.add_argument("--pairs-only", action="store_true",
                        help="draw loop/control pairs and stop before inference")
    args = parser.parse_args(argv)
    if args.batch_size < 2 or args.verify_original < 0:
        parser.error("Need --batch-size >= 2 and --verify-original >= 0")
    if args.gpu is not None:
        if args.gpu < 0:
            parser.error("--gpu must be a non-negative GPU index")
        # CUDA is not initialized before this point, so this selects the device.
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    if args.output is None:
        args.output = DEFAULT_OUTPUT / args.biosample.replace(" ", "_")
    for name in ("loops", "config", "checkpoint", "predictions_dir", "output"):
        setattr(args, name, getattr(args, name).resolve())
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    if not args.loops.is_file():
        raise FileNotFoundError(args.loops)
    cfg = read_training_config(args.config)
    table = load_biosample_table(cfg.biosamples_csv)
    names = [name for _row, _accession, name in table]
    if args.biosample not in names:
        raise ValueError(f"Unknown biosample {args.biosample!r}; expected one of {names}")
    cell_index, accession, _name = table[names.index(args.biosample)]
    genes = read_test_genes(cfg)
    pairs = load_or_build_pairs(args, cfg, cell_index, genes)
    if args.pairs_only:
        return 0

    result_path = args.output / "predictions.tsv.gz"
    if (args.output / "meta.json").is_file():
        print(f"Knockout results already complete: {result_path}")
        return 0
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    original = load_real_predictions(args, cfg, names)[cell_index]
    sequence = np.load(cfg.test_seq, mmap_mode="r")
    if sequence.shape != (N_GENES, N_BINS, int(cfg.embed_dim)):
        raise ValueError(f"Unexpected test sequence shape: {sequence.shape}")
    hic_path = Path(cfg.hic_root) / "test" / f"{accession}.pkl"
    if not hic_path.is_file():
        raise FileNotFoundError(hic_path)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    device = torch.device(args.device)
    configure_determinism(args.seed)
    model = load_model(args, cfg, device)
    print(f"Loading {hic_path}", flush=True)
    with hic_path.open("rb") as handle:
        windows = pickle.load(handle)
    started = time.time()
    loop_predictions, control_predictions, max_abs = run_knockouts(
        pairs, windows, genes, sequence, original, model, cfg, device,
        args.batch_size, args.verify_original,
    )
    del windows
    gc.collect()

    result = pairs[["instance_id", "cell_index", "biosample", "gene_index", "gene_name",
                    "annotation_index", "loop_class"]].copy()
    result["prediction_original"] = original[result.gene_index.to_numpy()]
    result["prediction_loop_KO"] = loop_predictions
    result["prediction_control_KO"] = control_predictions
    if list(result.columns) != RESULT_COLUMNS:
        raise AssertionError("Unexpected output columns")
    atomic_table(result, result_path, compression="gzip")
    group_summary, paired_tests = summarize(result)
    atomic_table(group_summary, args.output / "group_summary.tsv")
    atomic_table(paired_tests, args.output / "paired_tests.tsv")
    write_json(args.output / "meta.json", {
        "completed": True,
        "description": "Paired loop/control zero-KO predictions for Borzoi-split test genes",
        "biosample": args.biosample,
        "cell_index": int(cell_index),
        "accession": accession,
        "n_instances": len(result),
        "n_promoter_anchored": int(result.loop_class.eq("promoter_anchored").sum()),
        "n_non_promoter": int(result.loop_class.eq("non_promoter").sum()),
        "effect_direction": "KO prediction minus original prediction",
        "ko_value": 0.0,
        "input_space": "log10(genome-wide SCALE O/E + 1)",
        "paired_test": "one-sided Wilcoxon signed-rank, loop_KO < matched control_KO",
        "multiple_testing": "Holm correction across two loop classes",
        "pairs_sha256": sha256(args.output / "pairs.tsv.gz"),
        "hic_path": str(hic_path),
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": sha256(args.checkpoint),
        "config": str(args.config),
        "real_predictions_dir": str(args.predictions_dir),
        "verified_original_predictions": int(min(args.verify_original, pairs.gene_index.nunique())),
        "verified_original_max_abs_diff": max_abs,
        "batch_size": args.batch_size,
        "device": str(device),
        "deterministic_algorithms": True,
        "float32_matmul_precision": "highest",
        "seed": args.seed,
        "elapsed_sec": time.time() - started,
    })
    print(paired_tests[["loop_class", "n_pairs", "median_paired_difference",
                        "p_value_holm", "significance"]].to_string(index=False))
    print(f"Saved {result_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
