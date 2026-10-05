from __future__ import annotations

import argparse
import csv
from collections import defaultdict
import gzip
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from sklearn.metrics import roc_auc_score

from scripts.attribute_puget_ig import DEFAULT_OUTPUT as DEFAULT_IG_OUTPUT
from scripts.build_ig_baseline import (
    DEFAULT_CONFIG, N_BINS, WINDOW_BP, read_json, read_training_config, sha256, write_json,
)
from puget.biosamples import load_biosample_table

DEFAULT_BIOSAMPLE = "GM12878"
DEFAULT_TADS = ROOT / "data" / "annotations" / "tad" / "ENCFF788UTU_GM12878_TAD.bedpe"
DEFAULT_IG_RUN = DEFAULT_IG_OUTPUT / "paper-train14-distance-mean_steps100"
DEFAULT_OUTPUT = ROOT / "outputs" / "borzoi_split_interpretation" / "tad_attribution"
N_GENES = 1690
BIN_BP = 1024
PROMOTER_START = 254
PROMOTER_END = 258
PROMOTER_ROWS = (254, 255, 256, 257)
N_DISTANCES = 256
SEED = 42


def atomic_save_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def write_tsv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def read_genes(cfg) -> list[dict]:
    genes = []
    with Path(cfg.test_bedpe).open(newline="") as handle:
        for gene_index, fields in enumerate(csv.reader(handle, delimiter="\t")):
            if len(fields) < 10:
                raise ValueError(f"BEDPE row {gene_index + 1} has fewer than 10 columns")
            chrom, start, end, chrom2, start2, end2, gene_name, _, strand, strand2 = fields[:10]
            start, end, start2, end2 = map(int, (start, end, start2, end2))
            if not (
                chrom == chrom2 and start == start2 and end == end2
                and strand == strand2 and strand in {"+", "-"}
                and end - start == WINDOW_BP
            ):
                raise ValueError(f"Unexpected BEDPE geometry at gene row {gene_index}")
            genes.append({
                "gene_index": gene_index,
                "gene_name": gene_name,
                "chrom": chrom,
                "window_start": start,
                "window_end": end,
                "strand": strand,
                "tss": start + WINDOW_BP // 2,
            })
    if len(genes) != N_GENES:
        raise ValueError(f"Expected {N_GENES} Borzoi-split test genes")

    manifest = pd.read_csv(Path(cfg.test_bedpe).with_name("Borzoi_gene_manifest.csv"))
    test = manifest.loc[manifest["split"].eq("test")].sort_values("bed_row_index")
    if len(test) != len(genes):
        raise ValueError("Gene manifest and BEDPE test rows differ")
    for gene, row in zip(genes, test.itertuples(index=False)):
        observed = (
            gene["gene_name"], gene["chrom"], gene["strand"], gene["tss"],
            gene["window_start"], gene["window_end"],
        )
        expected = (
            str(row.gene_name), str(row.chrom), str(row.strand), int(row.TSS),
            int(row.gene_window_start), int(row.gene_window_end),
        )
        if observed != expected:
            raise ValueError(f"Manifest mismatch at gene {gene['gene_index']}: {observed} != {expected}")
    return genes


def read_tads(path: Path) -> tuple[list[dict], dict[str, list[dict]]]:
    tads = []
    with path.open(newline="") as handle:
        reader = csv.reader((line for line in handle if not line.startswith("#")), delimiter="\t")
        for annotation_index, fields in enumerate(reader):
            if len(fields) < 6:
                raise ValueError(f"Malformed TAD row in {path}")
            chrom, start, end, chrom2, start2, end2 = fields[:6]
            start, end, start2, end2 = map(int, (start, end, start2, end2))
            if not (chrom == chrom2 and start == start2 and end == end2 and start < end):
                raise ValueError(f"Invalid duplicated-domain BEDPE geometry: {path}")
            tads.append({"annotation_index": annotation_index, "chrom": chrom, "start": start, "end": end})
    by_chrom: dict[str, list[dict]] = defaultdict(list)
    for tad in tads:
        by_chrom[tad["chrom"]].append(tad)
    for values in by_chrom.values():
        values.sort(key=lambda item: (item["start"], item["end"], item["annotation_index"]))
    return tads, dict(by_chrom)


def ig_source(args, cfg, manifest_row: int, accession: str) -> tuple[Path, dict]:
    run_dir = args.ig_run / f"{manifest_row:02d}_{accession}"
    meta_path = run_dir / "meta.json"
    if not meta_path.is_file():
        raise FileNotFoundError(f"Run scripts/attribute_puget_ig.py for {args.biosample} first: {meta_path}")
    meta = read_json(meta_path)
    expected = {
        "completed": True,
        "manifest_row": manifest_row,
        "accession": accession,
        "biosample": args.biosample,
        "n_examples": N_GENES,
        "shape": [N_GENES, N_BINS, N_BINS],
        "dtype": "float32",
        "input_semantics": "log10(SCALE O/E + 1)",
    }
    for key, wanted in expected.items():
        if meta.get(key) != wanted:
            raise ValueError(f"Unexpected {key} in {meta_path}")
    signature = meta["signature"]
    if signature.get("test_bedpe_sha256") != sha256(cfg.test_bedpe):
        raise ValueError(f"IG maps were computed for a different test BEDPE: {meta_path}")
    if signature.get("method") != "integrated_gradients" or not signature.get("multiply_by_inputs"):
        raise ValueError(f"Unexpected attribution method: {meta_path}")
    scalars = pd.read_csv(run_dir / "scalars.tsv", sep="\t", usecols=["array_row", "gene_index", "biosample"])
    if (
        scalars["array_row"].tolist() != list(range(N_GENES))
        or scalars["gene_index"].tolist() != list(range(N_GENES))
        or set(scalars["biosample"]) != {args.biosample}
    ):
        raise ValueError(f"Unexpected IG row mapping: {run_dir / 'scalars.tsv'}")
    attrs_path = run_dir / "hic_attrs.npy"
    attrs = np.load(attrs_path, mmap_mode="r")
    if attrs.shape != (N_GENES, N_BINS, N_BINS) or attrs.dtype != np.float32:
        raise ValueError(f"Unexpected attribution array: {attrs_path}")
    return attrs_path, signature


def classify_columns(gene: dict, qualifying_tads: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    starts = gene["window_start"] + BIN_BP * np.arange(N_BINS, dtype=np.int64)
    ends = starts + BIN_BP
    inside = np.zeros(N_BINS, dtype=bool)
    overlap = np.zeros(N_BINS, dtype=bool)
    for tad in qualifying_tads:
        inside |= (starts >= tad["start"]) & (ends <= tad["end"])
        overlap |= (starts < tad["end"]) & (ends > tad["start"])
    outside = ~overlap
    boundary = overlap & ~inside
    if np.any(inside & outside) or not np.all(inside | outside | boundary):
        raise AssertionError("TAD masks do not form an exclusive partition")
    return inside, outside, boundary


def pool_contacts(attrs_path: Path, by_chrom: dict, genes: list[dict], biosample: str):
    attrs = np.load(attrs_path, mmap_mode="r")
    distance_by_column = (
        np.floor(np.abs(np.arange(N_BINS, dtype=np.float64) - 255.5)).astype(np.int16) + 1
    )
    groups = {
        group: {"attr": [], "gene": [], "column": [], "distance": []}
        for group in ("inside", "outside")
    }
    eligibility_rows = []
    n_boundary_columns = 0
    n_eligible = 0
    for gene in genes:
        promoter_start = gene["window_start"] + PROMOTER_START * BIN_BP
        promoter_end = gene["window_start"] + PROMOTER_END * BIN_BP
        tss_tads = [
            tad for tad in by_chrom.get(gene["chrom"], [])
            if tad["start"] <= gene["tss"] < tad["end"]
        ]
        qualifying = [
            tad for tad in tss_tads
            if tad["start"] <= promoter_start and promoter_end <= tad["end"]
        ]
        if qualifying:
            status = "eligible"
        elif tss_tads:
            status = "tss_tad_does_not_contain_full_promoter"
        else:
            status = "no_tss_containing_tad"
        eligibility_rows.append({
            "biosample": biosample,
            "gene_index": gene["gene_index"],
            "gene_name": gene["gene_name"],
            "chrom": gene["chrom"],
            "window_start": gene["window_start"],
            "window_end": gene["window_end"],
            "strand": gene["strand"],
            "tss": gene["tss"],
            "eligible": int(bool(qualifying)),
            "status": status,
            "n_tss_containing_tads": len(tss_tads),
            "n_qualifying_tads": len(qualifying),
        })
        if not qualifying:
            continue
        n_eligible += 1
        inside, outside, boundary = classify_columns(gene, qualifying)
        if gene["strand"] == "-":
            inside, outside, boundary = inside[::-1], outside[::-1], boundary[::-1]
        if not inside[PROMOTER_START:PROMOTER_END].all():
            raise AssertionError(f"Promoter columns not inside for gene {gene['gene_index']}")
        values = np.asarray(
            attrs[gene["gene_index"], PROMOTER_START:PROMOTER_END, :], dtype=np.float32
        ).T
        if values.shape != (N_BINS, 4) or not np.isfinite(values).all():
            raise ValueError(f"Invalid IG promoter slice for gene {gene['gene_index']}")
        for group, mask in (("inside", inside), ("outside", outside)):
            columns = np.flatnonzero(mask).astype(np.int16)
            groups[group]["attr"].append(values[mask])
            groups[group]["gene"].append(np.full(len(columns), gene["gene_index"], dtype=np.int32))
            groups[group]["column"].append(columns)
            groups[group]["distance"].append(distance_by_column[columns])
        n_boundary_columns += int(boundary.sum())

    pool = {
        "promoter_model_rows": np.asarray(PROMOTER_ROWS, dtype=np.int16),
        "distance_bin_by_model_column": distance_by_column,
    }
    counts = {"n_eligible_genes": n_eligible}
    for group in ("inside", "outside"):
        pool[f"{group}_attribution"] = np.concatenate(groups[group]["attr"], axis=0)
        pool[f"{group}_gene_index"] = np.concatenate(groups[group]["gene"])
        pool[f"{group}_distal_model_column"] = np.concatenate(groups[group]["column"])
        pool[f"{group}_distance_bin"] = np.concatenate(groups[group]["distance"])
        n_pairs = len(pool[f"{group}_gene_index"])
        if pool[f"{group}_attribution"].shape != (n_pairs, 4):
            raise AssertionError(f"Misaligned {group} attribution array")
        counts[f"{group}_gene_column_pairs"] = n_pairs
    counts["boundary_gene_column_pairs"] = n_boundary_columns
    if (counts["inside_gene_column_pairs"] + counts["outside_gene_column_pairs"]
            + n_boundary_columns != n_eligible * N_BINS):
        raise AssertionError("Eligible gene/column accounting failed")
    return pool, counts, eligibility_rows


def randomized_key_order(keys: np.ndarray, rng: np.random.Generator):
    order = np.lexsort((rng.random(len(keys)), keys))
    ordered = keys[order]
    unique, starts, counts = np.unique(ordered, return_index=True, return_counts=True)
    return order, unique, starts, counts


def matched_pairs(pool: dict[str, np.ndarray], rng: np.random.Generator) -> dict[str, np.ndarray]:
    inside_gene = pool["inside_gene_index"].astype(np.int32, copy=True)
    outside_gene = pool["outside_gene_index"].astype(np.int32, copy=True)
    inside_distance = pool["inside_distance_bin"].astype(np.int16, copy=True)
    outside_distance = pool["outside_distance_bin"].astype(np.int16, copy=True)
    inside_key = inside_gene.astype(np.int64) * (N_DISTANCES + 1) + inside_distance
    outside_key = outside_gene.astype(np.int64) * (N_DISTANCES + 1) + outside_distance
    io, iu, istart, icount = randomized_key_order(inside_key, rng)
    oo, ou, ostart, ocount = randomized_key_order(outside_key, rng)
    common, ipos, opos = np.intersect1d(iu, ou, assume_unique=True, return_indices=True)
    capacity = np.minimum(icount[ipos], ocount[opos]).astype(np.int64)
    stratum = np.repeat(np.arange(len(common), dtype=np.int64), capacity)
    cumulative = np.cumsum(capacity) - capacity
    offset = np.arange(int(capacity.sum()), dtype=np.int64) - np.repeat(cumulative, capacity)
    inside_index = io[istart[ipos[stratum]] + offset]
    outside_index = oo[ostart[opos[stratum]] + offset]
    key = common[stratum]
    gene = (key // (N_DISTANCES + 1)).astype(np.int32)
    distance = (key % (N_DISTANCES + 1)).astype(np.int16)
    if not (
        np.array_equal(gene, inside_gene[inside_index])
        and np.array_equal(gene, outside_gene[outside_index])
        and np.array_equal(distance, inside_distance[inside_index])
        and np.array_equal(distance, outside_distance[outside_index])
    ):
        raise AssertionError("Exact matched keys differ")
    return {
        "gene_index": gene,
        "distance_bin": distance,
        "inside_source_pair_index": inside_index.astype(np.int32),
        "outside_source_pair_index": outside_index.astype(np.int32),
        "inside_distal_model_column": pool["inside_distal_model_column"][inside_index],
        "outside_distal_model_column": pool["outside_distal_model_column"][outside_index],
    }


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--biosample", default=DEFAULT_BIOSAMPLE)
    parser.add_argument("--tads", type=Path, default=DEFAULT_TADS,
                        help="ENCODE TAD BEDPE for --biosample")
    parser.add_argument("--ig-run", type=Path, default=DEFAULT_IG_RUN,
                        help="run directory of scripts/attribute_puget_ig.py")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG,
                        help="Puget training YAML in configs/training")
    parser.add_argument("--output", type=Path, default=None,
                        help="default: outputs/borzoi_split_interpretation/tad_attribution/<biosample>")
    parser.add_argument("--target-per-class", type=int, default=None,
                        help="random subsample of matched pairs per class (default: keep all)")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args(argv)
    if args.target_per_class is not None and args.target_per_class <= 0:
        parser.error("--target-per-class must be positive")
    if args.output is None:
        args.output = DEFAULT_OUTPUT / args.biosample.replace(" ", "_")
    for name in ("tads", "ig_run", "config", "output"):
        setattr(args, name, getattr(args, name).resolve())
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {args.output}")
    partial = args.output.with_name(args.output.name + ".partial")
    if partial.exists():
        raise FileExistsError(f"Inspect interrupted output before rerunning: {partial}")
    if not args.tads.is_file():
        raise FileNotFoundError(args.tads)
    cfg = read_training_config(args.config)
    table = load_biosample_table(cfg.biosamples_csv)
    names = [name for _row, _accession, name in table]
    if args.biosample not in names:
        raise ValueError(f"Unknown biosample {args.biosample!r}; expected one of {names}")
    manifest_row, accession, _name = table[names.index(args.biosample)]
    genes = read_genes(cfg)
    attrs_path, signature = ig_source(args, cfg, manifest_row, accession)
    tads, by_chrom = read_tads(args.tads)

    pool, counts, eligibility_rows = pool_contacts(attrs_path, by_chrom, genes, args.biosample)
    print(
        f"{args.biosample}: {len(tads):,} TADs; eligible genes={counts['n_eligible_genes']:,}; "
        f"inside={counts['inside_gene_column_pairs']:,}, outside={counts['outside_gene_column_pairs']:,}, "
        f"boundary={counts['boundary_gene_column_pairs']:,} gene-column contacts",
        flush=True,
    )
    partial.mkdir(parents=True)
    atomic_save_npz(partial / "pool.npz", pool)
    with gzip.open(partial / "gene_tad_eligibility.tsv.gz", "wt", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(eligibility_rows[0]),
                                delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(eligibility_rows)

    rng = np.random.default_rng(args.seed)
    matched = matched_pairs(pool, rng)
    capacity = len(matched["gene_index"])
    target = capacity if args.target_per_class is None else args.target_per_class
    if target > capacity:
        raise ValueError(f"Requested {target:,} pairs per class, capacity is {capacity:,}")
    if target < capacity:
        selected = np.sort(rng.choice(capacity, size=target, replace=False))
        matched = {key: value[selected] for key, value in matched.items()}
    inside = np.abs(pool["inside_attribution"][matched["inside_source_pair_index"]]).sum(
        axis=1, dtype=np.float64).astype(np.float32)
    outside = np.abs(pool["outside_attribution"][matched["outside_source_pair_index"]]).sum(
        axis=1, dtype=np.float64).astype(np.float32)
    if not (np.isfinite(inside).all() and np.isfinite(outside).all()):
        raise AssertionError("Nonfinite matched absolute sums")
    atomic_save_npz(partial / "matched_pairs.npz", {
        **matched,
        "inside_abs_attribution_sum": inside,
        "outside_abs_attribution_sum": outside,
    })
    print(f"Matched {capacity:,} inside/outside pairs per class; using {target:,}", flush=True)

    labels = np.concatenate((np.ones(target, dtype=np.uint8), np.zeros(target, dtype=np.uint8)))
    scores = np.concatenate((inside, outside)).astype(np.float64, copy=False)
    auc = float(roc_auc_score(labels, scores))
    mw = mannwhitneyu(inside, outside, alternative="two-sided", method="asymptotic")
    probability = float(mw.statistic / (target * target))
    if not np.isclose(auc, probability, atol=2e-8):
        raise AssertionError("ROC AUC and normalized Mann-Whitney U disagree")
    statistics = {
        "biosample": args.biosample,
        "n_per_class": target,
        "max_matched_pairs_per_class": capacity,
        "inside_mean": float(np.mean(inside, dtype=np.float64)),
        "inside_median": float(np.median(inside)),
        "outside_mean": float(np.mean(outside, dtype=np.float64)),
        "outside_median": float(np.median(outside)),
        "auroc_inside_positive": auc,
        "mann_whitney_u": float(mw.statistic),
        "mann_whitney_two_sided_p": float(mw.pvalue),
        "rank_biserial_inside_vs_outside": 2 * probability - 1,
    }
    write_tsv(partial / "statistics.tsv", [statistics])
    distance_counts = np.bincount(matched["distance_bin"], minlength=N_DISTANCES + 1)
    write_tsv(partial / "counts_by_distance.tsv", [
        {"distance_bin": distance, "n_per_class": int(distance_counts[distance])}
        for distance in range(1, N_DISTANCES + 1)
    ])
    write_json(partial / "meta.json", {
        "completed": True,
        "description": "Distance-matched inside/outside-TAD absolute promoter-row IG",
        "biosample": args.biosample,
        "n_genes": N_GENES,
        "n_tads": len(tads),
        **counts,
        "target_pairs_per_class": target,
        "maximal_exact_match_capacity_per_class": capacity,
        "seed": args.seed,
        "matching_unit": "gene_index x unsigned distance_bin",
        "matching": (
            "Maximal 1:1 inside/outside matching within every stratum; optionally a fixed-seed "
            "simple random sample without replacement from all matched units."
        ),
        "eligibility": (
            "A TSS-containing TAD must fully contain genomic promoter bins 254:258. "
            "Inside columns are fully contained, outside columns have no overlap, and boundary "
            "columns are excluded. Genomic masks are reversed for minus-strand gene-oriented IG."
        ),
        "distance": "floor(abs(model_column - 255.5)) + 1; integer 1..256; TSS sides pooled",
        "scalar_value": (
            "abs(attr[row254,column]) + abs(attr[row255,column]) + "
            "abs(attr[row256,column]) + abs(attr[row257,column])"
        ),
        "statistics": statistics,
        "note": "Descriptive discrimination of one scalar score; contacts within a gene are not independent.",
        "inputs": {
            "ig_attrs": str(attrs_path),
            "ig_signature": signature,
            "tads": str(args.tads),
            "tads_sha256": sha256(args.tads),
            "test_bedpe": str(cfg.test_bedpe),
            "test_bedpe_sha256": sha256(cfg.test_bedpe),
        },
    })
    partial.rename(args.output)
    print(f"AUROC (inside positive) = {auc:.4f}; Mann-Whitney p = {mw.pvalue:.3g}")
    print(f"Saved {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
