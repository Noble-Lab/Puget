from __future__ import annotations

import bisect
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


SPLITS = ("train", "valid", "test")
TEST_BIOSAMPLES = ("GM12878", "K562")


class IntervalIndex:
    def __init__(self, records: Iterable[tuple[str, int, int, str]]):
        grouped: dict[str, dict[str, list[tuple[int, int]]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for chrom, start, end, label in records:
            if start >= end:
                raise ValueError(f"Invalid interval: {chrom}:{start}-{end} ({label})")
            grouped[label][chrom].append((start, end))

        self._data: dict[str, dict[str, tuple[list[int], list[int], list[int]]]] = {}
        for label, by_chrom in grouped.items():
            self._data[label] = {}
            for chrom, intervals in by_chrom.items():
                intervals.sort()
                starts = [start for start, _ in intervals]
                ends = [end for _, end in intervals]
                prefix_max_ends: list[int] = []
                maximum = -1
                for end in ends:
                    maximum = max(maximum, end)
                    prefix_max_ends.append(maximum)
                self._data[label][chrom] = (starts, ends, prefix_max_ends)

    def overlaps(self, chrom: str, start: int, end: int, labels: Iterable[str]) -> bool:
        if start >= end:
            return False
        for label in labels:
            entry = self._data.get(label, {}).get(chrom)
            if entry is None:
                continue
            starts, ends, prefix_max_ends = entry
            interval_i = bisect.bisect_left(starts, end) - 1
            while interval_i >= 0 and prefix_max_ends[interval_i] > start:
                # starts[interval_i] < end by construction.  Therefore both
                # strict inequalities below mean an overlap of at least 1 bp.
                if ends[interval_i] > start:
                    return True
                interval_i -= 1
        return False

    def contains_tss(self, chrom: str, tss: int, labels: Iterable[str]) -> bool:
        # A TSS at an interval end is outside that BED interval, as required by
        # the BED half-open convention.
        return self.overlaps(chrom, tss, tss + 1, labels)


def read_chrom_sizes(path: Path) -> dict[str, int]:
    sizes: dict[str, int] = {}
    with path.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip() or line.startswith("#"):
                continue
            fields = line.split()
            if len(fields) < 2:
                raise ValueError(f"{path}:{line_number}: expected chromosome and size")
            sizes[fields[0]] = int(fields[1])
    return sizes


def read_native_windows(
    path: Path, expected_window_bp: int, allowed_labels: Sequence[str]
) -> tuple[IntervalIndex, Counter[str]]:
    records: list[tuple[str, int, int, str]] = []
    counts: Counter[str] = Counter()
    allowed = set(allowed_labels)
    with path.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip() or line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 4:
                raise ValueError(f"{path}:{line_number}: expected at least four BED columns")
            chrom, start_text, end_text, label = fields[:4]
            start, end = int(start_text), int(end_text)
            if end - start != expected_window_bp:
                raise ValueError(
                    f"{path}:{line_number}: expected {expected_window_bp}-bp native interval, "
                    f"found {end - start} bp"
                )
            if label not in allowed:
                raise ValueError(f"{path}:{line_number}: unexpected native label {label!r}")
            records.append((chrom, start, end, label))
            counts[label] += 1
    if not records:
        raise ValueError(f"No native split intervals found in {path}")
    return IntervalIndex(records), counts


def model_interval(tss: int, chrom_size: int, length_bp: int) -> tuple[int, int]:
    half = length_bp // 2
    start, end = tss - half, tss + half
    if start < 0:
        end -= start
        start = 0
    if end > chrom_size:
        start -= end - chrom_size
        end = chrom_size
    if start < 0 or end > chrom_size or end - start != length_bp:
        raise ValueError(
            f"Cannot place a {length_bp}-bp model interval around TSS {tss} "
            f"on a {chrom_size}-bp chromosome"
        )
    return start, end


def _chrom_rank(chrom: str) -> tuple[int, int | str]:
    token = chrom.removeprefix("chr")
    if token.isdigit():
        return (0, int(token))
    if token == "X":
        return (1, 23)
    if token == "Y":
        return (1, 24)
    return (2, token)


def _write_bed_and_bedpe(model_name: str, length_label: str, output_dir: Path, split: str, genes: pd.DataFrame) -> None:
    bed_path = output_dir / f"{model_name}_genes_{length_label}_{split}.bed"
    bedpe_path = output_dir / f"{model_name}_genes_{length_label}_{length_label}_{split}.bedpe"
    with bed_path.open("w") as bed, bedpe_path.open("w") as bedpe:
        for row in genes.itertuples(index=False):
            bed.write(
                f"{row.chrom}\t{row.gene_window_start}\t{row.gene_window_end}\t"
                f"{row.gene_name}\t0\t{row.strand}\n"
            )
            bedpe.write(
                f"{row.chrom}\t{row.gene_window_start}\t{row.gene_window_end}\t"
                f"{row.chrom}\t{row.gene_window_start}\t{row.gene_window_end}\t"
                f"{row.gene_name}\t0\t{row.strand}\t{row.strand}\n"
            )


def _expression_columns(manifest: pd.DataFrame, biosample_csv: Path) -> tuple[list[str], list[str], list[tuple[str, str]], list[tuple[str, str]]]:
    bios = pd.read_csv(biosample_csv)
    required = {"Biosample", "Hi-C accession"}
    missing = required - set(bios.columns)
    if missing:
        raise ValueError(f"{biosample_csv}: missing columns {sorted(missing)}")

    pairs = [(str(row["Biosample"]), str(row["Hi-C accession"])) for _, row in bios.iterrows()]
    missing_expression = [accession for _, accession in pairs if accession not in manifest.columns]
    if missing_expression:
        raise ValueError(
            "The gene manifest is missing RNA columns for: " + ", ".join(missing_expression)
        )
    if len({accession for _, accession in pairs}) != len(pairs):
        raise ValueError(f"{biosample_csv}: duplicate Hi-C accessions")

    test_pairs = [pair for pair in pairs if pair[0] in TEST_BIOSAMPLES]
    if tuple(biosample for biosample, _ in test_pairs) != TEST_BIOSAMPLES:
        raise ValueError(
            f"Expected exactly the test biosamples {TEST_BIOSAMPLES}; found {test_pairs}"
        )
    train_pairs = [pair for pair in pairs if pair[0] not in TEST_BIOSAMPLES]
    return (
        [accession for _, accession in train_pairs],
        [accession for _, accession in test_pairs],
        train_pairs,
        test_pairs,
    )


def _write_labels(
    output_dir: Path,
    sorted_splits: dict[str, pd.DataFrame],
    train_columns: list[str],
    test_columns: list[str],
) -> None:
    for split, genes in sorted_splits.items():
        for cell_group, columns in (("traincell", train_columns), ("testcell", test_columns)):
            labels = genes[columns].to_numpy(dtype=np.float32).T
            expected_shape = (len(columns), len(genes))
            if labels.shape != expected_shape:
                raise AssertionError(
                    f"{cell_group}_{split}gene labels have shape {labels.shape}, expected {expected_shape}"
                )
            np.save(output_dir / f"{cell_group}_{split}gene.npy", labels)


def _write_label_columns_log(
    output_dir: Path, train_pairs: list[tuple[str, str]], test_pairs: list[tuple[str, str]]
) -> None:
    lines = [
        "RNA label row order",
        "",
        "group\trow_index\tbiosample\texpression_column",
    ]
    for group, pairs in (("traincell", train_pairs), ("testcell", test_pairs)):
        lines.extend(
            f"{group}\t{i}\t{biosample}\t{accession}"
            for i, (biosample, accession) in enumerate(pairs)
        )
    (output_dir / "RNA_label_columns.log").write_text("\n".join(lines) + "\n")


def _write_summary(
    path: Path,
    *,
    model_name: str,
    annotation_csv: Path,
    native_bed: Path,
    output_dir: Path,
    native_window_bp: int,
    gene_window_bp: int,
    native_counts: Counter[str],
    stage_counts: Counter[str],
    final_counts: Counter[str],
    train_pairs: list[tuple[str, str]],
    test_pairs: list[tuple[str, str]],
) -> None:
    lines = [
        f"{model_name} leakage-safe native-window gene split summary",
        "",
        f"input_gene_table\t{annotation_csv}",
        f"native_split_bed\t{native_bed}",
        f"output_directory\t{output_dir}",
        f"native_window_bp\t{native_window_bp}",
        f"model_input_gene_window_bp\t{gene_window_bp}",
        "overlap_rule\thalf-open BED intervals; intersection must be >= 1 bp",
        "",
        "native_window_counts",
    ]
    lines.extend(f"{label}\t{native_counts[label]}" for label in sorted(native_counts))
    lines.extend(["", "assignment_and_drop_counts"])
    ordered_counts = (
        "input_genes",
        "test_tss_inside_native_target",
        "test_assigned_preliminary",
        "test_dropped_native_non_target_overlap",
        "valid_tss_inside_native_target_after_test_stage",
        "valid_assigned_preliminary",
        "valid_dropped_native_non_target_overlap",
        "train_assigned_preliminary",
        "dropped_before_final_test_decontamination",
        "test_dropped_final_gene_interval_overlap",
        "total_dropped",
    )
    lines.extend(f"{key}\t{stage_counts[key]}" for key in ordered_counts)
    lines.extend(["", "final_gene_split_counts"])
    lines.extend(f"{split}\t{final_counts[split]}" for split in (*SPLITS, "exclude"))
    lines.extend([
        "",
        "label_cell_groups",
        "traincell\t" + ",".join(f"{bio}:{acc}" for bio, acc in train_pairs),
        "testcell\t" + ",".join(f"{bio}:{acc}" for bio, acc in test_pairs),
        "label_layout\t(cell_lines, genes_in_the_matching_BED_order), float32 log2(TPM + 1)",
    ])
    path.write_text("\n".join(lines) + "\n")


def build_split_data(
    *,
    model_name: str,
    native_window_bp: int,
    gene_window_bp: int,
    gene_window_label: str,
    test_label: str,
    valid_label: str,
    train_labels: Sequence[str],
    annotation_csv: Path,
    biosample_csv: Path,
    native_bed: Path,
    chrom_sizes_path: Path,
    output_dir: Path,
) -> None:
    if not annotation_csv.exists():
        raise FileNotFoundError(annotation_csv)
    if not biosample_csv.exists():
        raise FileNotFoundError(biosample_csv)
    if not native_bed.exists():
        raise FileNotFoundError(native_bed)
    if not chrom_sizes_path.exists():
        raise FileNotFoundError(chrom_sizes_path)

    labels = tuple(train_labels) + (valid_label, test_label)
    if len(set(labels)) != len(labels):
        raise ValueError("Native train, validation, and test labels must be distinct")
    native_index, native_counts = read_native_windows(native_bed, native_window_bp, labels)
    chrom_sizes = read_chrom_sizes(chrom_sizes_path)

    manifest = pd.read_csv(annotation_csv)
    required = {"gene_id", "gene_name", "chrom", "strand", "TSS"}
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(f"{annotation_csv}: missing columns {sorted(missing)}")
    if not manifest["gene_id"].is_unique:
        raise ValueError(f"{annotation_csv}: gene_id is not unique")
    if manifest["chrom"].map(chrom_sizes).isna().any():
        bad_chroms = sorted(manifest.loc[manifest["chrom"].map(chrom_sizes).isna(), "chrom"].unique())
        raise ValueError(f"Chromosome sizes missing for {bad_chroms}")

    manifest["TSS"] = pd.to_numeric(manifest["TSS"], downcast="integer")
    intervals = [
        model_interval(int(tss), chrom_sizes[chrom], gene_window_bp)
        for chrom, tss in zip(manifest["chrom"], manifest["TSS"])
    ]
    manifest["gene_window_start"] = [start for start, _ in intervals]
    manifest["gene_window_end"] = [end for _, end in intervals]
    manifest["preliminary_split"] = "pending"
    manifest["split"] = "pending"
    manifest["assignment_reason"] = "pending"
    manifest["bed_row_index"] = pd.Series(pd.NA, index=manifest.index, dtype="Int64")

    other_than_test = tuple(train_labels) + (valid_label,)
    other_than_valid = tuple(train_labels) + (test_label,)
    held_out = (valid_label, test_label)

    test_member = np.fromiter(
        (
            native_index.contains_tss(chrom, int(tss), (test_label,))
            for chrom, tss in zip(manifest["chrom"], manifest["TSS"])
        ),
        dtype=bool,
        count=len(manifest),
    )
    test_clean = np.fromiter(
        (
            not native_index.overlaps(chrom, int(start), int(end), other_than_test)
            for chrom, start, end in zip(
                manifest["chrom"], manifest["gene_window_start"], manifest["gene_window_end"]
            )
        ),
        dtype=bool,
        count=len(manifest),
    )
    initial_test = test_member & test_clean
    manifest.loc[test_member, "assignment_reason"] = "test_native_membership_rejected_by_non_test_native_overlap"
    manifest.loc[initial_test, ["preliminary_split", "split", "assignment_reason"]] = (
        "test",
        "test",
        "test_assigned_from_clean_native_test_window",
    )

    pending_after_test = manifest["split"].eq("pending").to_numpy()
    valid_member = np.fromiter(
        (
            native_index.contains_tss(chrom, int(tss), (valid_label,))
            for chrom, tss in zip(manifest["chrom"], manifest["TSS"])
        ),
        dtype=bool,
        count=len(manifest),
    )
    valid_clean = np.fromiter(
        (
            not native_index.overlaps(chrom, int(start), int(end), other_than_valid)
            for chrom, start, end in zip(
                manifest["chrom"], manifest["gene_window_start"], manifest["gene_window_end"]
            )
        ),
        dtype=bool,
        count=len(manifest),
    )
    initial_valid = pending_after_test & valid_member & valid_clean
    manifest.loc[pending_after_test & valid_member, "assignment_reason"] = (
        "valid_native_membership_rejected_by_non_valid_native_overlap"
    )
    manifest.loc[initial_valid, ["preliminary_split", "split", "assignment_reason"]] = (
        "valid",
        "valid",
        "valid_assigned_from_clean_native_valid_window",
    )

    pending_after_valid = manifest["split"].eq("pending").to_numpy()
    safe_for_train = np.fromiter(
        (
            not native_index.overlaps(chrom, int(start), int(end), held_out)
            for chrom, start, end in zip(
                manifest["chrom"], manifest["gene_window_start"], manifest["gene_window_end"]
            )
        ),
        dtype=bool,
        count=len(manifest),
    )
    initial_train = pending_after_valid & safe_for_train
    prefinal_exclude = pending_after_valid & ~safe_for_train
    manifest.loc[initial_train, ["preliminary_split", "split", "assignment_reason"]] = (
        "train",
        "train",
        "train_assigned_without_native_valid_or_test_overlap",
    )
    manifest.loc[prefinal_exclude, ["preliminary_split", "split", "assignment_reason"]] = (
        "exclude",
        "exclude",
        "excluded_gene_interval_overlaps_native_valid_or_test_window",
    )
    if manifest["split"].eq("pending").any():
        raise AssertionError("Some genes were not assigned after the train stage")

    preliminary_other_records = [
        (row.chrom, int(row.gene_window_start), int(row.gene_window_end), "train_or_valid")
        for row in manifest.loc[manifest["split"].isin(("train", "valid"))].itertuples(index=False)
    ]
    preliminary_other_index = IntervalIndex(preliminary_other_records)
    final_test_overlap = np.fromiter(
        (
            preliminary_other_index.overlaps(
                row.chrom,
                int(row.gene_window_start),
                int(row.gene_window_end),
                ("train_or_valid",),
            )
            if row.split == "test"
            else False
            for row in manifest.itertuples(index=False)
        ),
        dtype=bool,
        count=len(manifest),
    )
    manifest.loc[final_test_overlap, "split"] = "exclude"
    manifest.loc[final_test_overlap, "assignment_reason"] = (
        "excluded_test_gene_interval_overlaps_preliminary_train_or_valid_gene"
    )

    # Final invariant required by the split scheme.
    final_other_records = [
        (row.chrom, int(row.gene_window_start), int(row.gene_window_end), "train_or_valid")
        for row in manifest.loc[manifest["split"].isin(("train", "valid"))].itertuples(index=False)
    ]
    final_other_index = IntervalIndex(final_other_records)
    for row in manifest.loc[manifest["split"].eq("test")].itertuples(index=False):
        if final_other_index.overlaps(
            row.chrom, int(row.gene_window_start), int(row.gene_window_end), ("train_or_valid",)
        ):
            raise AssertionError(f"Residual test-gene overlap: {row.gene_id}")

    final_counts: Counter[str] = Counter(manifest["split"])
    stage_counts: Counter[str] = Counter(
        {
            "input_genes": len(manifest),
            "test_tss_inside_native_target": int(test_member.sum()),
            "test_assigned_preliminary": int(initial_test.sum()),
            "test_dropped_native_non_target_overlap": int((test_member & ~test_clean).sum()),
            "valid_tss_inside_native_target_after_test_stage": int(
                (pending_after_test & valid_member).sum()
            ),
            "valid_assigned_preliminary": int(initial_valid.sum()),
            "valid_dropped_native_non_target_overlap": int(
                (pending_after_test & valid_member & ~valid_clean).sum()
            ),
            "train_assigned_preliminary": int(initial_train.sum()),
            "dropped_before_final_test_decontamination": int(prefinal_exclude.sum()),
            "test_dropped_final_gene_interval_overlap": int(final_test_overlap.sum()),
            "total_dropped": int(final_counts["exclude"]),
        }
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    sorted_splits: dict[str, pd.DataFrame] = {}
    for split in SPLITS:
        genes = manifest.loc[manifest["split"].eq(split)].copy()
        genes["_chrom_rank"] = genes["chrom"].map(_chrom_rank)
        genes = genes.sort_values(
            by=["_chrom_rank", "gene_window_start", "gene_window_end", "gene_id"], kind="stable"
        ).drop(columns="_chrom_rank")
        genes["bed_row_index"] = np.arange(len(genes), dtype=np.int64)
        manifest.loc[genes.index, "bed_row_index"] = genes["bed_row_index"].astype("Int64")
        sorted_splits[split] = genes
        _write_bed_and_bedpe(model_name, gene_window_label, output_dir, split, genes)

    train_columns, test_columns, train_pairs, test_pairs = _expression_columns(manifest, biosample_csv)
    _write_labels(output_dir, sorted_splits, train_columns, test_columns)
    _write_label_columns_log(output_dir, train_pairs, test_pairs)

    manifest_path = output_dir / f"{model_name}_gene_manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    summary_path = output_dir / f"{model_name}_split_summary.log"
    _write_summary(
        summary_path,
        model_name=model_name,
        annotation_csv=annotation_csv,
        native_bed=native_bed,
        output_dir=output_dir,
        native_window_bp=native_window_bp,
        gene_window_bp=gene_window_bp,
        native_counts=native_counts,
        stage_counts=stage_counts,
        final_counts=final_counts,
        train_pairs=train_pairs,
        test_pairs=test_pairs,
    )

    print(f"Wrote {manifest_path}")
    print(f"Wrote {summary_path}")
    for split in SPLITS:
        print(f"{split}: {final_counts[split]} genes")
    print(f"exclude: {final_counts['exclude']} genes")
