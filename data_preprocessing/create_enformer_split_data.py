from __future__ import annotations

import argparse
from pathlib import Path

from _native_window_gene_split import build_split_data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotation-csv", type=Path, required=True,
                        help="Filtered EPInformer gene table from prepare_gene_table.py")
    parser.add_argument("--biosample-csv", type=Path, required=True,
                        help="Ordered biosample-to-accession mapping CSV")
    parser.add_argument("--native-bed", type=Path, required=True,
                        help="Enformer sequences.bed with train/valid/test labels")
    parser.add_argument("--chrom-sizes", type=Path, required=True,
                        help="hg38 chromosome sizes")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Directory for manifest, BED/BEDPE files, labels, and summaries")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_split_data(
        model_name="Enformer",
        native_window_bp=131_072,
        gene_window_bp=196_608,
        gene_window_label="196k",
        test_label="test",
        valid_label="valid",
        train_labels=("train",),
        annotation_csv=args.annotation_csv,
        biosample_csv=args.biosample_csv,
        native_bed=args.native_bed,
        chrom_sizes_path=args.chrom_sizes,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
