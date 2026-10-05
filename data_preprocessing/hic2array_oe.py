from __future__ import annotations

import argparse
from pathlib import Path

from _hic2array_common import convert_hic


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-hic", type=Path, required=True)
    parser.add_argument("--output-pkl", type=Path, required=True)
    parser.add_argument("--resolution", type=int, default=1000)
    parser.add_argument("--force", action="store_true", help="Replace an existing output")
    args = parser.parse_args()
    convert_hic(args.input_hic, args.output_pkl, args.resolution, "oe", "SCALE", args.force)


if __name__ == "__main__":
    main()
