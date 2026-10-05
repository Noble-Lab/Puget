from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from puget.seq_only_model import build_model
from scripts._puget_config import load_config


def train_biosamples(path: Path) -> list[str]:
    names = []
    with path.open() as handle:
        for line in handle:
            fields = line.rstrip("\n").split("\t")
            if len(fields) == 4 and fields[0] == "traincell":
                if int(fields[1]) != len(names):
                    raise ValueError(f"Unexpected traincell row order in {path}")
                names.append(fields[2])
    if len(names) != 14 or len(set(names)) != 14 or {"GM12878", "K562"} & set(names):
        raise ValueError(f"Expected 14 ordered training biosamples in {path}: {names}")
    return names


def check_gene_order(bedpe: Path, metadata: dict) -> int:
    source = Path(str(metadata.get("bed", "")))
    if not source.is_absolute():
        # The embedding job saved its BED relative to data_preprocessing/.
        source = (ROOT / "data_preprocessing" / source).resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Embedding source BED is missing: {source}")
    with source.open(newline="") as handle:
        source_rows = list(csv.reader(handle, delimiter="\t"))
    with bedpe.open(newline="") as handle:
        bedpe_rows = list(csv.reader(handle, delimiter="\t"))
    if not bedpe_rows or len(source_rows) != len(bedpe_rows):
        raise ValueError("Embedding BED and test BEDPE have different gene counts")
    for index, (bed, pair) in enumerate(zip(source_rows, bedpe_rows)):
        if len(bed) < 6 or len(pair) < 10 or pair[:3] != pair[3:6]:
            raise ValueError(f"Malformed or nonsquare test window at gene {index}")
        if (bed[0], bed[1], bed[2], bed[3], bed[5]) != (
                pair[0], pair[1], pair[2], pair[6], pair[8]):
            raise ValueError(f"Sequence embedding and test BEDPE differ at gene {index}")
    return len(bedpe_rows)


def atomic_write(path: Path, writer) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    try:
        with temporary.open("wb") as handle:
            writer(handle)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, help="inference YAML in configs/inference")
    parser.add_argument("--gpu", type=int, default=None,
                        help="physical GPU index for this job (sets CUDA_VISIBLE_DEVICES)")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    if args.gpu is not None:
        if args.gpu < 0:
            raise SystemExit("--gpu must be a non-negative GPU index")
        # CUDA is not initialized before this point, so this selects the device.
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    config = load_config(args.config)
    checkpoint, embedding, bedpe, biosample_log, output_dir = (
        ROOT / config[key] for key in
        ("checkpoint", "test_seq", "test_bedpe", "biosample_log", "output_dir"))
    batch_size = int(config["batch_size"])
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    for path in (checkpoint, embedding, bedpe, biosample_log):
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(path)

    names = train_biosamples(biosample_log)
    metadata = json.loads(embedding.with_suffix(".metadata.json").read_text())
    n_genes = check_gene_order(bedpe, metadata)
    values = np.load(embedding, mmap_mode="r")
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg = saved.get("config", {})
    if (cfg.get("biosample_names") != names or cfg.get("output_dim") != 14
            or cfg.get("embedding_key") != config["embedding_key"]
            or cfg.get("split_family") != config["split_family"]
            or cfg.get("output_activation") != "softplus"):
        raise ValueError(f"{checkpoint} does not match the ordered 14-head "
                         f"{config['model_name']} model")
    if "state_dict" not in saved:
        raise ValueError("Sequence checkpoint has no model state_dict")
    if (metadata.get("split") != "test" or tuple(metadata.get("shape", ())) != values.shape
            or values.shape != (n_genes, cfg["n_tokens"], cfg["embed_dim"])
            or metadata.get("seq_length") != cfg["n_tokens"] * 1024
            or metadata.get("bin_bp") != 1024 or values.dtype != np.float16):
        raise ValueError("Test embedding metadata, shape, dtype, or BEDPE count differs")

    paths = (output_dir / "seq_only_testgenes_train14_pred.npy",
             output_dir / "seq_only_testgenes_train14_biosamples.txt",
             output_dir / "seq_only_testgenes_train14_metadata.json")
    print(f"Preflight passed: {n_genes} test genes x {len(names)} training biosamples; "
          f"checkpoint={checkpoint}", flush=True)
    if args.preflight_only:
        return 0
    if any(path.exists() for path in paths) and not args.overwrite:
        raise FileExistsError(f"Inference output already exists: {paths}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; inference requires a GPU")
    device = torch.device("cuda")
    model = build_model(
        seq_input_dim=cfg["embed_dim"], n_tokens=cfg["n_tokens"],
        output_dim=cfg["output_dim"], proj_dim=cfg["proj_dim"],
        num_heads=cfg["num_heads"], num_layers=cfg["num_layers"],
        mlp_ratio=cfg["mlp_ratio"], dropout_p=cfg["dropout"],
        promoter_tokens=cfg["promoter_tokens"], input_norm=cfg["input_norm"],
        attn_impl=cfg["attn"], output_activation=cfg["output_activation"],
    )
    model.load_state_dict(saved["state_dict"], strict=True)
    del saved
    model.eval().to(device)
    precision = str(cfg.get("precision", "fp32"))
    use_amp = precision in {"bf16", "fp16"}
    amp_dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    predictions = np.empty((14, n_genes), dtype=np.float32)
    with torch.inference_mode():
        for start in range(0, n_genes, batch_size):
            stop = min(start + batch_size, n_genes)
            sequence = torch.from_numpy(np.array(values[start:stop], copy=True)).to(device)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                output = model(sequence)
            predictions[:, start:stop] = output.float().cpu().numpy().T
    if not np.isfinite(predictions).all() or (predictions < 0).any():
        raise ValueError("Sequence-only predictions contain nonfinite or negative values")

    atomic_write(paths[0], lambda handle: np.save(handle, predictions))
    atomic_write(paths[1], lambda handle: handle.write(("\n".join(names) + "\n").encode()))
    result = {
        "model": config["model_name"], "group": "train14",
        "embedding_key": cfg["embedding_key"], "split_family": cfg["split_family"],
        "config": str(Path(args.config).resolve()),
        "checkpoint": str(checkpoint.resolve()),
        "embedding": str(embedding.resolve()), "test_bedpe": str(bedpe.resolve()),
        "biosamples": names, "prediction_layout": "biosample_x_test_gene_in_bedpe_order",
        "shape": list(predictions.shape), "dtype": "float32", "n_nan": 0,
    }
    atomic_write(paths[2], lambda handle: handle.write((json.dumps(result, indent=2) + "\n").encode()))
    print(f"Saved {paths[0]}: {predictions.shape}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
