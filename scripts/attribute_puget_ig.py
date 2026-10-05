from __future__ import annotations

import argparse
import gc
import os
from pathlib import Path
import pickle
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(variable, "1")

import captum
from captum.attr import IntegratedGradients
import numpy as np
import pandas as pd
import torch

from scripts.build_ig_baseline import (
    ATTRIBUTION_DIR, DEFAULT_BASELINE_DIR, DEFAULT_CONFIG, EMBED_DIM, N_BINS, file_identity,
    gene_table, hic_source, read_json, read_training_config, sha256, transformed_map,
    write_json,
)
from puget.biosamples import load_biosample_table
from puget.puget_model import BiasModel, VariantLit

DEFAULT_CHECKPOINT = ROOT / "data" / "trained_models" / "borzoi_split" / "puget_alphagenome_best.ckpt"
DEFAULT_BASELINE = DEFAULT_BASELINE_DIR / "baseline.npy"
DEFAULT_OUTPUT = ATTRIBUTION_DIR / "puget_alphagenome_hic_ig_fp32"
ATTR_FILE = "hic_attrs.npy"
COMPLETENESS_ATOL = 1e-3
COMPLETENESS_RTOL = 1e-2


def cell_table(cfg) -> pd.DataFrame:
    manifest = load_biosample_table(cfg.biosamples_csv)
    frame = pd.DataFrame(manifest, columns=["manifest_row", "accession", "biosample"])
    train_rank = {name: index for index, name in enumerate(cfg.train_biosamples)}
    test_rank = {name: index for index, name in enumerate(cfg.test_biosamples)}
    groups, rows = [], []
    for name in frame.biosample:
        if name in train_rank:
            groups.append("train14")
            rows.append(train_rank[name])
        elif name in test_rank:
            groups.append("heldout2")
            rows.append(test_rank[name])
        else:
            raise ValueError(f"Manifest biosample is absent from the model split: {name}")
    frame["model_split"] = groups
    frame["label_row"] = rows
    return frame.sort_values("manifest_row").reset_index(drop=True)


def select_cells(cells: pd.DataFrame, specification: str) -> pd.DataFrame:
    if specification == "all":
        selected = cells
    elif specification == "seen":
        selected = cells[cells.model_split == "train14"]
    elif specification == "unseen":
        selected = cells[cells.model_split == "heldout2"]
    else:
        requested = specification.split(",")
        missing = [name for name in requested if name not in set(cells.biosample)]
        if missing:
            raise ValueError(f"Unknown biosamples: {missing}")
        rank = {name: index for index, name in enumerate(requested)}
        selected = cells[cells.biosample.isin(requested)].copy()
        selected["selection_order"] = selected.biosample.map(rank)
        selected = selected.sort_values("selection_order").drop(columns="selection_order")
    return selected.reset_index(drop=True)


def validate_static_inputs(args):
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    cfg = read_training_config(args.config)
    genes = gene_table(cfg.test_bedpe, strand=True)
    cells = cell_table(cfg)
    if len(cells) != 16 or cells.manifest_row.tolist() != list(range(16)):
        raise ValueError("Expected all 16 paper biosamples in manifest-row order")
    sequence = np.load(cfg.test_seq, mmap_mode="r")
    if sequence.shape != (len(genes), N_BINS, EMBED_DIM) or sequence.dtype != np.float16:
        raise ValueError(f"Unexpected test sequence array: {sequence.shape}, {sequence.dtype}")

    train_labels_path = Path(cfg.test_label).with_name("traincell_testgene.npy")
    test_labels_path = Path(cfg.test_label)
    labels = {
        "train14": np.load(train_labels_path, mmap_mode="r"),
        "heldout2": np.load(test_labels_path, mmap_mode="r"),
    }
    if labels["train14"].shape != (14, len(genes)):
        raise ValueError(f"Unexpected train-cell test-gene labels: {labels['train14'].shape}")
    if labels["heldout2"].shape != (2, len(genes)):
        raise ValueError(f"Unexpected held-out labels: {labels['heldout2'].shape}")
    if not all(np.isfinite(value).all() for value in labels.values()):
        raise ValueError("Nonfinite test-gene label value")

    for cell in cells.itertuples(index=False):
        source = hic_source(cfg, "test", cell.accession)
        if not source.is_file() or source.stat().st_size == 0:
            raise FileNotFoundError(source)
    return cfg, genes, cells, labels


def validate_baseline(args, cfg) -> tuple[np.ndarray, str]:
    path = args.hic_baseline_file
    metadata_path = path.with_name("meta.json")
    if not path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(
            f"Missing train-14 baseline. Build it first with scripts/build_ig_baseline.py: {path}"
        )
    value = np.load(path)
    if (
        value.shape != (N_BINS, N_BINS)
        or value.dtype != np.float32
        or not np.isfinite(value).all()
        or (value < 0).any()
        or not np.array_equal(value, value.T)
    ):
        raise ValueError(f"Invalid train-14 distance-mean baseline: {path}")
    meta = read_json(metadata_path)
    required = {
        "completed": True,
        "n_training_gene_rows": len(gene_table(cfg.train_bedpe)),
        "n_training_biosamples": 14,
        "shape": [N_BINS, N_BINS],
        "dtype": "float32",
        "training_config_sha256": sha256(args.config),
        "train_bedpe_sha256": sha256(cfg.train_bedpe),
        "biosamples_manifest_sha256": sha256(cfg.biosamples_csv),
        "baseline_sha256": sha256(path),
    }
    for key, wanted in required.items():
        if meta.get(key) != wanted:
            raise ValueError(f"Stale/incompatible baseline metadata {key}: {metadata_path}")
    current_sources = {
        name: file_identity(hic_source(cfg, "train", accession))
        for _row, accession, name in load_biosample_table(cfg.biosamples_csv)
        if name in set(cfg.train_biosamples)
    }
    recorded_sources = {item["name"]: item["source"] for item in meta.get("biosamples", [])}
    if recorded_sources != current_sources:
        raise ValueError(f"Baseline source Hi-C files changed: {metadata_path}")
    return value, required["baseline_sha256"]


def configure_precision() -> None:
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)


def load_model(args, cfg, device: str):
    module = VariantLit.load_from_checkpoint(
        str(args.checkpoint),
        model=BiasModel(seq_input_dim=cfg.embed_dim, n_cols=cfg.n_cols,
                        output_activation=cfg.output_activation, **cfg.model),
        strict=True,
        map_location="cpu",
    )
    model = module.model.float().eval().to(device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if model.output_activation != "softplus":
        raise ValueError("Loaded model is not the paper Softplus variant")
    return model


def signature(args, cfg, cells: pd.DataFrame, baseline_hash: str) -> dict:
    return {
        "run_name": str(cfg.run_name),
        "checkpoint_sha256": sha256(args.checkpoint),
        "training_config_sha256": sha256(args.config),
        "test_bedpe_sha256": sha256(cfg.test_bedpe),
        "biosamples_manifest_sha256": sha256(cfg.biosamples_csv),
        "test_sequence": file_identity(cfg.test_seq),
        "traincell_test_labels": file_identity(Path(cfg.test_label).with_name("traincell_testgene.npy")),
        "heldout2_test_labels": file_identity(cfg.test_label),
        "hic_sources": [
            file_identity(hic_source(cfg, "test", cell.accession))
            for cell in cells.itertuples(index=False)
        ],
        "producer_sha256": sha256(__file__),
        "baseline_builder_sha256": sha256(Path(__file__).with_name("build_ig_baseline.py")),
        "puget_source_sha256": {
            path.name: sha256(path) for path in sorted((ROOT / "puget").glob("*.py"))
        },
        "method": "integrated_gradients",
        "attributed_input": "Hi-C only; observed AlphaGenome embedding held fixed",
        "n_steps": args.steps,
        "integration_method": "gausslegendre",
        "multiply_by_inputs": True,
        "hic_baseline": "paper-train14-distance-mean",
        "hic_baseline_sha256": baseline_hash,
        "precision": "float32; no autocast; TF32 disabled; math SDP",
        "captum_version": captum.__version__,
        "torch_version": torch.__version__,
    }


def convergence_summary(delta: np.ndarray, effect: np.ndarray) -> dict:
    result = {
        "abs_delta_median": float(np.median(np.abs(delta))),
        "abs_delta_p95": float(np.percentile(np.abs(delta), 95)),
        "abs_delta_max": float(np.max(np.abs(delta))),
    }
    for floor in (0.1, 0.5):
        mask = np.abs(effect) > floor
        relative = np.abs(delta[mask]) / np.abs(effect[mask])
        result[f"effect_gt_{floor}"] = {
            "n": int(mask.sum()),
            "relative_delta_median": float(np.median(relative)) if mask.any() else None,
            "relative_delta_p95": float(np.percentile(relative, 95)) if mask.any() else None,
        }
    return result


def attribute_batch(
    ig: IntegratedGradients,
    image: torch.Tensor,
    sequence: torch.Tensor,
    baseline: torch.Tensor,
    steps: int,
    internal_batch_size: int,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    image = image.detach()
    sequence = sequence.detach()
    reference = baseline.expand_as(image)
    with torch.no_grad():
        pred_input = ig.forward_func(image, sequence).flatten()
        pred_baseline = ig.forward_func(reference, sequence).flatten()
    attrs, captum_delta = ig.attribute(
        inputs=image,
        baselines=reference,
        additional_forward_args=(sequence,),
        target=0,
        n_steps=steps,
        method="gausslegendre",
        internal_batch_size=internal_batch_size,
        return_convergence_delta=True,
    )
    maps = attrs.detach()[:, 0].float().cpu().numpy()
    values = {
        "pred_input": pred_input.detach().float().cpu().numpy(),
        "pred_baseline": pred_baseline.detach().float().cpu().numpy(),
        "hic_attr_sum": maps.sum(axis=(1, 2), dtype=np.float64),
        "captum_delta": captum_delta.detach().float().cpu().numpy().reshape(-1),
    }
    if not np.isfinite(maps).all() or not all(np.isfinite(value).all() for value in values.values()):
        raise FloatingPointError("Nonfinite attribution, prediction, or completeness scalar")
    values["effect_signed"] = (
        values["pred_input"].astype(np.float64) - values["pred_baseline"].astype(np.float64)
    )
    values["delta"] = values["hic_attr_sum"] - values["effect_signed"]
    values["completeness_flag"] = np.abs(values["delta"]) > (
        COMPLETENESS_ATOL + COMPLETENESS_RTOL * np.abs(values["effect_signed"])
    )
    return maps, values


def run_root(args) -> Path:
    return args.output / f"paper-train14-distance-mean_steps{args.steps}"


def cell_dir(root: Path, cell) -> Path:
    return root / f"{int(cell.manifest_row):02d}_{cell.accession}"


def extract_cell(args, ig, cell, cfg, genes, labels, sequence, baseline, expected_signature):
    destination = cell_dir(run_root(args), cell)
    partial = destination.with_name(destination.name + ".partial")
    if destination.exists() or partial.exists():
        raise FileExistsError(f"Refusing to overwrite {destination} or {partial}")
    partial.mkdir(parents=True)
    source = hic_source(cfg, "test", cell.accession)
    started = time.monotonic()
    with source.open("rb") as handle:
        windows = pickle.load(handle)
    attrs = np.lib.format.open_memmap(
        partial / ATTR_FILE,
        mode="w+",
        dtype=np.float32,
        shape=(len(genes), N_BINS, N_BINS),
    )
    scalar_batches = []
    torch.cuda.reset_peak_memory_stats(args.device)
    last_report = time.monotonic()
    for start in range(0, len(genes), args.batch_size):
        stop = min(start + args.batch_size, len(genes))
        images = []
        for gene in genes.iloc[start:stop].itertuples(index=False):
            image, status = transformed_map(windows.get(gene.window_key))
            if image is None:
                raise ValueError(
                    f"{cell.biosample}, gene {gene.gene_index}: {status}; all test genes are required"
                )
            images.append(image)
        image_tensor = torch.from_numpy(np.stack(images)[:, None]).to(args.device)
        sequence_tensor = torch.from_numpy(
            np.asarray(sequence[start:stop], dtype=np.float32)
        ).to(args.device)
        maps, scalars = attribute_batch(
            ig, image_tensor, sequence_tensor, baseline, args.steps, args.internal_batch_size
        )
        attrs[start:stop] = maps
        scalar_batches.append(pd.DataFrame(scalars))
        del maps, image_tensor, sequence_tensor
        if time.monotonic() - last_report >= 20 or stop == len(genes):
            elapsed = time.monotonic() - started
            eta = (len(genes) - stop) * elapsed / stop / 60
            print(
                f"{cell.biosample}: {stop}/{len(genes)}; {elapsed / 60:.1f} min elapsed; "
                f"cell ETA {eta:.1f} min",
                flush=True,
            )
            last_report = time.monotonic()
    attrs.flush()
    del attrs, windows
    gc.collect()

    frame = genes.copy()
    frame.insert(0, "array_row", np.arange(len(genes), dtype=np.int64))
    frame["manifest_row"] = int(cell.manifest_row)
    frame["label_row"] = int(cell.label_row)
    frame["biosample"] = cell.biosample
    frame["accession"] = cell.accession
    frame["model_split"] = cell.model_split
    frame["y"] = labels[cell.model_split][int(cell.label_row)]
    frame = pd.concat([frame, pd.concat(scalar_batches, ignore_index=True)], axis=1)
    frame.to_csv(partial / "scalars.tsv", sep="\t", index=False)
    write_json(
        partial / "meta.json",
        {
            "completed": True,
            "signature": expected_signature,
            "n_examples": len(genes),
            "biosample": cell.biosample,
            "accession": cell.accession,
            "manifest_row": int(cell.manifest_row),
            "label_row": int(cell.label_row),
            "model_split": cell.model_split,
            "shape": [len(genes), N_BINS, N_BINS],
            "dtype": "float32",
            "delta_definition": "sum(saved Hi-C attrs) - (pred_input - pred_baseline)",
            "completeness": convergence_summary(
                frame.delta.to_numpy(), frame.effect_signed.to_numpy()
            ),
            "completeness_flag_threshold": {
                "atol": COMPLETENESS_ATOL,
                "rtol": COMPLETENESS_RTOL,
            },
            "n_completeness_flagged": int(frame.completeness_flag.sum()),
            "scalars_sha256": sha256(partial / "scalars.tsv"),
            "elapsed_sec": time.monotonic() - started,
            "device": args.device,
            "gpu": torch.cuda.get_device_name(args.device),
            "peak_gpu_gib": torch.cuda.max_memory_allocated(args.device) / 2**30,
            "batch_size": args.batch_size,
            "internal_batch_size": args.internal_batch_size,
            "input_semantics": "log10(SCALE O/E + 1)",
            "bin_bp": 1024,
            "axis_order": ["gene_index", "Hi-C row", "Hi-C column"],
            "full_window_extent_kb": [-262.144, 262.144],
        },
    )
    partial.rename(destination)
    print(
        f"Saved {destination}; completeness flags: {int(frame.completeness_flag.sum())}/{len(genes)}",
        flush=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG,
                        help="Puget training YAML in configs/training")
    parser.add_argument("--hic-baseline-file", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--biosamples", default="all", help="all, seen, unseen, or comma-separated names")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--internal-batch-size", type=int, default=4)
    args = parser.parse_args()
    if args.steps < 2 or args.batch_size < 1 or args.internal_batch_size < args.batch_size:
        parser.error("Need steps >= 2 and internal-batch-size >= batch-size >= 1")
    for name in ("output", "checkpoint", "config", "hic_baseline_file"):
        setattr(args, name, getattr(args, name).resolve())
    return args


def main() -> int:
    args = parse_args()
    cfg, genes, all_cells, labels = validate_static_inputs(args)
    cells = select_cells(all_cells, args.biosamples)
    baseline_array, baseline_hash = validate_baseline(args, cfg)
    expected_signature = signature(args, cfg, all_cells, baseline_hash)
    n_examples = len(genes) * len(cells)
    payload = n_examples * N_BINS * N_BINS * 4
    print(
        f"{len(genes)} genes x {len(cells)} biosamples = {n_examples:,} examples; "
        f"steps={args.steps}; output payload={payload / 2**30:.2f} GiB",
        flush=True,
    )
    print("Cells: " + ", ".join(cells.biosample), flush=True)
    print(f"Baseline: {args.hic_baseline_file} ({baseline_hash})", flush=True)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable: run on a GPU host")
    if not 0 <= args.gpu < torch.cuda.device_count():
        raise ValueError(f"Requested GPU {args.gpu} is unavailable")
    args.device = f"cuda:{args.gpu}"
    args.output.mkdir(parents=True, exist_ok=True)
    pending = []
    for cell in cells.itertuples(index=False):
        destination = cell_dir(run_root(args), cell)
        if destination.exists():
            print(f"Reusing completed {destination}", flush=True)
        elif destination.with_name(destination.name + ".partial").exists():
            raise FileExistsError(f"Inspect interrupted output: {destination}.partial")
        else:
            pending.append(cell)
    required = len(pending) * len(genes) * N_BINS * N_BINS * 4
    if shutil.disk_usage(args.output).free < required + 2 * 2**30:
        raise OSError("Insufficient output space, including the 2-GiB worker reserve")
    if pending:
        configure_precision()
        model = load_model(args, cfg, args.device)
        ig = IntegratedGradients(model, multiply_by_inputs=True)
        baseline = torch.from_numpy(baseline_array)[None, None].to(args.device)
        sequence = np.load(cfg.test_seq, mmap_mode="r")
        print(f"Using {args.device}: {torch.cuda.get_device_name(args.device)}", flush=True)
        for cell in pending:
            extract_cell(
                args, ig, cell, cfg, genes, labels, sequence, baseline,
                expected_signature,
            )

    summaries = [
        read_json(cell_dir(run_root(args), cell) / "meta.json")
        for cell in cells.itertuples(index=False)
    ]
    manifest_name = (
        "meta.json"
        if len(cells) == len(all_cells)
        else "meta.cells_" + "-".join(str(int(value)) for value in sorted(cells.manifest_row)) + ".json"
    )
    write_json(
        run_root(args) / manifest_name,
        {
            "completed": True,
            "signature": expected_signature,
            "n_examples": n_examples,
            "array_payload_bytes": payload,
            "n_completeness_flagged": sum(item["n_completeness_flagged"] for item in summaries),
            "max_abs_completeness_delta": max(
                item["completeness"]["abs_delta_max"] for item in summaries
            ),
            "biosamples": [
                {
                    "directory": cell_dir(run_root(args), cell).name,
                    "biosample": cell.biosample,
                    "manifest_row": int(cell.manifest_row),
                    "model_split": cell.model_split,
                }
                for cell in cells.itertuples(index=False)
            ],
        },
    )
    print(f"Completed: {run_root(args)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
