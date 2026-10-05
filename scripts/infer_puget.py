from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from scripts._puget_config import load_config
from puget.puget_data import build_split_loader, load_biosample_table, resolve_subset
from puget.puget_model import BiasModel, VariantLit
from scripts.train_puget import resolve_inputs


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

    inference = load_config(args.config)
    cfg = load_config(str(ROOT / inference.training_config))
    resolve_inputs(cfg)
    # Match the archived training and prediction entry points, rather than
    # inheriting a fresh process's default float32 matmul precision.
    matmul_precision = str(cfg.get("float32_matmul_precision", "high"))
    torch.set_float32_matmul_precision(matmul_precision)
    if str(cfg.hic_semantics).lower() != "oe_scale":
        raise ValueError("Puget inference requires O/E SCALE Hi-C windows")
    cfg.batch_size = int(inference.batch_size)
    cfg.num_workers = int(inference.num_workers)
    run = str(cfg.run_name)
    checkpoint = ROOT / inference.checkpoint
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if inference.group not in ("train14", "test2", "both"):
        raise ValueError("group must be train14, test2, or both")
    groups = ("train14", "test2") if inference.group == "both" else (inference.group,)
    table = load_biosample_table(cfg.biosamples_csv)
    destination = ROOT / inference.output_dir
    for group in groups:
        if group == "train14":
            cfg.test_label = str(Path(cfg.test_label).with_name("traincell_testgene.npy"))
            cfg.test_label_group = "traincell"
            names = cfg.train_biosamples
            stem = f"{run}_testgenes_train14"
            suffix = "_pred"
            biosample_suffix = "_biosamples"
        else:
            cfg.test_label = str(Path(cfg.test_label).with_name("testcell_testgene.npy"))
            cfg.test_label_group = "testcell"
            names = cfg.test_biosamples
            stem = run
            suffix = "_testpred"
            biosample_suffix = "_test_biosamples"
        loaded_names, _rows, accessions = resolve_subset(names, table)
        if tuple(loaded_names) != tuple(cfg.train_biosamples if group == "train14" else cfg.test_biosamples):
            raise ValueError("Biosample order differs from training config")
        with Path(cfg.test_bedpe).open() as handle:
            n_genes = sum(bool(line.strip()) for line in handle)
        labels = np.load(cfg.test_label, mmap_mode="r")
        sequence = np.load(cfg.test_seq, mmap_mode="r")
        if labels.shape != (len(loaded_names), n_genes) or labels.dtype != np.float32:
            raise ValueError(f"{cfg.test_label}: wrong label shape or dtype")
        if sequence.shape != (n_genes, int(cfg.n_cols), int(cfg.embed_dim)) or sequence.dtype != np.float16:
            raise ValueError(f"{cfg.test_seq}: wrong sequence shape or dtype")
        missing = [str(path) for accession in accessions
                   if not (path := Path(cfg.hic_root) / "test" / f"{accession}.pkl").is_file()
                   or path.stat().st_size == 0]
        if missing:
            raise FileNotFoundError(f"Missing {len(missing)} test O/E Hi-C windows; first: {missing[0]}")
        prediction_path = destination / f"{stem}{suffix}.npy"
        names_path = destination / f"{stem}{biosample_suffix}.txt"
        metadata_path = destination / f"{stem}_metadata.json"
        existing = [p for p in (prediction_path, names_path, metadata_path) if p.exists()]
        if existing and not args.overwrite and not args.preflight_only:
            raise FileExistsError(f"Refusing to overwrite: {existing}")
        print(f"{group}: {len(loaded_names)} biosamples x {n_genes} genes -> {prediction_path}")
        if args.preflight_only:
            continue
        loader, loaded_names = build_split_loader(
            cfg, "test", cfg.test_bedpe, cfg.test_seq, cfg.test_label, names, table, False)
        for accession, windows in zip(accessions, loader.dataset.windows):
            first = next(iter(windows.values()), None)
            if first is None or str(first.get("source_semantics", "")).lower() != "oe_scale":
                raise ValueError(f"{accession}: expected O/E SCALE test windows")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; inference requires a GPU")
        device = torch.device("cuda")
        model = VariantLit.load_from_checkpoint(
            str(checkpoint),
            model=BiasModel(seq_input_dim=cfg.embed_dim, n_cols=cfg.n_cols,
                            output_activation=cfg.output_activation, **cfg.model),
            map_location="cpu",
        ).eval().to(device)
        predictions = np.full((len(loaded_names), loader.dataset.n_pairs), np.nan, dtype=np.float32)
        precision = str(cfg.precision)
        use_amp = "mixed" in precision
        amp_dtype = torch.bfloat16 if "bf16" in precision else torch.float16
        with torch.inference_mode():
            for image, sequence, _target, metadata in loader:
                with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                    output = model(image.to(device, non_blocking=True),
                                   sequence.to(device, non_blocking=True))
                for value, meta in zip(output.float().cpu().numpy().reshape(-1), metadata):
                    predictions[int(meta[5]), int(meta[6])] = float(value)
        finite = np.isfinite(predictions)
        if not finite.any(axis=1).all() or np.isinf(predictions).any() or (predictions[finite] < 0).any():
            raise RuntimeError("Invalid Softplus prediction grid")
        destination.mkdir(parents=True, exist_ok=True)
        temporary = prediction_path.with_suffix(".tmp.npy")
        np.save(temporary, predictions)
        os.replace(temporary, prediction_path)
        names_path.write_text("\n".join(loaded_names) + "\n")
        metadata_path.write_text(json.dumps({
            "run_name": run, "config": str(Path(args.config).resolve()),
            "checkpoint": str(checkpoint),
            "gene_split": "test", "biosample_group": group, "biosamples": loaded_names,
            "prediction_layout": "biosample_x_test_gene_in_bedpe_order",
            "prediction_shape": list(predictions.shape), "prediction_dtype": "float32",
            "test_bedpe": str(cfg.test_bedpe), "test_sequence": str(cfg.test_seq),
            "test_label": str(cfg.test_label), "hic_root": str(cfg.hic_root),
            "hic_semantics": str(cfg.hic_semantics), "output_activation": str(cfg.output_activation),
            "precision": str(cfg.precision), "float32_matmul_precision": matmul_precision,
            "n_finite": int(finite.sum()), "n_nan": int(np.isnan(predictions).sum()),
            "dropped_hic_windows": loader.dataset.dropped,
        }, indent=2) + "\n")
        print(f"Saved {prediction_path}: shape={predictions.shape}, NaN={int(np.isnan(predictions).sum())}")
        del model, loader
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
