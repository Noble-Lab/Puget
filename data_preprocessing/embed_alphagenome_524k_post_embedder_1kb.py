#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from numpy.lib.format import open_memmap
from pyfaidx import Fasta
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from alphagenome_pytorch import AlphaGenome
from _sequence_embedding_jobs import run_jobs

REPO_ROOT = Path(__file__).resolve().parents[1]

SEQ_LENGTH = 524288
BIN_1KB = 1024
BINS_1KB = SEQ_LENGTH // BIN_1KB   # 512, matching the 512x512 Hi-C grid
ORGANISM_HUMAN = 0

MODEL_DIR = "alphagenome_embeddings_524k_1kb_mean"   # one dir per model

# native bin width -> pooling group, per tap
LAYER_SPEC = {"post_embedder": {"dim": 3072, "bin_bp": 128, "group": 8}}
LAYERS_1BP = set()

BASE_TO_INDEX = {"A": 0, "C": 1, "G": 2, "T": 3}
RC_INDEX = np.array([3, 2, 1, 0], dtype=np.int64)


def read_bed(path: Path, limit: int | None = None):
    rows = []
    with open(path) as handle:
        for line in handle:
            if not line.strip() or line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            chrom, start, end = fields[:3]
            name = fields[3] if len(fields) >= 4 else f"row{len(rows)}"
            strand = fields[5] if len(fields) >= 6 else "+"
            rows.append((chrom, int(start), int(end), name, strand))
    if not rows:
        raise ValueError(f"No rows read from {path}")
    if limit is not None:
        rows = rows[:limit]
    return rows


def one_hot_from_region(fasta: Fasta, row):
    chrom, start, end, name, strand = row
    if end - start != SEQ_LENGTH:
        raise ValueError(f"{name} window length {end - start} != {SEQ_LENGTH}")

    seq = str(fasta[chrom][start:end]).upper()
    if len(seq) != SEQ_LENGTH:
        raise ValueError(f"{name} fetched sequence length {len(seq)} != {SEQ_LENGTH}")

    arr = np.zeros((SEQ_LENGTH, 4), dtype=np.float32)
    idx = np.frombuffer(seq.encode(), dtype=np.uint8)
    for base, j in BASE_TO_INDEX.items():
        arr[idx == ord(base), j] = 1.0
    # Ambiguous bases (N) stay all-zero, matching every upstream encoder:
    #   Enformer  one_hot_embed[N] = [0,0,0,0]   (0.25 is reserved for ".")
    #   AlphaGenome sequence_to_onehot: N/other -> all-zeros, per the JAX reference
    #   Borzoi    baskerville dna_1hot default (n_uniform=False)
    # Using 0.25 here instead diverges badly on N-containing windows: on the
    # Enformer test split, gene row 112 (19,000 N) fell to corr 0.918 / max|d| 6.68
    # against the archived embeddings, while N-free genes matched at 0.99999995.

    if strand == "-":
        arr = arr[::-1][:, RC_INDEX].copy()
    # NLC: AlphaGenome takes (B, S, 4)
    return torch.from_numpy(arr)


class BedShardDataset(Dataset):
    def __init__(self, rows, fasta_path: Path, start: int, end: int):
        self.rows = rows
        self.start = start
        self.end = end
        self.fasta_path = fasta_path
        self.fasta = None

    def __len__(self):
        return self.end - self.start

    def __getitem__(self, i):
        if self.fasta is None:
            self.fasta = Fasta(str(self.fasta_path), as_raw=True, sequence_always_upper=True)
        global_idx = self.start + i
        return one_hot_from_region(self.fasta, self.rows[global_idx]), global_idx


def shard_bounds(n: int, num_shards: int, shard_id: int):
    bounds = np.linspace(0, n, num_shards + 1, dtype=int)
    return int(bounds[shard_id]), int(bounds[shard_id + 1])


def pool_1kb_ncl(emb_ncl: torch.Tensor, group: int) -> torch.Tensor:
    b, c, bins = emb_ncl.shape
    if bins != BINS_1KB * group:
        raise ValueError(f"expected {BINS_1KB * group} positions for group {group}, got {bins}")
    return emb_ncl.float().reshape(b, c, BINS_1KB, group).mean(dim=3).transpose(1, 2)


def layer_dir(out_root: Path, layer: str) -> Path:
    return out_root / MODEL_DIR


def layer_prefix(layer: str) -> str:
    return f"human_seq_embeddings_alphagenome_fold1_524k_{layer}_1kb_mean"


def paths(out_root: Path, layer: str, split: str):
    d = layer_dir(out_root, layer)
    prefix = layer_prefix(layer)
    return d / f"{prefix}_{split}.npy", d / f"{prefix}_{split}.metadata.json"


def parse_layers(spec: str):
    if spec not in {"all", "post_embedder"}:
        raise ValueError("this standalone extractor only writes post_embedder")
    return ["post_embedder"]


def _pkg_version():
    try:
        from importlib.metadata import version
        return version("alphagenome-pytorch")
    except Exception:
        return "unknown"


def allocate(args):
    rows = read_bed(args.bed, args.limit)
    for layer in parse_layers(args.layers):
        spec = LAYER_SPEC[layer]
        emb_path, meta_path = paths(args.out_root, layer, args.split)
        emb_path.parent.mkdir(parents=True, exist_ok=True)

        arr = open_memmap(emb_path, mode="w+", dtype=args.dtype,
                          shape=(len(rows), BINS_1KB, spec["dim"]))
        arr.flush()
        del arr

        meta = {
            "bed": str(args.bed),
            "split": args.split,
            "num_rows": len(rows),
            "limit": args.limit,
            "seq_length": SEQ_LENGTH,
            "bin_bp": BIN_1KB,
            "shape": [len(rows), BINS_1KB, spec["dim"]],
            "embedding_layer": layer,
            "native_bin_bp": spec["bin_bp"],
            "pool": f"mean over {spec['group']} x {spec['bin_bp']}bp bins",
            "pool_accumulation_dtype": "float32",
            "checkpoint": args.checkpoint,
            "model_version": "FOLD_1",
            "alphagenome_pytorch_version": _pkg_version(),
            "organism_index": ORGANISM_HUMAN,
            "channels_last": args.channels_last,
            "ambiguous_base_encoding": "N -> [0,0,0,0] (upstream convention)",
            "augmentation": "none; minus-strand genes RC'd by strand",
            "compute_dtype": "float32 (TF32 matmul)",
            "dtype": args.dtype,
            "npy": str(emb_path),
        }
        meta_path.write_text(json.dumps(meta, indent=2) + "\n")
        print(json.dumps(meta, indent=2))


def load_model(args, device):
    # Default DtypePolicy is full float32, matching how the Enformer/Borzoi
    # embedding sets were generated. Do NOT switch to mixed precision: at
    # batch_size=1 this peaks at ~18 GiB allocated on an 80 GiB card, so there
    # is no memory pressure to relieve, and bf16 would make AlphaGenome the only
    # arm at reduced precision.
    model = AlphaGenome.from_pretrained(args.checkpoint, device=device)
    model.eval()
    return model


def run_shard(args):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for embedding generation")
    if not (0 <= args.shard_id < args.num_shards):
        raise ValueError("shard_id must be in [0, num_shards)")

    layers = parse_layers(args.layers)
    want = set(layers)
    need_1bp = bool(want & LAYERS_1BP)
    resolutions = (1, 128) if need_1bp else (128,)

    rows = read_bed(args.bed, args.limit)
    start, end = shard_bounds(len(rows), args.num_shards, args.shard_id)
    if start == end:
        print(f"[shard {args.shard_id}] empty shard")
        return

    memmaps = {}
    for layer in layers:
        emb_path, _ = paths(args.out_root, layer, args.split)
        arr = open_memmap(emb_path, mode="r+")
        expected = (len(rows), BINS_1KB, LAYER_SPEC[layer]["dim"])
        assert arr.shape == expected, f"{layer}: {arr.shape} != {expected}"
        memmaps[layer] = arr

    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_grad_enabled(False)
    torch.cuda.set_device(0)
    torch.cuda.reset_peak_memory_stats(device)

    ds = BedShardDataset(rows, args.fasta, start, end)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True, drop_last=False)

    model = load_model(args, device)

    # Neither pre-embedder tensor is returned by encode(); grab both on the way in.
    captured = {}
    hooks = []
    if "trunk" in want:
        hooks.append(model.embedder_128bp.register_forward_pre_hook(
            lambda mod, a: captured.__setitem__("trunk", a[0])))       # (B, 1536, 4096)
    if "decoder_raw" in want:
        hooks.append(model.embedder_1bp.register_forward_pre_hook(
            lambda mod, a: captured.__setitem__("decoder_raw", a[0]))) # (B, 768, 524288)

    t0 = time.perf_counter()
    ptr = start
    desc = f"{args.split}:{'+'.join(layers)}:shard{args.shard_id}"
    with torch.no_grad():
        for seqs, global_idx in tqdm(loader, desc=desc):
            assert int(global_idx[0]) == ptr, f"expected row {ptr}, got {int(global_idx[0])}"
            seqs = seqs.to(device, non_blocking=True)

            out = model.encode(seqs, ORGANISM_HUMAN, resolutions=resolutions,
                               channels_last=args.channels_last)

            taps = {}
            if "post_embedder" in want:
                taps["post_embedder"] = out["embeddings_128bp"]        # (B, 3072, 4096) NCL
            if "post_1bp_embedder" in want:
                taps["post_1bp_embedder"] = out["embeddings_1bp"]      # (B, 1536, 524288) NCL
            for k in ("trunk", "decoder_raw"):
                if k in want:
                    taps[k] = captured[k]

            b = None
            for layer in layers:
                pooled = pool_1kb_ncl(taps[layer], LAYER_SPEC[layer]["group"])
                pooled = pooled.cpu().numpy().astype(args.dtype)
                b = pooled.shape[0]
                memmaps[layer][ptr:ptr + b] = pooled
            ptr += b
            captured.clear()
            del taps, out

    for h in hooks:
        h.remove()
    assert ptr == end, f"shard {args.shard_id} wrote up to {ptr}, expected {end}"
    for arr in memmaps.values():
        arr.flush()

    elapsed = time.perf_counter() - t0
    peak_gb = torch.cuda.max_memory_allocated(device) / 1e9
    peak_res_gb = torch.cuda.max_memory_reserved(device) / 1e9
    done_path = layer_dir(args.out_root, layers[0]) / \
        f".alphagenome_fold1_524k_{args.split}.shard{args.shard_id}.done.json"
    done_path.write_text(json.dumps({
        "split": args.split,
        "layers": layers,
        "resolutions": list(resolutions),
        "channels_last": args.channels_last,
        "seq_length": SEQ_LENGTH,
        "shard_id": args.shard_id,
        "num_shards": args.num_shards,
        "start": start,
        "end": end,
        "rows": end - start,
        "elapsed_seconds": elapsed,
        "seconds_per_gene": elapsed / max(end - start, 1),
        "peak_cuda_allocated_gb": peak_gb,
        "peak_cuda_reserved_gb": peak_res_gb,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }, indent=2) + "\n")
    print(f"[shard {args.shard_id}] wrote rows [{start}, {end}) in {elapsed:.1f}s; "
          f"peak={peak_gb:.2f}GB alloc / {peak_res_gb:.2f}GB reserved")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=("run", "allocate", "shard"), default="run")
    p.add_argument("--bed", type=Path)
    p.add_argument("--bed-dir", type=Path, default=REPO_ROOT / "outputs/borzoi_data")
    p.add_argument("--bed-prefix", default="Borzoi_genes_524k")
    p.add_argument("--fasta", type=Path,
                   default=REPO_ROOT / "data/hg38.fa")
    p.add_argument("--out-root", type=Path, default=REPO_ROOT / "outputs/seq_embeddings")
    p.add_argument("--split", choices=("train", "valid", "test"))
    p.add_argument("--splits", nargs="+", choices=("train", "valid", "test"),
                   default=("valid", "test", "train"))
    p.add_argument("--gpu-ids", nargs="+", type=int, metavar="GPU")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--layers", default="all",
                   help="'all' or 'post_embedder'; only the selected layer is written")
    p.add_argument("--checkpoint", default=str(REPO_ROOT / "data/model_fold_1.safetensors"))
    p.add_argument("--dtype", choices=("float16", "float32"), default="float16")
    p.add_argument("--channels-last", action="store_true", default=False,
                   help="default False; True adds a full transpose+contiguous copy "
                        "of the (B,1536,524288) 1bp tensor for no benefit")
    p.add_argument("--num-shards", type=int, default=2)
    p.add_argument("--shard-id", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=1,
                   help="1 is optimal: bs=2 is 2%% faster for 26 GiB more reserved, bs=4 OOMs")
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--limit", type=int, default=None,
                   help="use only the first N BED rows (smoke tests)")
    args = p.parse_args()

    if args.mode == "run":
        if not args.gpu_ids:
            p.error("--gpu-ids is required in run mode")
        if not Path(args.checkpoint).is_file():
            p.error(f"checkpoint not found: {args.checkpoint}")
        run_jobs(
            args, script=Path(__file__).resolve(), model="alphagenome",
            marker_prefix="alphagenome_fold1_524k", bed_prefix=args.bed_prefix,
            layers=parse_layers(args.layers), paths=paths, allocate=allocate,
            model_options=["--checkpoint", args.checkpoint]
                          + (["--channels-last"] if args.channels_last else []),
        )
    elif args.mode == "allocate":
        if args.bed is None or args.split is None:
            p.error("--bed and --split are required in allocate mode")
        allocate(args)
    else:
        if args.bed is None or args.split is None:
            p.error("--bed and --split are required in shard mode")
        run_shard(args)


if __name__ == "__main__":
    main()
