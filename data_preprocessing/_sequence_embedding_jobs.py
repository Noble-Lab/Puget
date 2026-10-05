from __future__ import annotations

import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable


def run_jobs(
    args,
    *,
    script: Path,
    model: str,
    marker_prefix: str,
    bed_prefix: str,
    layers: list[str],
    paths: Callable[[Path, str, str], tuple[Path, Path]],
    allocate: Callable,
    model_options: list[str],
) -> None:
    gpu_ids = args.gpu_ids
    if not gpu_ids or any(gpu < 0 for gpu in gpu_ids) or len(set(gpu_ids)) != len(gpu_ids):
        raise ValueError("--gpu-ids requires distinct, nonnegative CUDA device IDs")
    if len(set(args.splits)) != len(args.splits):
        raise ValueError("--splits must not repeat a split")
    if args.batch_size < 1 or args.num_workers < 0 or (args.limit is not None and args.limit < 1):
        raise ValueError("batch size and limit must be positive; workers must be nonnegative")
    if not args.fasta.is_file():
        raise FileNotFoundError(args.fasta)

    # Check every input and output before allocating a large, zero-filled array.
    requests = []
    for split in args.splits:
        bed = args.bed_dir / f"{bed_prefix}_{split}.bed"
        if not bed.is_file() or bed.stat().st_size == 0:
            raise FileNotFoundError(bed)
        outputs = [path for layer in layers for path in paths(args.out_root, layer, split)]
        if not args.overwrite:
            existing = [path for path in outputs if path.exists() or path.is_symlink()]
            if existing:
                raise FileExistsError(f"{existing[0]} exists; pass --overwrite to replace it")
        requests.append((split, bed, outputs))

    for gpu in gpu_ids:
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
        check = subprocess.run(
            [sys.executable, "-c", "import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)"],
            env=env, check=False,
        )
        if check.returncode:
            raise RuntimeError(f"CUDA device {gpu} is unavailable in {sys.executable}")

    log_dir = args.out_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    for split, bed, outputs in requests:
        args.bed, args.split = bed, split
        # A prior interrupted run may have left markers from an incomplete array.
        marker_pattern = f".{marker_prefix}_{split}.shard*.done.json"
        for marker in outputs[0].parent.glob(marker_pattern):
            marker.unlink()

        running: list[tuple[subprocess.Popen, Path]] = []
        try:
            allocate(args)
            for shard_id, gpu in enumerate(gpu_ids):
                log = log_dir / f"{model}_{split}_shard{shard_id}_{stamp}.log"
                command = [
                    sys.executable, str(script), "--mode", "shard",
                    "--bed", str(bed), "--split", split,
                    "--fasta", str(args.fasta), "--out-root", str(args.out_root),
                    "--layers", args.layers, "--dtype", args.dtype,
                    "--num-shards", str(len(gpu_ids)), "--shard-id", str(shard_id),
                    "--batch-size", str(args.batch_size), "--num-workers", str(args.num_workers),
                    *model_options,
                ]
                if args.limit is not None:
                    command += ["--limit", str(args.limit)]
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
                with log.open("w") as output:
                    process = subprocess.Popen(command, env=env, stdout=output, stderr=subprocess.STDOUT)
                running.append((process, log))
                print(f"{model} {split}: GPU {gpu}, shard {shard_id}/{len(gpu_ids)} -> {log}", flush=True)
            failures = []
            for process, log in running:
                if process.wait():
                    failures.append(log)
            if failures:
                raise RuntimeError("Embedding shard failed; inspect " + ", ".join(map(str, failures)))
        except BaseException:
            for process, _ in running:
                if process.poll() is None:
                    process.terminate()
            for process, _ in running:
                process.wait()
            for path in outputs:
                path.unlink(missing_ok=True)
            for marker in outputs[0].parent.glob(marker_pattern):
                marker.unlink()
            raise
        print(f"{model} {split}: complete", flush=True)
