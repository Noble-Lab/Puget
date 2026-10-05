from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import yaml

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import rankdata

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
_THREADS = os.environ.setdefault("OMP_NUM_THREADS", "2")
for _v in ("MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, _THREADS)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from puget.seq_only_data import (DEFAULT_EMBED_ROOT, SplitData, choose_device,  # noqa: E402
                                 discover_embeddings, load_labels, select_biosamples)
from puget.seq_only_model import build_model, count_parameters               # noqa: E402

DEFAULT_LRS = (1e-5,)
DEFAULT_INPUT_NORMS = ("layernorm",)
DEFAULT_OUTPUT_ACTIVATIONS = ("softplus",)
NORM_TAG = {"layernorm": "ln", "none": "noln"}


# Validation metrics for gene-major prediction arrays.
def hvg_gene_mask(labels: np.ndarray, top_frac: float = 0.25) -> np.ndarray:
    var = np.nanvar(labels, axis=1)
    mask = np.zeros(var.shape[0], dtype=bool)
    keep = np.where(np.isfinite(var) & (var > 0))[0]
    if keep.size == 0:
        return mask
    k = max(1, int(np.ceil(top_frac * keep.size)))
    mask[keep[np.argsort(var[keep])[::-1][:k]]] = True
    return mask


def _corr_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a.astype(np.float64) - a.astype(np.float64).mean(axis=1, keepdims=True)
    b = b.astype(np.float64) - b.astype(np.float64).mean(axis=1, keepdims=True)
    num = (a * b).sum(axis=1)
    den = np.sqrt((a * a).sum(axis=1) * (b * b).sum(axis=1))
    out = np.full(a.shape[0], np.nan, dtype=np.float64)
    ok = den > 0
    out[ok] = num[ok] / den[ok]
    return out


def _nanmean(x: np.ndarray) -> float:
    finite = x[np.isfinite(x)]
    return float(finite.mean()) if finite.size else float("nan")


def _pearson_spearman(a: np.ndarray, b: np.ndarray):
    if a.shape[0] == 0 or a.shape[1] < 2:
        return float("nan"), float("nan")
    pear = _nanmean(_corr_rows(a, b))
    # Spearman is Pearson on average-tied ranks; rankdata handles ties exactly.
    spear = _nanmean(_corr_rows(rankdata(a, axis=1), rankdata(b, axis=1)))
    return pear, spear


def evaluate(preds: np.ndarray, targets: np.ndarray,
             hvg_mask: Optional[np.ndarray] = None) -> Dict[str, float]:
    if preds.shape != targets.shape:
        raise ValueError(f"shape mismatch: {preds.shape} vs {targets.shape}")
    out: Dict[str, float] = {}
    out["mse"] = float(np.mean((preds - targets) ** 2))
    out["mae"] = float(np.mean(np.abs(preds - targets)))

    # across-gene: one row per biosample
    pear, spear = _pearson_spearman(preds.T, targets.T)
    out["pearson"] = pear
    out["spearman"] = spear

    # across-cell-type: one row per gene, HVG only
    if hvg_mask is not None and bool(np.any(hvg_mask)):
        gp, gs = _pearson_spearman(preds[hvg_mask], targets[hvg_mask])
    else:
        gp = gs = float("nan")
    out["gene_pearson_bio"] = gp
    out["gene_spearman_bio"] = gs
    return out


def run_name(base: str, input_norm: str, output_activation: str, lr: float,
             batch_size: int, seed: int, defaults: dict, prefix: str = "") -> str:
    parts = [base]
    if input_norm != defaults["input_norm"]:
        parts.append(NORM_TAG[input_norm])
    if output_activation != defaults["output_activation"]:
        parts.append(output_activation)
    if lr != defaults["lr"]:
        parts.append(f"lr{lr:.0e}")
    if batch_size != defaults["batch_size"]:
        parts.append(f"bs{batch_size}")
    if seed != defaults["seed"]:
        parts.append(f"seed{seed}")
    return prefix + "_".join(parts)


# Only fields that change the RESULT belong here. Anything absent (output_dir,
# num_workers, data_device, wandb settings, overwrite) can differ between a run
# and its resume without invalidating it.
FINGERPRINT_KEYS = (
    "embedding_key", "embed_dim", "n_tokens", "split_family",
    "input_norm", "output_activation", "lr", "batch_size", "seed", "epochs", "warmup_epochs",
    "min_lr", "weight_decay", "grad_clip", "proj_dim", "num_heads", "num_layers",
    "mlp_ratio", "dropout", "promoter_tokens", "attn", "precision", "max_steps",
)


def fingerprint(cfg: dict, spec) -> Tuple[str, dict]:
    fields = {k: cfg.get(k) for k in FINGERPRINT_KEYS}
    for key in ("biosample_set", "biosample_indices", "biosample_names", "output_dim"):
        fields[key] = cfg.get(key)
    for split in ("train", "valid"):
        for kind, path in (
            ("data", spec.npy[split]),
            ("labels", os.path.join(spec.label_dir, f"traincell_{split}gene.npy")),
        ):
            stat = os.stat(path)
            fields[f"{kind}_{split}"] = [os.path.realpath(path), stat.st_size, stat.st_mtime_ns]
    blob = json.dumps(fields, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:8], fields


def _dedup(seq):
    out = []
    for v in seq:
        if v not in out:
            out.append(v)
    return out


def _cfg_for_fingerprint(args, spec, lr: float, input_norm: str,
                         output_activation: str) -> dict:
    d = dict(vars(args))
    d.update(lr=lr, input_norm=input_norm, output_activation=output_activation,
             embedding_key=spec.key,
             embed_dim=spec.embed_dim, n_tokens=spec.n_tokens,
             split_family=spec.split_family)
    return d


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def lr_lambda(step, warmup_steps, decay_steps, base_lr, min_lr) -> float:
    if warmup_steps > 0 and step < warmup_steps:
        return float(step + 1) / float(warmup_steps)
    t = min(1.0, (step - warmup_steps) / max(1, decay_steps))
    cos = 0.5 * (1.0 + math.cos(math.pi * t))
    return (min_lr + (base_lr - min_lr) * cos) / base_lr


@torch.no_grad()
def predict(model, data: SplitData, amp_dtype, on_gpu: bool) -> np.ndarray:
    model.eval()
    preds = np.zeros((data.n_genes, data.n_bio), dtype=np.float32)
    for x, _y, idx in data.iter_batches(shuffle=False):
        if not on_gpu:
            x = x.cuda(non_blocking=True)
        with torch.autocast("cuda", dtype=amp_dtype, enabled=amp_dtype != torch.float32):
            out = model(x)
        preds[idx.cpu().numpy()] = out.float().cpu().numpy()
    model.train()
    return preds


def make_model(args, spec, input_norm: str, output_activation: str):
    return build_model(
        seq_input_dim=spec.embed_dim, n_tokens=spec.n_tokens, output_dim=args.output_dim,
        proj_dim=args.proj_dim, num_heads=args.num_heads, num_layers=args.num_layers,
        mlp_ratio=args.mlp_ratio, dropout_p=args.dropout,
        promoter_tokens=args.promoter_tokens, input_norm=input_norm,
        attn_impl=args.attn, output_activation=output_activation).cuda()


def train_one(args, spec, lr, input_norm, output_activation, run, dirs, train, valid,
              hvg_valid, on_gpu):
    ckpt_path = os.path.join(dirs["output"], f"{run}_best.pt")

    # Identical init across every cell: LayerNorm's ones/zeros draw nothing from
    # the RNG, so the ln and noln arms share the same random Linear weights.
    set_seed(args.seed)
    model = make_model(args, spec, input_norm, output_activation)
    n_params = count_parameters(model)
    if args.compile:
        model = torch.compile(model, dynamic=False)

    steps_per_epoch = max(1, math.ceil(train.n_genes / args.batch_size))
    warm = steps_per_epoch * args.warmup_epochs
    decay = steps_per_epoch * (args.epochs - args.warmup_epochs)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lr_lambda=lambda s: lr_lambda(s, warm, decay, lr, args.min_lr))
    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16,
                 "fp32": torch.float32}[args.precision]
    scaler = torch.amp.GradScaler("cuda", enabled=(args.precision == "fp16"))
    criterion = nn.MSELoss()

    cfg = dict(vars(args))
    cfg.update(lr=lr, input_norm=input_norm, output_activation=output_activation,
               run=run, embedding_key=spec.key,
               family=spec.family, layer=spec.layer,
               split_family=spec.split_family, n_tokens=spec.n_tokens,
               embed_dim=spec.embed_dim, n_params=n_params,
               data_device="cuda" if on_gpu else "cpu", steps_per_epoch=steps_per_epoch)
    fp, fp_fields = fingerprint(cfg, spec)
    cfg["fingerprint"] = fp

    wb = None
    if not args.no_wandb:
        import wandb
        wb = wandb.init(project=args.wandb_project, entity=args.wandb_entity,
                        name=run, group=spec.key,
                        tags=[spec.family, spec.layer, spec.split_family,
                              f"lr{lr:.0e}", NORM_TAG[input_norm], output_activation],
                        config=cfg, dir=dirs["logs"], reinit="finish_previous")

    print(f"  -- {run}\n     params={n_params/1e6:.3f} M  lr={lr:.1e}  "
          f"input_norm={input_norm}  output_activation={output_activation}  "
          f"steps/epoch={steps_per_epoch}  "
          f"fingerprint={fp}", flush=True)

    gen = torch.Generator().manual_seed(args.seed)
    best = {"val_mse": float("inf"), "epoch": -1}
    history, gstep, nonfinite_total = [], 0, 0
    t_start = time.time()

    for epoch in range(args.epochs):
        model.train()
        t0, run_loss, nb, nonfinite_skipped = time.time(), 0.0, 0, 0
        for x, y, _idx in train.iter_batches(shuffle=True, generator=gen):
            if not on_gpu:
                x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
            with torch.autocast("cuda", dtype=amp_dtype, enabled=args.precision != "fp32"):
                loss = criterion(model(x), y)
            # An unnormalized embedding with |x| ~ 2e4 can overflow the loss.
            # Drop the step rather than poisoning every weight with NaN, and
            # count it so the metrics file says how unstable the run was.
            if not torch.isfinite(loss):
                nonfinite_skipped += 1
                nonfinite_total += 1
                opt.zero_grad(set_to_none=True)
                sched.step()
                gstep += 1
                continue
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            if args.grad_clip > 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(opt)
            scaler.update()
            sched.step()
            run_loss += float(loss.detach()); nb += 1; gstep += 1
            if wb is not None and gstep % 100 == 0:
                wb.log({"train/loss": run_loss / nb, "train/lr": sched.get_last_lr()[0]}, step=gstep)
            if args.max_steps and nb >= args.max_steps:
                break

        vm = evaluate(predict(model, valid, amp_dtype, on_gpu), valid.labels_np, hvg_valid)
        dt = time.time() - t0
        history.append({"epoch": epoch, "train_mse": run_loss / max(nb, 1),
                        "lr": sched.get_last_lr()[0], "seconds": dt,
                        "nonfinite_steps": nonfinite_skipped,
                        **{f"val_{k}": v for k, v in vm.items()}})
        if wb is not None:
            wb.log({"epoch": epoch, "train/epoch_loss": run_loss / max(nb, 1),
                    "train/nonfinite_steps": nonfinite_skipped,
                    **{f"val/{k}": v for k, v in vm.items()}}, step=gstep)
        print(f"     epoch {epoch:3d}  train_mse={run_loss/max(nb,1):.4f}  "
              f"val_mse={vm['mse']:.4f}  val_r={vm['pearson']:.4f}  "
              f"val_r_cell={vm['gene_pearson_bio']:.4f}  {dt:.1f}s"
              + (f"  [{nonfinite_skipped} non-finite steps skipped]"
                 if nonfinite_skipped else ""), flush=True)

        if np.isfinite(vm["mse"]) and vm["mse"] < best["val_mse"]:
            best = {"val_mse": vm["mse"], "epoch": epoch,
                    **{f"val_{k}": v for k, v in vm.items()}}
            sd = (model._orig_mod if hasattr(model, "_orig_mod") else model).state_dict()
            torch.save({"state_dict": sd, "config": cfg, "epoch": epoch, "val": vm}, ckpt_path)

    diverged = best["epoch"] < 0
    if diverged:
        # Never produced a finite validation loss. Save the final weights so the
        # run still has a checkpoint and a metrics file, and flag it.
        print("     WARNING: no finite validation loss in any epoch -- diverged",
              flush=True)
        sd = (model._orig_mod if hasattr(model, "_orig_mod") else model).state_dict()
        torch.save({"state_dict": sd, "config": cfg, "epoch": args.epochs - 1,
                    "val": vm, "diverged": True}, ckpt_path)
    summary = {"run": run, "fingerprint": fp, "fingerprint_fields": fp_fields,
               "config": cfg, "best": best,
               "diverged": bool(diverged), "nonfinite_steps_total": int(nonfinite_total),
               "history": history, "wall_minutes": (time.time() - t_start) / 60}
    # Written via a temp file + rename: an interruption mid-write would otherwise
    # leave a truncated JSON that resume would have to guess about.
    mpath = os.path.join(dirs["output"], f"{run}_metrics.json")
    tmp = mpath + ".tmp"
    with open(tmp, "w") as f:
        json.dump(summary, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, mpath)

    print(f"     best epoch {best['epoch']}  |  VAL mse={best['val_mse']:.4f} "
          f"pearson={best.get('val_pearson', float('nan')):.4f} "
          f"gene_pearson_bio={best.get('val_gene_pearson_bio', float('nan')):.4f}  "
          f"({(time.time()-t_start)/60:.1f} min)", flush=True)

    if wb is not None:
        wb.summary.update({"best_epoch": best["epoch"], "best_val_mse": best["val_mse"],
                           "diverged": bool(diverged),
                           "nonfinite_steps_total": int(nonfinite_total),
                           **{k: v for k, v in best.items() if k.startswith("val_")}})
        wb.finish()
    del model, opt
    torch.cuda.empty_cache()
    return summary


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="training YAML configuration")
    ap.add_argument("--gpu", type=int, default=None,
                    help="physical GPU index for this single-GPU job (sets CUDA_VISIBLE_DEVICES)")
    ap.add_argument("--embedding", default=None)
    ap.add_argument("--embed-root", default=DEFAULT_EMBED_ROOT)
    ap.add_argument("--label-dir", default=None,
                    help="directory containing traincell_*gene.npy and RNA_label_columns.log")
    ap.add_argument("--biosample-set", choices=("14bio",), default="14bio")
    ap.add_argument("--lrs", type=float, nargs="+", default=list(DEFAULT_LRS))
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--warmup-epochs", type=int, default=5)
    ap.add_argument("--min-lr", type=float, default=1e-7)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    # architecture -- defaults mirror the Puget 524 kb pointwise config
    ap.add_argument("--proj-dim", type=int, default=512)
    ap.add_argument("--num-heads", type=int, default=8)
    ap.add_argument("--num-layers", type=int, default=4)
    ap.add_argument("--mlp-ratio", type=float, default=4.0)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--promoter-tokens", type=int, default=4)
    ap.add_argument("--input-norms", nargs="+", default=list(DEFAULT_INPUT_NORMS),
                    choices=("layernorm", "none"),
                    help="swept axis: 'layernorm' normalizes the raw embedding, "
                         "'none' reproduces Puget exactly")
    ap.add_argument(
        "--output-activations", nargs="+", default=list(DEFAULT_OUTPUT_ACTIVATIONS),
        choices=("linear", "softplus"),
        help="final prediction activation; use both values for a matched control",
    )
    ap.add_argument("--attn", default="sdpa", choices=("sdpa", "flex", "math"))
    ap.add_argument("--precision", default="bf16", choices=("bf16", "fp16", "fp32"))
    ap.add_argument("--compile", action="store_true")
    # data residency
    ap.add_argument("--data-device", default="auto", choices=("auto", "cuda", "cpu", "mmap"))
    ap.add_argument("--headroom-gb", type=float, default=14.0)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--pin-memory", action="store_true")
    # bookkeeping
    ap.add_argument("--run-name", default="seqonly", help="name of the configured run")
    ap.add_argument("--output-dir", default=os.path.join(ROOT, "outputs"))
    ap.add_argument("--wandb-project", default="SeqBaselines",
                    help="every run in this sweep logs into this ONE project")
    ap.add_argument("--wandb-entity", default=None,
                    help="W&B entity owning --wandb-project (account default if omitted)")
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--max-steps", type=int, default=0, help="cap train steps/epoch (smoke tests)")
    ap.add_argument("--validate-only", action="store_true", help="check embedding and label alignment without training")
    ap.add_argument("--smoke", action="store_true",
                    help="2 epochs x 8 steps, mmap data, no wandb -- pipeline check only")
    # Load paper defaults before parsing the final CLI so explicit options win.
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", required=True)
    config_path = Path(pre.parse_known_args(argv)[0].config).expanduser().resolve()
    config = yaml.safe_load(config_path.read_text())
    if not isinstance(config, dict):
        raise ValueError(f"{config_path}: expected a YAML mapping")
    unknown = set(config) - {action.dest for action in ap._actions}
    if unknown:
        raise ValueError(f"{config_path}: unknown settings {sorted(unknown)}")
    ap.set_defaults(**config)
    args = ap.parse_args(argv)
    if args.gpu is not None:
        if args.gpu < 0:
            raise SystemExit("--gpu must be a non-negative GPU index")
        # CUDA is not initialized before this point, so this selects the device.
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    # The configured cell keeps the bare run_name; anything else is suffixed.
    defaults = {
        "input_norm": list(config.get("input_norms", DEFAULT_INPUT_NORMS))[0],
        "output_activation": list(config.get("output_activations", DEFAULT_OUTPUT_ACTIVATIONS))[0],
        "lr": float(list(config.get("lrs", DEFAULT_LRS))[0]),
        "batch_size": int(config.get("batch_size", 32)),
        "seed": int(config.get("seed", 42)),
    }
    if not args.embedding:
        ap.error("embedding must be set in the config or with --embedding")
    for key in ("embed_root", "label_dir", "output_dir"):
        value = getattr(args, key)
        if value is not None and not os.path.isabs(value):
            setattr(args, key, os.path.normpath(os.path.join(ROOT, value)))

    # Smoke runs are pipeline checks on 512 genes. They get their own out-root
    # and their own name prefix, so they can never be mistaken for a result nor
    # overwrite one -- and `--overwrite` applies only inside that sandbox.
    run_prefix = ""
    if args.smoke:
        args.epochs, args.warmup_epochs, args.max_steps = 2, 1, 8
        args.data_device, args.no_wandb, args.overwrite = "mmap", True, True
        args.output_dir = os.path.join(args.output_dir, "smoke")
        run_prefix = "smoke_"

    specs = discover_embeddings(args.embed_root, only_key=args.embedding,
                                splits=("train", "valid"))
    if args.embedding not in specs:
        raise SystemExit(f"unknown embedding {args.embedding!r}\navailable:\n  "
                         + "\n  ".join(sorted(specs)))
    spec = specs[args.embedding]
    if args.label_dir is not None:
        spec = replace(spec, label_dir=args.label_dir)
    args.biosample_indices, args.biosample_names = select_biosamples(spec, args.biosample_set)
    args.output_dim = len(args.biosample_names)
    print(f"[{args.biosample_set}] {args.output_dim} outputs: {args.biosample_names}", flush=True)

    if args.validate_only:
        for split in ("train", "valid"):
            labels = load_labels(spec, split)
            assert labels.shape == (spec.n_rows[split], 14)
        print(f"[validated] {spec.describe()}")
        return

    dirs = {"output": args.output_dir, "logs": os.path.join(args.output_dir, "logs")}
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)

    args.lrs = _dedup(args.lrs)
    args.input_norms = _dedup(args.input_norms)
    args.output_activations = _dedup(args.output_activations)

    # Every cell of the grid gets its own name -- checked over the FULL grid, not
    # just the not-yet-done part, so a collision is caught even on a resume. The
    # only way to trip this is two learning rates that agree to one significant
    # figure (1e-4 and 1.04e-4 both render "1e-04"); the fix is a coarser grid or
    # separate invocations, not a silently overwritten run.
    grid = [
        (lr, nm, activation,
         run_name(args.run_name, nm, activation, lr, args.batch_size, args.seed,
                  defaults, run_prefix))
        for nm in args.input_norms
        for activation in args.output_activations
        for lr in args.lrs
    ]
    names = [g[3] for g in grid]
    if len(set(names)) != len(names):
        dupes = sorted({n for n in names if names.count(n) > 1})
        raise SystemExit(
            "run-name collision -- these would overwrite each other:\n  "
            + "\n  ".join(dupes)
            + f"\nfrom lrs={args.lrs} input_norms={args.input_norms} "
              f"output_activations={args.output_activations}")

    plan, stale = [], []
    for lr, input_norm, output_activation, run in grid:
        mpath = os.path.join(dirs["output"], f"{run}_metrics.json")
        want_fp, want_fields = fingerprint(
            _cfg_for_fingerprint(args, spec, lr, input_norm, output_activation), spec)
        if os.path.exists(mpath) and not args.overwrite:
            # Existence alone is not proof the stored result answers THIS
            # question: epochs, precision, attention impl, architecture and even
            # the embedding file's mtime all change the answer without changing
            # the name. Compare fingerprints and refuse to reuse a mismatch.
            try:
                prev = json.load(open(mpath))
                have_fp = prev.get("fingerprint")
                have_fields = prev.get("fingerprint_fields", {})
            except (json.JSONDecodeError, OSError) as e:
                stale.append((run, mpath, [f"unreadable metrics file: {e}"]))
                continue
            if have_fp == want_fp:
                print(f"[skip] {run} -- complete, fingerprint {have_fp}")
                continue
            diff = [f"{k}: stored={have_fields.get(k)!r} now={want_fields.get(k)!r}"
                    for k in sorted(set(want_fields) | set(have_fields))
                    if have_fields.get(k) != want_fields.get(k)]
            stale.append((run, mpath, diff or ["fingerprint field set changed"]))
            continue
        plan.append((lr, input_norm, output_activation, run))

    if stale:
        print("\nRefusing to run: these completed runs were produced under a "
              "DIFFERENT configuration.\nReusing them would mix experiments; "
              "overwriting them would destroy the old one.\n")
        for run, mpath, diff in stale:
            print(f"  {run}")
            for d in diff[:8]:
                print(f"      {d}")
        print("\nPick one:\n"
              "  --overwrite                 replace them with the new config\n"
              "  --out-root <dir>            keep both, in separate trees\n"
              "  restore the old settings    to resume the original sweep")
        raise SystemExit(2)

    if not plan:
        print(f"[done] nothing to do for {spec.key}")
        return

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable; run training on a GPU host. No training started.")

    torch.set_num_threads(int(os.environ["OMP_NUM_THREADS"]))
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    print(f"=== {spec.key}   {len(plan)} run(s) = "
          f"{len(args.input_norms)} input-norm(s) x "
          f"{len(args.output_activations)} output activation(s) x "
          f"{len(args.lrs)} lr(s), "
          f"minus completed")
    print(f"    input_norms={args.input_norms}  "
          f"output_activations={args.output_activations}  "
          f"lrs={[f'{lr:.0e}' for lr in args.lrs]}  "
          f"wandb_project={'disabled' if args.no_wandb else args.wandb_project}")
    print(f"    {spec.describe()}", flush=True)

    dev = choose_device(spec, args.data_device, args.headroom_gb)
    on_gpu = dev == "cuda"
    t_load = time.time()
    splits = {s: SplitData(spec, s, device=dev, batch_size=args.batch_size,
                           num_workers=args.num_workers, pin_memory=args.pin_memory,
                           biosample_indices=args.biosample_indices,
                           log_prefix="    ")
              for s in ("train", "valid")}
    if args.smoke:
        for d in splits.values():
            d.n_genes = min(512, d.n_genes)
            d.labels_np = d.labels_np[:d.n_genes]
            d.emb = d.emb[:d.n_genes]
    print(f"    train/valid splits resident on {dev} in {time.time() - t_load:.1f}s", flush=True)

    hvg_valid = hvg_gene_mask(splits["valid"].labels_np)

    results = []
    for lr, input_norm, output_activation, run in plan:
        results.append(train_one(args, spec, lr, input_norm, output_activation, run,
                                 dirs, splits["train"],
                                 splits["valid"], hvg_valid, on_gpu))

    print(f"\n=== {spec.key} summary (selection = best val MSE)")
    print(f"    {'input_norm':>10s}  {'activation':>10s}  {'lr':>8s}  "
          f"{'best_ep':>7s}  {'val_mse':>8s}  "
          f"{'val_r':>7s}  {'val_rho':>8s}  {'val_r_cell':>11s}  {'note':>9s}")
    for r in sorted(results, key=lambda r: r["best"]["val_mse"]):
        note = "DIVERGED" if r["diverged"] else (
            f"{r['nonfinite_steps_total']} nf" if r["nonfinite_steps_total"] else "")
        print(f"    {r['config']['input_norm']:>10s}  "
              f"{r['config']['output_activation']:>10s}  {r['config']['lr']:8.0e}  "
              f"{r['best']['epoch']:7d}  {r['best']['val_mse']:8.4f}  "
              f"{r['best'].get('val_pearson', float('nan')):7.4f}  "
              f"{r['best'].get('val_spearman', float('nan')):8.4f}  "
              f"{r['best'].get('val_gene_pearson_bio', float('nan')):11.4f}  {note:>9s}")


if __name__ == "__main__":
    main()
