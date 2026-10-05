from __future__ import annotations

import glob
import json
import os
import time
from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import BatchSampler, DataLoader, Dataset, RandomSampler, SequentialSampler

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_EMBED_ROOT = os.path.join(REPO_ROOT, "outputs", "seq_embeddings")
DATASETS_ROOT = os.path.join(REPO_ROOT, "outputs")

# bed-file prefix -> (label directory, human-readable split family)
_SPLIT_FAMILIES = {
    "Borzoi_genes": ("borzoi_data", "borzoi"),
    "Enformer_genes": ("enformer_data", "enformer"),
}
SPLITS = ("train", "valid", "test")


@dataclass(frozen=True)
class EmbeddingSpec:
    key: str                       # e.g. borzoi_rep0_524k_trunk
    family: str                    # borzoi | alphagenome | enformer
    layer: str                     # trunk | post_transformer | ...
    split_family: str              # borzoi | enformer  (which gene split)
    label_dir: str
    npy: Dict[str, str]
    n_rows: Dict[str, int]
    n_tokens: int
    embed_dim: int
    seq_length: int

    @property
    def train_bytes(self) -> int:
        return self.n_rows["train"] * self.n_tokens * self.embed_dim * 2

    @property
    def valid_bytes(self) -> int:
        return self.n_rows["valid"] * self.n_tokens * self.embed_dim * 2

    @property
    def test_bytes(self) -> int:
        return self.n_rows.get("test", 0) * self.n_tokens * self.embed_dim * 2

    @property
    def all_bytes(self) -> int:
        return sum(rows * self.n_tokens * self.embed_dim * 2
                   for rows in self.n_rows.values())

    def describe(self) -> str:
        rows = "/".join(f"{split}:{self.n_rows[split]}" for split in self.n_rows)
        return (f"{self.key:46s} {self.split_family:8s} split  "
                f"tokens={self.n_tokens:4d} dim={self.embed_dim:5d}  "
                f"rows={rows}  loaded splits={self.all_bytes / 1e9:6.1f} GB")


def check_complete(directory: str, run_prefix: str, layer: str,
                   n_rows: Dict[str, int]) -> List[str]:
    problems: List[str] = []
    for split in n_rows:
        pattern = os.path.join(directory, f".{run_prefix}_{split}.shard*.done.json")
        markers = sorted(glob.glob(pattern))
        if not markers:
            problems.append(f"{split}: no shard-completion markers ({os.path.basename(pattern)})")
            continue

        seen, covered, n_shards = {}, [], set()
        for m in markers:
            try:
                d = json.load(open(m))
            except (json.JSONDecodeError, OSError) as e:
                problems.append(f"{split}: unreadable marker {os.path.basename(m)}: {e}")
                continue
            n_shards.add(int(d.get("num_shards", -1)))
            seen[int(d.get("shard_id", -1))] = d
            covered.append((int(d.get("start", -1)), int(d.get("end", -1))))
            if layer not in d.get("layers", []):
                problems.append(
                    f"{split}: shard {d.get('shard_id')} produced {d.get('layers')}, "
                    f"not {layer!r}")

        if len(n_shards) != 1:
            problems.append(f"{split}: markers disagree on num_shards: {sorted(n_shards)}")
            continue
        total = n_shards.pop()
        missing = [i for i in range(total) if i not in seen]
        if missing:
            problems.append(f"{split}: missing shard marker(s) {missing} of {total}")

        # the shards must tile [0, n_rows) with no gap and no overlap
        covered.sort()
        cursor, expect = 0, n_rows[split]
        for start, end in covered:
            if start != cursor:
                problems.append(f"{split}: row coverage breaks at {cursor} "
                                f"(next shard starts at {start})")
                break
            cursor = end
        else:
            if cursor != expect:
                problems.append(f"{split}: shards cover {cursor} rows, "
                                f"metadata says {expect}")
    return problems


def _split_family_for(bed_path: str) -> Tuple[str, str]:
    base = os.path.basename(bed_path)
    for prefix, (label_dir, family) in _SPLIT_FAMILIES.items():
        if base.startswith(prefix):
            return label_dir, family
    raise ValueError(f"cannot map bed {base!r} to a label directory; "
                     f"known prefixes: {sorted(_SPLIT_FAMILIES)}")


def discover_embeddings(embed_root: str = DEFAULT_EMBED_ROOT, *,
                        require_complete: bool = True,
                        report: bool = False,
                        only_key: Optional[str] = None,
                        splits: Sequence[str] = SPLITS) -> Dict[str, EmbeddingSpec]:
    splits = tuple(splits)
    if not splits or len(set(splits)) != len(splits) or any(s not in SPLITS for s in splits):
        raise ValueError(f"splits must be distinct members of {SPLITS}")
    if "train" not in splits:
        raise ValueError("embedding discovery requires the train split")
    if not os.path.isdir(embed_root):
        raise FileNotFoundError(f"embedding root not found: {embed_root}")
    by_key: Dict[str, Dict[str, dict]] = {}
    for sub in sorted(os.listdir(embed_root)):
        d = os.path.join(embed_root, sub)
        if not os.path.isdir(d):
            continue
        for name in sorted(os.listdir(d)):
            if not name.endswith(tuple(f"_1kb_mean_{s}.metadata.json" for s in splits)):
                continue
            meta = json.load(open(os.path.join(d, name)))
            stem = name[len("human_seq_embeddings_"):-len(".metadata.json")]
            split = meta["split"]
            if split not in splits or not stem.endswith(f"_1kb_mean_{split}"):
                continue
            key = stem[: -len(f"_1kb_mean_{split}")]
            if only_key is not None and key != only_key:
                continue
            npy = os.path.join(d, name.replace(".metadata.json", ".npy"))
            if not os.path.exists(npy):
                continue
            by_key.setdefault(key, {})[split] = {"meta": meta, "npy": npy}

    specs: Dict[str, EmbeddingSpec] = {}
    for key, per_split in sorted(by_key.items()):
        missing = [s for s in splits if s not in per_split]
        if missing:
            print(f"[discover] skipping {key}: missing splits {missing}")
            continue
        metas = {s: per_split[s]["meta"] for s in splits}
        shapes = {s: tuple(metas[s]["shape"]) for s in splits}
        tokens = {shapes[s][1] for s in splits}
        dims = {shapes[s][2] for s in splits}
        if len(tokens) != 1 or len(dims) != 1:
            raise ValueError(f"{key}: inconsistent shapes across splits: {shapes}")
        label_dir, split_family = _split_family_for(metas["train"]["bed"])
        for split in splits:
            other_dir, other_family = _split_family_for(metas[split]["bed"])
            if (other_dir, other_family) != (label_dir, split_family):
                raise ValueError(f"{key}: {split} BED belongs to another gene split")
            arr = np.load(per_split[split]["npy"], mmap_mode="r")
            if arr.shape != shapes[split] or arr.dtype != np.float16:
                raise ValueError(f"{key}/{split}: array shape or dtype disagrees with metadata")
        layer = metas["train"]["embedding_layer"]
        directory = os.path.dirname(per_split["train"]["npy"])
        n_rows = {s: shapes[s][0] for s in splits}

        # The .npy exists and its sidecar looks right -- neither proves the shards
        # actually ran. Check the completion markers before trusting the array.
        run_prefix = key[: -len(f"_{layer}")] if key.endswith(f"_{layer}") else key
        problems = check_complete(directory, run_prefix, layer, n_rows)
        if problems:
            msg = (f"[discover] INCOMPLETE, SKIPPING {key}:\n    "
                   + "\n    ".join(problems))
            if require_complete:
                print(msg)
                continue
            print(msg.replace("INCOMPLETE, SKIPPING", "INCOMPLETE (not skipped)"))
        elif report:
            print(f"[discover] complete: {key}")

        specs[key] = EmbeddingSpec(
            key=key,
            family=key.split("_")[0],
            layer=layer,
            split_family=split_family,
            label_dir=os.path.join(DATASETS_ROOT, label_dir),
            npy={s: per_split[s]["npy"] for s in splits},
            n_rows=n_rows,
            n_tokens=tokens.pop(),
            embed_dim=dims.pop(),
            seq_length=int(metas["train"]["seq_length"]),
        )
    return specs


def load_labels(spec: EmbeddingSpec, split: str) -> np.ndarray:
    path = os.path.join(spec.label_dir, f"traincell_{split}gene.npy")
    labels = np.load(path, mmap_mode="r")
    expected = (14, spec.n_rows[split])
    if labels.shape != expected or labels.dtype != np.float32:
        raise ValueError(f"{path}: expected {expected} float32, got {labels.shape} {labels.dtype}")
    if not np.isfinite(labels).all():
        raise ValueError(f"{path}: non-finite labels")
    return np.ascontiguousarray(labels.T)


def load_biosample_names(spec: EmbeddingSpec) -> List[str]:
    path = os.path.join(spec.label_dir, "RNA_label_columns.log")
    names = []
    for line in open(path):
        fields = line.rstrip("\n").split("\t")
        if len(fields) == 4 and fields[0] == "traincell":
            if int(fields[1]) != len(names):
                raise ValueError(f"{path}: unexpected traincell row order")
            names.append(fields[2])
    if len(names) != 14 or len(set(names)) != 14:
        raise ValueError(f"{path}: expected 14 unique training biosamples, got {names}")
    if {"GM12878", "K562"} & set(names):
        raise ValueError(f"{path}: held-out biosample in training labels")
    return names


def select_biosamples(spec: EmbeddingSpec, biosample_set: str = "14bio"):
    if biosample_set != "14bio":
        raise ValueError("the paper split only supports the 14 training biosamples")
    names = load_biosample_names(spec)
    return list(range(14)), names


def _read_into(path: str, device: str, chunk_rows: int = 512, log_prefix: str = "") -> torch.Tensor:
    src = np.load(path, mmap_mode="r")
    shape = tuple(src.shape)
    dst = torch.empty(shape, dtype=torch.float16, device=device)
    t0 = time.time()
    for i in range(0, shape[0], chunk_rows):
        j = min(i + chunk_rows, shape[0])
        # np.array(..., copy=True): the memmap slice is read-only, and
        # torch.from_numpy warns on non-writable buffers even for a pure read.
        block = torch.from_numpy(np.array(src[i:j], dtype=np.float16))
        dst[i:j].copy_(block, non_blocking=False)
    dt = time.time() - t0
    gb = dst.numel() * 2 / 1e9
    print(f"{log_prefix}loaded {os.path.basename(path)} {shape} -> {device} "
          f"{gb:.1f} GB in {dt:.1f}s ({gb / max(dt, 1e-9):.2f} GB/s)", flush=True)
    del src
    return dst


class _GeneBatchDataset(Dataset):
    def __init__(self, emb: np.ndarray, labels: np.ndarray):
        self.emb = emb
        self.labels = labels

    def __len__(self) -> int:
        return self.emb.shape[0]

    def __getitem__(self, idx):
        idx = np.asarray(idx, dtype=np.int64)
        return (torch.from_numpy(np.ascontiguousarray(self.emb[idx])),
                torch.from_numpy(np.ascontiguousarray(self.labels[idx])),
                torch.from_numpy(idx))


class SplitData:
    def __init__(self, spec: EmbeddingSpec, split: str, *, device: str,
                 batch_size: int, num_workers: int = 4, pin_memory: bool = False,
                 log_prefix: str = "", biosample_indices=None):
        self.spec = spec
        self.split = split
        self.device = device
        self.batch_size = int(batch_size)
        self.labels_np = load_labels(spec, split)          # (n_genes, 14) float32
        if biosample_indices is not None:
            self.labels_np = np.ascontiguousarray(self.labels_np[:, biosample_indices])
        self.n_genes = self.labels_np.shape[0]
        self.n_bio = self.labels_np.shape[1]

        if device == "cuda":
            self.emb = _read_into(spec.npy[split], "cuda", log_prefix=log_prefix)
            self.labels = torch.from_numpy(self.labels_np).cuda()
            self.loader = None
        else:
            if device == "mmap":
                self.emb = np.load(spec.npy[split], mmap_mode="r")
            else:
                self.emb = _read_into(spec.npy[split], "cpu", log_prefix=log_prefix).numpy()
            self.labels = None
            self.loader_kwargs = dict(num_workers=int(num_workers), pin_memory=bool(pin_memory))
            self.loader = None

    @property
    def bytes(self) -> int:
        if isinstance(self.emb, torch.Tensor):
            return self.emb.numel() * 2
        return int(self.emb.size) * 2

    def iter_batches(self, *, shuffle: bool, generator: Optional[torch.Generator] = None,
                     drop_last: bool = False) -> Iterator[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        if self.device == "cuda":
            if shuffle:
                order = torch.randperm(self.n_genes, generator=generator, device="cpu").cuda()
            else:
                order = torch.arange(self.n_genes, device="cuda")
            for i in range(0, self.n_genes, self.batch_size):
                idx = order[i:i + self.batch_size]
                if drop_last and idx.numel() < self.batch_size:
                    break
                yield self.emb.index_select(0, idx), self.labels.index_select(0, idx), idx
            return

        ds = _GeneBatchDataset(self.emb, self.labels_np)
        inner = (RandomSampler(ds, generator=generator) if shuffle else SequentialSampler(ds))
        sampler = BatchSampler(inner, self.batch_size, drop_last=drop_last)
        loader = DataLoader(ds, batch_size=None, sampler=sampler, **self.loader_kwargs)
        for x, y, idx in loader:
            yield x, y, idx

    def free(self):
        self.emb = None
        self.labels = None
        if self.device == "cuda":
            torch.cuda.empty_cache()


def choose_device(spec: EmbeddingSpec, requested: str, headroom_gb: float = 14.0) -> str:
    if requested != "auto":
        return requested
    if not torch.cuda.is_available():
        return "cpu"
    free_b, _total = torch.cuda.mem_get_info()
    need = spec.all_bytes
    fits = need + headroom_gb * 1e9 < free_b
    print(f"[device] need {need / 1e9:.1f} GB, free {free_b / 1e9:.1f} GB, "
          f"headroom {headroom_gb:.0f} GB -> {'cuda' if fits else 'cpu'}", flush=True)
    return "cuda" if fits else "cpu"


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--embed-root", default=DEFAULT_EMBED_ROOT)
    ap.add_argument("--no-require-complete", action="store_true",
                    help="list embeddings even if their shard markers are missing")
    args = ap.parse_args()
    specs = discover_embeddings(args.embed_root,
                                require_complete=not args.no_require_complete,
                                report=True)
    print(f"{len(specs)} embedding sets under {args.embed_root}\n")
    for key, spec in specs.items():
        print("  " + spec.describe())
        _ = spec.all_bytes
        for split in SPLITS:
            load_labels(spec, split)          # raises if the split family is wrong
    print("\nall label matrices matched their embedding row counts")
    fams = sorted({s.split_family for s in specs.values()})
    for fam in fams:
        keys = [k for k, s in specs.items() if s.split_family == fam]
        print(f"  split family {fam:8s}: {len(keys)} embeddings -> comparable to each other")
