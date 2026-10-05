import os
import pickle
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
from scipy.sparse import coo_matrix
from torch.utils.data import DataLoader, Dataset

from .biosamples import load_biosample_table

HIC_TRANSFORMS = ("log", "oe_log")


def _normalize_chrom(chrom: str) -> str:
    return chrom if str(chrom).startswith("chr") else f"chr{chrom}"


def _dense_from_triplets(H, W, row, col, data) -> np.ndarray:
    if row.size == 0:
        return np.zeros((H, W), dtype=np.float32)
    m = (row >= 0) & (row < H) & (col >= 0) & (col < W)
    if not np.all(m):
        row, col, data = row[m], col[m], data[m]
    return coo_matrix(
        (data.astype(np.float32, copy=False),
         (row.astype(np.int32, copy=False), col.astype(np.int32, copy=False))),
        shape=(H, W), dtype=np.float32,
    ).toarray()


def _dense_from_entry(H, W, row, col, dat, entry) -> np.ndarray:
    storage = str(entry.get("storage", "")).lower() if entry is not None else ""
    if storage in {"triu_diag", "upper_triangle_including_diagonal"}:
        off = row != col
        if np.any(off):
            mirror_row, mirror_col, mirror_dat = col[off], row[off], dat[off]
            row = np.concatenate([row, mirror_row])
            col = np.concatenate([col, mirror_col])
            dat = np.concatenate([dat, mirror_dat])
    return _dense_from_triplets(H, W, row, col, dat)


def _apply_hic_transform(dense: np.ndarray, mode: str) -> np.ndarray:
    if mode == "log":
        return np.log10(dense + 1.0).astype(np.float32)
    if mode == "oe_log":
        n = dense.shape[0]
        oe = np.empty_like(dense, dtype=np.float32)
        for d in range(n):
            idx = np.arange(n - d)
            vals = dense[idx, idx + d]
            exp = float(vals.mean()) if vals.size else 0.0
            scale = 1.0 / (exp + 1e-8)
            oe[idx, idx + d] = vals * scale
            oe[idx + d, idx] = vals * scale
        return np.log10(oe + 1.0).astype(np.float32)
    raise ValueError(f"hic_transform must be one of {HIC_TRANSFORMS}, got {mode!r}")


class SeqHiCDataset(Dataset):
    def __init__(
        self,
        bedpe_path: str,
        pkl_paths: List[str],
        label_rows: List[int],
        biosample_names: List[str],
        seq_emb_npy: str,
        label_npy: str,
        window_height: int = 192,
        window_width: int = 192,
        skip_missing_hic: bool = True,
        hic_transform: str = "log",
        seq_emb_mmap_mode: str = "r",
    ):
        super().__init__()
        assert len(pkl_paths) == len(label_rows) == len(biosample_names), "subset length mismatch"
        if hic_transform not in HIC_TRANSFORMS:
            raise ValueError(f"hic_transform must be one of {HIC_TRANSFORMS}, got {hic_transform!r}")
        self.H, self.W = int(window_height), int(window_width)
        self.hic_transform = hic_transform
        self.biosample_names = list(biosample_names)
        self.label_rows = [int(r) for r in label_rows]
        self.skip_missing_hic = bool(skip_missing_hic)

        self.windows = []
        for p in pkl_paths:
            with open(p, "rb") as f:
                self.windows.append(pickle.load(f))

        bedpe = pd.read_csv(
            bedpe_path, sep="\t", header=None,
            names=["chr1", "start1", "end1", "chr2", "start2", "end2"], usecols=range(6))
        bedpe["chr1"] = bedpe["chr1"].map(_normalize_chrom)
        self.n_pairs = len(bedpe)
        self.chroms = bedpe["chr1"].to_numpy()
        self.starts = bedpe["start1"].to_numpy().astype(np.int64)
        self.ends = bedpe["end1"].to_numpy().astype(np.int64)

        seq = np.load(seq_emb_npy, mmap_mode=seq_emb_mmap_mode)
        assert seq.ndim == 3, f"seq emb must be (M, n_bins, embed_dim), got {seq.shape}"
        assert seq.shape[0] == self.n_pairs, \
            f"seq rows ({seq.shape[0]}) != bedpe rows ({self.n_pairs})"
        self.seq = seq
        self.n_bins = int(seq.shape[1])
        self.embed_dim = int(seq.shape[2])

        lab = np.load(label_npy)
        assert lab.ndim == 2 and lab.shape[1] == self.n_pairs, \
            f"label shape {lab.shape} incompatible with n_pairs {self.n_pairs}"
        self.labels = lab.astype(np.float32, copy=False)
        assert max(self.label_rows) < lab.shape[0], "label_row out of range"

        # Biosample-major index; drop (gene, biosample) pairs with no usable Hi-C so
        # a blank contact map never enters training or metrics. Counts are printed.
        self.index = []
        self.dropped = {}
        for bi in range(len(pkl_paths)):
            win = self.windows[bi]
            n_missing = n_empty = 0
            for j in range(self.n_pairs):
                key = f"{self.chroms[j]}:{int(self.starts[j])},{int(self.ends[j])}"
                entry = win.get(key)
                if entry is None:
                    n_missing += 1
                    if self.skip_missing_hic:
                        continue
                elif bool(entry.get("is_empty", int(np.asarray(entry["row"]).size == 0))):
                    n_empty += 1
                    if self.skip_missing_hic:
                        continue
                self.index.append((bi, j))
            self.dropped[self.biosample_names[bi]] = (n_missing, n_empty)

        tot_m = sum(m for m, _ in self.dropped.values())
        tot_e = sum(e for _, e in self.dropped.values())
        if tot_m or tot_e:
            verb = "dropped" if self.skip_missing_hic else "KEPT AS BLANK"
            print(f"[SeqHiCDataset] {verb} {tot_m} missing + {tot_e} empty Hi-C windows "
                  f"across {len(pkl_paths)} biosamples")
            for name, (m, e) in self.dropped.items():
                if m or e:
                    print(f"    {name}: missing={m}, empty={e}")

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int):
        bi, j = self.index[i]
        chrom = str(self.chroms[j])
        s1, e1 = int(self.starts[j]), int(self.ends[j])
        entry = self.windows[bi].get(f"{chrom}:{s1},{e1}")

        if entry is None:
            row = col = np.empty((0,), dtype=np.int16)
            dat = np.empty((0,), dtype=np.float16)
            is_empty = True
        else:
            row = np.asarray(entry["row"], dtype=np.int16)
            col = np.asarray(entry["col"], dtype=np.int16)
            dat = np.asarray(entry["data"], dtype=np.float16)
            is_empty = bool(entry.get("is_empty", int(row.size == 0)))

        dense = np.nan_to_num(_dense_from_entry(self.H, self.W, row, col, dat, entry))
        img = torch.from_numpy(
            np.ascontiguousarray(_apply_hic_transform(dense, self.hic_transform))).unsqueeze(0)

        # Keep fp16: the .npy is fp16 and training runs under AMP fp16, so casting
        # to fp32 here would only double the bytes through collate / pin / H2D.
        seq_emb = torch.from_numpy(np.array(self.seq[j], dtype=np.float16))
        y = torch.as_tensor([self.labels[self.label_rows[bi], j]], dtype=torch.float32)
        meta = (self.biosample_names[bi], chrom, s1, e1, int(is_empty), int(bi), int(j))
        return img, seq_emb, y, meta


def collate_seq_hic(batch):
    imgs, seqs, ys, metas = zip(*batch)
    return (torch.stack(imgs, 0), torch.stack(seqs, 0), torch.stack(ys, 0), list(metas))


def get_dataloader(
    bedpe_path: str,
    pkl_paths: List[str],
    label_rows: List[int],
    biosample_names: List[str],
    seq_emb_npy: str,
    label_npy: str,
    window_height: int = 192,
    window_width: int = 192,
    batch_size: int = 128,
    num_workers: int = 8,
    shuffle: bool = False,
    pin_memory: bool = True,
    persistent_workers: bool = True,
    prefetch_factor: Optional[int] = 2,
    skip_missing_hic: bool = True,
    hic_transform: str = "log",
    seq_emb_mmap_mode: Optional[str] = "r",
) -> DataLoader:
    ds = SeqHiCDataset(
        bedpe_path=bedpe_path, pkl_paths=pkl_paths, label_rows=label_rows,
        biosample_names=biosample_names, seq_emb_npy=seq_emb_npy, label_npy=label_npy,
        window_height=window_height, window_width=window_width,
        skip_missing_hic=skip_missing_hic, hic_transform=hic_transform,
        seq_emb_mmap_mode=seq_emb_mmap_mode)
    return DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle, drop_last=False,
        num_workers=num_workers, pin_memory=pin_memory,
        persistent_workers=bool(persistent_workers and num_workers > 0),
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        collate_fn=collate_seq_hic)


# --------------------------------------------------------------------------- #
# Biosample subset resolution from data/deeply_profiled.csv
# --------------------------------------------------------------------------- #
def resolve_subset(spec, table):
    all_names = [r[2] for r in table]
    if isinstance(spec, str) and spec == "all":
        names = all_names
    elif isinstance(spec, dict) and "all_except" in spec:
        ex = set(spec["all_except"])
        names = [n for n in all_names if n not in ex]
    elif isinstance(spec, (list, tuple)):
        names = list(spec)
    else:
        raise ValueError(f"bad biosample spec: {spec!r}")
    name2 = {r[2]: (r[0], r[1]) for r in table}
    missing = [n for n in names if n not in name2]
    if missing:
        raise ValueError(f"unknown biosample(s): {missing}")
    return names, [name2[n][0] for n in names], [name2[n][1] for n in names]


def build_split_loader(cfg, split, bedpe, seq_npy, label_npy, spec, table, shuffle):
    names, _manifest_rows, accs = resolve_subset(spec, table)
    group = str(cfg.get(f"{split}_label_group", ""))
    train_cells = (
        "A673", "Caco-2", "Calu3", "HCT116", "HepG2", "IMR-90",
        "MCF 10A", "MCF-7", "OCI-LY7", "PC-3", "PC-9", "Panc1",
        "endothelial cell of umbilical vein", "mammary epithelial cell",
    )
    test_cells = ("GM12878", "K562")
    expected = train_cells if group == "traincell" else test_cells if group == "testcell" else None
    if expected is None or tuple(names) != expected:
        raise ValueError(f"{split}: biosamples must match the ordered {group} label rows")
    label_rows = list(range(len(names)))
    labels = np.load(label_npy, mmap_mode="r")
    with open(bedpe) as handle:
        n_genes = sum(bool(line.strip()) for line in handle)
    if labels.shape != (len(names), n_genes) or labels.dtype != np.float32:
        raise ValueError(f"{split}: {label_npy} expected {(len(names), n_genes)} float32, got {labels.shape} {labels.dtype}")
    sequence = np.load(seq_npy, mmap_mode="r")
    if sequence.shape != (n_genes, int(cfg.n_cols), int(cfg.embed_dim)) or sequence.dtype != np.float16:
        raise ValueError(f"{split}: {seq_npy} has unexpected shape or dtype: {sequence.shape} {sequence.dtype}")
    pkl_paths = [os.path.join(cfg.hic_root, split, f"{a}.pkl") for a in accs]
    for p in pkl_paths:
        if not os.path.isfile(p) or os.path.getsize(p) == 0:
            raise FileNotFoundError(f"Hi-C pkl missing or empty: {p}")
    print(f"[{split}] {len(names)} biosamples: {names}")
    loader = get_dataloader(
        bedpe_path=bedpe, pkl_paths=pkl_paths, label_rows=label_rows, biosample_names=names,
        seq_emb_npy=seq_npy, label_npy=label_npy,
        window_height=int(cfg.window_height), window_width=int(cfg.window_width),
        batch_size=int(cfg.batch_size), num_workers=int(cfg.num_workers), shuffle=shuffle,
        pin_memory=bool(cfg.get("pin_memory", True)),
        persistent_workers=bool(cfg.get("persistent_workers", True)),
        prefetch_factor=int(cfg.get("prefetch_factor", 2)),
        skip_missing_hic=bool(cfg.get("skip_missing_hic", True)),
        hic_transform=str(cfg.get("hic_transform", "log")),
        # None makes np.load copy the complete array into anonymous RAM.  Keep
        # mmap as the default: it is shared across forked workers and the Linux
        # page cache already retains hot embedding pages without a second copy.
        seq_emb_mmap_mode=(None if bool(cfg.get("preload_seq", False)) else "r"))
    return loader, names
