from __future__ import annotations

import os
import pickle
from typing import Iterable, Optional

import numpy as np
import pandas as pd
import torch
from scipy.sparse import coo_matrix
from torch.utils.data import DataLoader, Dataset

from .biosamples import load_biosample_table


IMAGENET_MEAN = np.asarray((0.485, 0.456, 0.406), dtype=np.float32)[:, None, None]
IMAGENET_STD = np.asarray((0.229, 0.224, 0.225), dtype=np.float32)[:, None, None]
TRIANGLE_STORAGE = {"triu_diag", "upper_triangle_including_diagonal"}


def normalize_chrom(chrom: str) -> str:
    chrom = str(chrom)
    return chrom if chrom.startswith("chr") else f"chr{chrom}"


def resolve_biosamples(
    names: Iterable[str], table: list[tuple[int, str, str]]
) -> tuple[list[str], list[int], list[str]]:
    requested = list(names)
    by_name = {name: (row, accession) for row, accession, name in table}
    missing = [name for name in requested if name not in by_name]
    if missing:
        raise ValueError(f"Unknown biosamples: {missing}")
    return (
        requested,
        [by_name[name][0] for name in requested],
        [by_name[name][1] for name in requested],
    )


def dense_from_entry(entry: Optional[dict], size: int = 512) -> np.ndarray:
    if entry is None:
        return np.zeros((size, size), dtype=np.float32)
    row = np.asarray(entry.get("row", ()), dtype=np.int16)
    col = np.asarray(entry.get("col", ()), dtype=np.int16)
    data = np.asarray(entry.get("data", ()), dtype=np.float16)
    if not (row.size == col.size == data.size):
        raise ValueError("Sparse row, col, and data arrays differ in length")
    if row.size == 0:
        return np.zeros((size, size), dtype=np.float32)
    keep = (row >= 0) & (row < size) & (col >= 0) & (col < size)
    row, col, data = row[keep], col[keep], data[keep]
    storage = str(entry.get("storage", "full")).lower()
    if storage in TRIANGLE_STORAGE:
        offdiag = row != col
        original_row, original_col, original_data = row, col, data
        row = np.concatenate((original_row, original_col[offdiag]))
        col = np.concatenate((original_col, original_row[offdiag]))
        data = np.concatenate((original_data, original_data[offdiag]))
    elif storage != "full":
        raise ValueError(f"Unsupported sparse storage convention {storage!r}")
    dense = coo_matrix(
        (data.astype(np.float32, copy=False), (row.astype(np.int32), col.astype(np.int32))),
        shape=(size, size),
        dtype=np.float32,
    ).toarray()
    dense = np.nan_to_num(dense, nan=0.0, posinf=0.0, neginf=0.0)
    if np.any(dense < 0):
        raise ValueError("Raw Hi-C input contains negative contacts")
    return dense


def hicfoundation_image(dense_counts: np.ndarray, is_empty: bool = False) -> torch.Tensor:
    data_log = np.log10(dense_counts + 1.0).astype(np.float32, copy=False)
    maximum = float(data_log.max(initial=0.0))
    if is_empty or maximum == 0.0:
        rgb_chw = np.zeros((3, *data_log.shape), dtype=np.float32)
    else:
        inverse = (maximum - data_log) / maximum
        rgb_chw = np.stack((np.ones_like(data_log), inverse, inverse), axis=0)
    normalized = (rgb_chw - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(np.ascontiguousarray(normalized, dtype=np.float32))


class HiCFoundationRawDataset(Dataset):
    def __init__(
        self,
        *,
        bedpe_path: str,
        pkl_paths: list[str],
        label_rows: list[int],
        biosample_names: list[str],
        label_npy: str,
        size: int = 512,
        skip_missing: bool = True,
    ):
        super().__init__()
        if not (len(pkl_paths) == len(label_rows) == len(biosample_names)):
            raise ValueError("Hi-C paths, label rows, and biosample names differ in length")
        self.size = int(size)
        self.label_rows = [int(value) for value in label_rows]
        self.biosample_names = list(biosample_names)
        self.skip_missing = bool(skip_missing)

        self.windows = []
        for path in pkl_paths:
            with open(path, "rb") as handle:
                self.windows.append(pickle.load(handle))

        bedpe = pd.read_csv(
            bedpe_path,
            sep="\t",
            header=None,
            names=["chrom", "start", "end"],
            usecols=range(3),
        )
        self.chroms = bedpe["chrom"].map(normalize_chrom).to_numpy()
        self.starts = bedpe["start"].to_numpy(dtype=np.int64)
        self.ends = bedpe["end"].to_numpy(dtype=np.int64)
        self.n_genes = len(bedpe)

        labels = np.load(label_npy, mmap_mode="r")
        if labels.ndim != 2 or labels.shape[1] != self.n_genes:
            raise ValueError(
                f"Labels {label_npy} have shape {labels.shape}; expected (*, {self.n_genes})"
            )
        if not self.label_rows or max(self.label_rows) >= labels.shape[0]:
            raise ValueError("A requested label row is outside the label matrix")
        self.labels = labels

        self.total_counts = []
        self.index: list[tuple[int, int]] = []
        self.dropped: dict[str, tuple[int, int]] = {}
        for bio_idx, (name, windows) in enumerate(zip(self.biosample_names, self.windows)):
            if not windows:
                raise ValueError(f"{name}: window PKL is empty")
            first = next(iter(windows.values()))
            storage = str(first.get("storage", "")).lower()
            semantics = str(first.get("source_semantics", "")).lower()
            if storage not in TRIANGLE_STORAGE:
                raise ValueError(f"{name}: expected triu_diag storage, found {storage!r}")
            if semantics not in {"raw", "raw_none"}:
                raise ValueError(f"{name}: expected raw_NONE contacts, found {semantics!r}")
            total_count = float(first.get("total_count", 0.0))
            if not np.isfinite(total_count) or total_count <= 0:
                raise ValueError(f"{name}: invalid total_count {total_count}")
            self.total_counts.append(total_count)
            missing = empty = 0
            for gene_idx in range(self.n_genes):
                entry = windows.get(self.key(gene_idx))
                if entry is None:
                    missing += 1
                    if self.skip_missing:
                        continue
                elif bool(entry.get("is_empty", np.asarray(entry.get("row", ())).size == 0)):
                    empty += 1
                    if self.skip_missing:
                        continue
                self.index.append((bio_idx, gene_idx))
            self.dropped[name] = (missing, empty)

        total_missing = sum(value[0] for value in self.dropped.values())
        total_empty = sum(value[1] for value in self.dropped.values())
        if total_missing or total_empty:
            action = "dropped" if self.skip_missing else "kept as zero maps"
            print(f"[HiCFoundationRawDataset] {action}: missing={total_missing}, empty={total_empty}")
            for name, counts in self.dropped.items():
                if any(counts):
                    print(f"  {name}: missing={counts[0]}, empty={counts[1]}")

    def key(self, gene_idx: int) -> str:
        return f"{self.chroms[gene_idx]}:{int(self.starts[gene_idx])},{int(self.ends[gene_idx])}"

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, item: int):
        bio_idx, gene_idx = self.index[item]
        entry = self.windows[bio_idx].get(self.key(gene_idx))
        is_empty = entry is None or bool(
            entry.get("is_empty", np.asarray(entry.get("row", ())).size == 0)
        )
        image = hicfoundation_image(dense_from_entry(entry, self.size), is_empty)
        total_count = torch.tensor(self.total_counts[bio_idx], dtype=torch.float32)
        target = torch.tensor(
            [float(self.labels[self.label_rows[bio_idx], gene_idx])], dtype=torch.float32
        )
        metadata = (self.biosample_names[bio_idx], int(bio_idx), int(gene_idx))
        return image, total_count, target, metadata


def collate_hicfoundation(batch):
    images, counts, targets, metadata = zip(*batch)
    return torch.stack(images), torch.stack(counts), torch.stack(targets), list(metadata)


def init_hicfoundation_worker(_worker_id: int) -> None:
    torch.set_num_threads(1)


def build_hicfoundation_loader(
    *, data_cfg: dict, training_cfg: dict, split: str,
    biosamples: list[str], shuffle: bool, batch_size: int,
) -> tuple[DataLoader, list[str]]:
    table = load_biosample_table(data_cfg["biosamples_csv"])
    names, _global_rows, accessions = resolve_biosamples(biosamples, table)
    group = str(data_cfg[f"{split}_label_group"])
    expected = [row[2] for row in table if (row[0] < 14) == (group == "traincell")]
    if group not in {"traincell", "testcell"} or names != expected:
        raise ValueError(f"{split}: biosamples do not match ordered {group} labels")
    label_path = data_cfg[f"{split}_label"]
    bedpe_path = data_cfg[f"{split}_bedpe"]
    labels = np.load(label_path, mmap_mode="r")
    with open(bedpe_path) as handle:
        n_genes = sum(bool(line.strip()) for line in handle)
    if labels.shape != (len(names), n_genes) or labels.dtype != np.float32:
        raise ValueError(f"{label_path}: expected {(len(names), n_genes)} float32 labels")
    paths = [os.path.join(data_cfg["hic_root"], split, f"{acc}.pkl") for acc in accessions]
    missing = [path for path in paths if not os.path.isfile(path) or os.path.getsize(path) == 0]
    if missing:
        raise FileNotFoundError(f"Missing raw Hi-C windows: {missing}")
    dataset = HiCFoundationRawDataset(
        bedpe_path=bedpe_path, pkl_paths=paths,
        label_rows=list(range(len(names))), biosample_names=names,
        label_npy=label_path, size=int(data_cfg["size"]),
        skip_missing=bool(data_cfg.get("skip_missing_hic", True)),
    )
    workers = int(training_cfg.get("num_workers", 0))
    loader = DataLoader(
        dataset, batch_size=int(batch_size), shuffle=bool(shuffle), drop_last=False,
        num_workers=workers, pin_memory=bool(training_cfg.get("pin_memory", False)),
        persistent_workers=bool(training_cfg.get("persistent_workers", True) and workers > 0),
        prefetch_factor=int(training_cfg.get("prefetch_factor", 2)) if workers > 0 else None,
        collate_fn=collate_hicfoundation,
        worker_init_fn=init_hicfoundation_worker if workers > 0 else None,
    )
    print(f"[{split}] raw/NONE, {len(names)} biosamples, {len(dataset)} usable examples")
    return loader, names
