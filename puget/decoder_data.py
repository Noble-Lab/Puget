import os
import math
import torch
import numpy as np
import random
from torch.utils.data import Dataset, Sampler, DataLoader, get_worker_info
from typing import List, Optional, Tuple, Iterator

import numpy as np
import torch
from torch.utils.data import Dataset

class TransformerSeqDataset(Dataset):
    def __init__(self, x_npy_path, y_npy_path, mmap_mode="r", embed_dim=3072):
        self.x = np.load(x_npy_path, mmap_mode=mmap_mode)
        assert self.x.ndim == 2, "x must be 2D"
        m, a = self.x.shape

        y_full = np.load(y_npy_path)
        self.y = y_full.transpose()
        assert self.y.ndim == 2, "y must be 2D after transpose"

        self.m_sequences, self.n_biosamples = self.y.shape
        assert m == self.m_sequences, (
            f"Mismatch: x has {m} rows but y has {self.m_sequences} rows"
        )

        assert a % embed_dim == 0, (
            f"x second dim ({a}) must be divisible by embed_dim ({embed_dim})"
        )
        n_bins = a // embed_dim
        self.x = self.x.reshape(m, n_bins, embed_dim)

    def __len__(self):
        return self.m_sequences

    def __getitem__(self, idx):
        x_np = self.x[idx]
        y_np = self.y[idx]

        x_arr = np.ascontiguousarray(x_np)
        y_arr = np.ascontiguousarray(y_np)

        x_t = torch.from_numpy(x_arr.copy()).half()
        y_t = torch.from_numpy(y_arr.copy()).float()

        return x_t, y_t

class TransformerHiCDataset(Dataset):
    def __init__(self, x_npy_path, y_npy_path, mmap_mode="r"):
        self.x_mm = np.load(x_npy_path, mmap_mode=mmap_mode)     # (B, M, R, C, D)
        self.B, self.M, self.R, self.C, self.D = self.x_mm.shape
        self.n_rows = self.B * self.M                                      # total examples
        self.embed_dim = self.D

        y_full = np.load(y_npy_path).astype(np.float32)          # (B, M)
        assert y_full.shape == (self.B, self.M), "x/y biosample or seq counts mismatch"
        self.y_flat = y_full.reshape(self.n_rows, 1)             # (B*M, 1)

    def __len__(self):
        return self.n_rows

    def __getitem__(self, idx):
        b = idx // self.M
        m = idx % self.M

        x_arr = self.x_mm[b, m]
        y_arr = self.y_flat[idx]

        x_arr = np.ascontiguousarray(x_arr)
        y_arr = np.ascontiguousarray(y_arr)

        x_t = torch.from_numpy(x_arr.copy()).half()
        y_t = torch.from_numpy(y_arr.copy()).float()

        return x_t, y_t

class TransformerSeqHiCNewDataset(Dataset):
    def __init__(
        self,
        hic_x_path: str,
        seq_x_path: str,
        y_path: str,
        embed_dim: int,
        seq_weights_path: Optional[str] = None,
        mmap_mode: str = "r",
    ):
        self.hic = np.load(hic_x_path, mmap_mode=mmap_mode)
        assert self.hic.ndim == 5, "hic must be 5D (B, M, R, C, D_hic)"
        self.B, self.M, self.n_row, self.n_col, self.D_hic = self.hic.shape

        seq_mm = np.load(seq_x_path, mmap_mode=mmap_mode)  # (M, embed_dim * n_bins)
        assert seq_mm.ndim == 2, "seq must be 2D"
        m_seq, total_dim = seq_mm.shape
        assert m_seq == self.M, f"seq has {m_seq} rows but hic expects M={self.M}"
        assert total_dim % embed_dim == 0, (
            f"total_dim={total_dim} not divisible by embed_dim={embed_dim}"
        )
        n_bins = total_dim // embed_dim

        self.seq = seq_mm.reshape(self.M, n_bins, embed_dim)

        y_full = np.load(y_path).astype(np.float32)
        assert y_full.shape == (self.B, self.M), "y must be (B, M)"
        self.y_flat = y_full.reshape(self.B * self.M, 1)

        if seq_weights_path is not None:
            w = np.load(seq_weights_path).astype(np.float32)
            assert w.shape == (self.M,), "seq_weights must be (M,)"
            self.seq_weights = w
        else:
            self.seq_weights = np.ones(self.M, dtype=np.float32)

    def __len__(self) -> int:
        return self.B * self.M

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        i = idx // self.M
        j = idx % self.M

        hic_t    = torch.from_numpy(self.hic[i, j].copy()) 
        
        seq_t    = torch.from_numpy(self.seq[j].copy()) 

        y_t      = torch.from_numpy(self.y_flat[idx].copy()) 
        weight_t = torch.from_numpy(self.seq_weights[j:j+1].copy())

        return hic_t, seq_t, y_t, weight_t

    def get_biosample_sequence_pair(self, biosample_idx: int, sequence_idx: int):
        flat_idx = biosample_idx * self.M + sequence_idx
        return self.__getitem__(flat_idx)
    
def get_sequence_dataloader(
    x_npy_path,
    y_npy_path,
    batch_size,
    shuffle=True,
    num_workers=0,
    prefetch_factor=2,
    persistent_workers=False,
    pin_memory=True,
    mmap_mode="r",
    embed_dim=3072):
    dataset = TransformerSeqDataset(x_npy_path, y_npy_path, mmap_mode=mmap_mode, embed_dim=embed_dim)
    use_persistent = persistent_workers and (num_workers > 0)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=use_persistent,
        prefetch_factor=prefetch_factor if num_workers > 0 else None
    )

def get_hic_dataloader(
    x_npy_path,
    y_npy_path,
    batch_size,
    shuffle=True,
    num_workers=0,
    prefetch_factor=2,
    persistent_workers=False,
    pin_memory=True,
    mmap_mode="r"
):
    dataset = TransformerHiCDataset(x_npy_path, y_npy_path, mmap_mode=mmap_mode)
    use_persistent = persistent_workers and (num_workers > 0)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=use_persistent,
        prefetch_factor=prefetch_factor if num_workers > 0 else None
    )

def get_seq_hic_new_dataloader(
    hic_x_path,
    seq_x_path,
    y_path,
    batch_size,
    shuffle=True,
    num_workers=0,
    prefetch_factor=4,
    persistent_workers=True,
    pin_memory=True,
    mmap_mode="r",
    embed_dim=3072,
    seq_weights_path=None,
):
    dataset = TransformerSeqHiCNewDataset(hic_x_path, seq_x_path, y_path, 
                               embed_dim=embed_dim, seq_weights_path=seq_weights_path, 
                               mmap_mode=mmap_mode)
    use_persistent = persistent_workers and (num_workers > 0)
        
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=use_persistent,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
    )
