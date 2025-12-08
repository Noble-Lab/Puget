import os
import pickle
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from typing import List, Tuple, Optional, Dict, Any, Union

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torchvision.transforms as T
from enformer_pytorch import FastaInterval

# seq_indices_to_one_hot adapted from enformer_pytorch codebase
# https://github.com/lucidrains/enformer-pytorch
def seq_indices_to_one_hot(t, padding = -1):
    is_padding = t == padding
    t = t.clamp(min = 0)
    one_hot = F.one_hot(t, num_classes = 5)
    out = one_hot[..., :4].float()
    out = out.masked_fill(is_padding[..., None], 0.25)
    return out

def get_seq_indices_dataloader(
    seq_indices_npy: str,
    batch_size: int = 4,
    num_workers: int = 0,
    shuffle: bool = False
) -> DataLoader:
    """
    Loads a .npy file containing sequence indices (uint8), wraps it in a TensorDataset,
    and returns a DataLoader.
    """
    seq_idx = np.load(seq_indices_npy, mmap_mode=None)
    
    assert seq_idx.ndim == 2 and seq_idx.dtype == np.uint8, \
        f"Expected (N, L) uint8, got {seq_idx.shape} {seq_idx.dtype}"
    
    N, L = int(seq_idx.shape[0]), int(seq_idx.shape[1])
    print(f"Loaded sequence indices: N={N}, L={L}, dtype={seq_idx.dtype}")

    ds = TensorDataset(torch.from_numpy(seq_idx))
    
    return DataLoader(
        ds, 
        batch_size=batch_size, 
        num_workers=num_workers, 
        shuffle=shuffle,
        pin_memory=True, 
        drop_last=False
    )

def _normalize_chrom_name(chrom: str) -> str:
    return chrom if chrom.startswith("chr") else f"chr{chrom}"

def _biosample_from_path(p: str) -> str:
    base = os.path.basename(p)
    name, _ = os.path.splitext(base)
    return name

def _convert_rgb_from_log(data_log: np.ndarray, is_empty: bool) -> np.ndarray:
    """
    data_log: (H, W) in log10(count + 1)
    Returns: (H, W, 3) float32 in [0,1] except when empty_policy="nan" & max==0 → NaNs
    """
    max_value = np.max(data_log)
    if is_empty or max_value == 0.0:
        # Stable, valid image of zeros (after Normalize it will be constant)
        data_rgb = np.zeros((data_log.shape[0], data_log.shape[1], 3), dtype=np.float32)
        return data_rgb

    data_red = np.ones(data_log.shape, dtype=np.float32)
    data_log1 = (max_value - data_log) / max_value
    data_rgb = np.stack([data_red,data_log1,data_log1],axis=-1)
    return data_rgb

def _dense_from_triplets(H, W, row: np.ndarray, col: np.ndarray, data: np.ndarray) -> np.ndarray:
    """
    Build a dense HxW float32 array from sparse triplets (offsets are within the window),
    """
    if row.size == 0:
        return np.zeros((H, W), dtype=np.float32)

    mask = (row >= 0) & (row < H) & (col >= 0) & (col < W)
    if not np.all(mask):
        row = row[mask]
        col = col[mask]
        data = data[mask]

    dense = coo_matrix(
        (data.astype(np.float32, copy=False), (row.astype(np.int32, copy=False), col.astype(np.int32, copy=False))),
        shape=(H, W),
        dtype=np.float32,
    ).toarray()
    
    return dense
    
class HiCSubmatDataset(Dataset):
    """
    Dataset that:
      1) reads BEDPE rows (order-preserving),
      2) constructs keys 'chr:row_start,col_start' using `resolution`,
      3) fetches sparse triplets from a pre-saved windows .pkl,
      4) builds a dense (H x W) array, log10-transforms, converts to 'RGB',
      5) applies transforms (ToTensor + Normalize by default),
      6) returns: (img: FloatTensor[3,H,W], total_count: FloatTensor[1], meta: list)
         where meta = [bio, chrom, rs, cs, int(is_empty)]
    """
    def __init__(
        self,
        bedpe_path: str,
        windows_pkl_paths: Union[str, List[str]],
        resolution: int,
        window_height: int,
        window_width: int,
        transform: Optional[torch.nn.Module] = None,
        biosample_names: Optional[List[str]] = None,
    ):
        super().__init__()
        self.resolution = int(resolution)
        self.H = int(window_height)
        self.W = int(window_width)
        
        if isinstance(windows_pkl_paths, str):
            windows_pkl_paths = [windows_pkl_paths]

        if biosample_names is None:
            biosample_names = [_biosample_from_path(p) for p in windows_pkl_paths]
        assert len(biosample_names) == len(windows_pkl_paths), "biosample_names length mismatch"
        self.biosamples = biosample_names
        
        self.windows_multi = {}
        for bio, pkl_path in zip(self.biosamples, windows_pkl_paths):
            with open(pkl_path, "rb") as f:
                self.windows_multi[bio] = pickle.load(f)

        bedpe = pd.read_csv(
            bedpe_path,
            sep="\t",
            header=None,
            names=["chr1", "start1", "end1", "chr2", "start2", "end2"],
            usecols=range(6),
        )
        bedpe["chr1"] = bedpe["chr1"].map(_normalize_chrom_name)
        bedpe["chr2"] = bedpe["chr2"].map(_normalize_chrom_name)

        # Build ordered index: [(chrom, rs, cs), ...] in the same order as the BEDPE file
        self.index = []
        for bio in self.biosamples:
            for _, r in bedpe.iterrows():
                chrom = r["chr1"]
                rs = int(r["start1"] // self.resolution)
                cs = int(r["start2"] // self.resolution)
                self.index.append((bio, chrom, rs, cs))

        # Per-biosample total_count
        self._total_count_per_bio = {}
        for bio, win_dict in self.windows_multi.items():
            if len(win_dict) > 0:
                any_entry = next(iter(win_dict.values()))
                tot = float(np.asarray(any_entry.get("total_count", 0.0), dtype=np.float64))
            else:
                tot = 0.0
            self._total_count_per_bio[bio] = tot

        # Default transforms
        if transform is None:
            self.transform = T.Compose([
                T.ToTensor(),  # HxWxC float32 [0..1] -> CxHxW
                T.Normalize(mean=[0.485, 0.456, 0.406],
                            std =[0.229, 0.224, 0.225]),
            ])
        else:
            self.transform = transform

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int):
        bio, chrom, rs, cs = self.index[i]
        key = f"{chrom}:{rs},{cs}"

        entry = self.windows_multi[bio].get(key, None)
        if entry is None:
            # Key missing → treat as empty
            row = np.empty((0,), dtype=np.int16)
            col = np.empty((0,), dtype=np.int16)
            dat = np.empty((0,), dtype=np.float16)
            is_empty = True
        else:
            row = np.asarray(entry["row"],  dtype=np.int16)
            col = np.asarray(entry["col"],  dtype=np.int16)
            dat = np.asarray(entry["data"], dtype=np.float16)
            # rely on saved flag; if not present, infer
            is_empty = bool(entry.get("is_empty", int(row.size == 0)))

        # Build dense counts and transform
        dense_counts = _dense_from_triplets(self.H, self.W, row, col, dat)
        dense_counts = np.nan_to_num(dense_counts)
        
        # log10 transform on counts
        data_log = np.log10(dense_counts + 1.0).astype(np.float32)

        rgb = _convert_rgb_from_log(data_log, is_empty)  # (H, W, 3) float32
        img = self.transform(rgb)

        total_count = torch.tensor(self._total_count_per_bio[bio], dtype=torch.float32)
        meta = (bio, chrom, rs, cs, int(is_empty))

        return img, total_count, meta

class SeqHiCSubmatDataset(Dataset):
    """
    Returns:
      (seq: FloatTensor[L, 4], img: FloatTensor[3, H, W], total_count: FloatTensor[1], meta)
      where meta = (bio, chrom, rs, cs, start1, end1, is_empty_int)
    """
    def __init__(
        self,
        bedpe_path: str,
        windows_pkl_paths: Union[str, List[str]],
        fasta_file: Optional[str],
        resolution: int,
        window_height: int,
        window_width: int,
        seq_length: Optional[int] = None,
        seq_return_indices: bool = False,        # keep False to get one-hot
        seq_shift_augs: Optional[Tuple[int,int]] = None,
        seq_rc_aug: bool = False,
        transform_hic: Optional[torch.nn.Module] = None,
        biosample_names: Optional[List[str]] = None,
        seq_indices_npy: Optional[str] = None
    ):
        super().__init__()

        self.resolution = int(resolution)
        self.H = int(window_height)
        self.W = int(window_width)

        bedpe = pd.read_csv(
            bedpe_path,
            sep="\t",
            header=None,
            names=["chr1", "start1", "end1", "chr2", "start2", "end2"],
            usecols=range(6),
        )
        bedpe["chr1"] = bedpe["chr1"].map(_normalize_chrom_name)
        bedpe["chr2"] = bedpe["chr2"].map(_normalize_chrom_name)
        self.bedpe = bedpe

        self._use_seq_indices = seq_indices_npy is not None
        self._seq_idx = None
        if self._use_seq_indices:
            arr = np.load(seq_indices_npy, mmap_mode=None)
            assert arr.ndim == 2 and arr.dtype == np.uint8, f"seq_indices must be (N, L) uint8"
            self._N_bedpe = len(self.bedpe)
            assert arr.shape[0] == self._N_bedpe, "rows of seq_indices must equal #BEDPE rows"
            # assert arr.min() >= 0 and arr.max() <= 3, "tokens must be in {0,1,2,3}"
            if seq_length is not None:
                assert arr.shape[1] == seq_length, "seq_length mismatch"
            self._seq_idx = arr
            self.fasta_interval = None
        else:
            self.fasta_interval = FastaInterval(
                fasta_file=fasta_file,
                context_length=seq_length,
                return_seq_indices=seq_return_indices,
                shift_augs=seq_shift_augs,
                rc_aug=seq_rc_aug
            )

        if isinstance(windows_pkl_paths, str):
            windows_pkl_paths = [windows_pkl_paths]
        if biosample_names is None:
            biosample_names = [_biosample_from_path(p) for p in windows_pkl_paths]
        assert len(biosample_names) == len(windows_pkl_paths), "biosample_names length mismatch"
        self.biosamples = biosample_names

        self.windows_multi = {}
        for bio, pkl_path in zip(self.biosamples, windows_pkl_paths):
            with open(pkl_path, "rb") as f:
                self.windows_multi[bio] = pickle.load(f)

        # Each item: (bio, chr1, rs, cs, start1, end1)
        self.index = []
        for bio in self.biosamples:
            for ridx, r in bedpe.iterrows():
                chrom = r["chr1"]
                rs = int(r["start1"] // self.resolution)
                cs = int(r["start2"] // self.resolution)
                self.index.append((bio, chrom, rs, cs, int(r["start1"]), int(r["end1"]), int(ridx)))

        self._total_count_per_bio = {}
        for bio, win_dict in self.windows_multi.items():
            if len(win_dict) > 0:
                any_entry = next(iter(win_dict.values()))
                tot = float(np.asarray(any_entry.get("total_count", 0.0), dtype=np.float64))
            else:
                tot = 0.0
            self._total_count_per_bio[bio] = tot

        # Default transforms
        self.transform_hic = transform_hic or T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std =[0.229, 0.224, 0.225]),
        ])

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int):
        bio, chrom, rs, cs, start1, end1, bedpe_idx = self.index[i]
        key = f"{chrom}:{rs},{cs}"

        # --- Hi-C ---
        entry = self.windows_multi[bio].get(key, None)
        if entry is None:
            row = np.empty((0,), dtype=np.int16)
            col = np.empty((0,), dtype=np.int16)
            dat = np.empty((0,), dtype=np.float16)
            is_empty = True
        else:
            row = np.asarray(entry["row"],  dtype=np.int16)
            col = np.asarray(entry["col"],  dtype=np.int16)
            dat = np.asarray(entry["data"], dtype=np.float16)
            is_empty = bool(entry.get("is_empty", int(row.size == 0)))

        dense_counts = _dense_from_triplets(self.H, self.W, row, col, dat)
        dense_counts = np.nan_to_num(dense_counts)
        data_log = np.log10(dense_counts + 1.0).astype(np.float32)
        rgb = _convert_rgb_from_log(data_log, is_empty)
        img = self.transform_hic(rgb)

        total_count = torch.tensor(self._total_count_per_bio[bio], dtype=torch.float32)

        if self._use_seq_indices:
            seq_tokens = torch.from_numpy(self._seq_idx[bedpe_idx]).long()   # (L,)
            seq = seq_indices_to_one_hot(seq_tokens)        #(L, 4)
        else:
            seq = self.fasta_interval(chrom, start1, end1, return_augs=False)
            if seq.ndim == 1 or seq.dtype == torch.long:
                seq = seq_indices_to_one_hot(seq)           # (L, 4)

        meta = (bio, chrom, rs, cs, start1, end1, int(is_empty))
        return seq, img, total_count, meta

def _cast_meta_tuple(m):
    out = []
    for x in m:
        if isinstance(x, (int, np.integer, bool, np.bool_)):
            out.append(int(x))
        else:
            out.append(str(x))
    return tuple(out)

def collate_keep_meta(batch):
    """
    HiCSubmatDataset  -> returns (imgs, counts, metas)
    SeqHiCSubmatDataset -> returns (seqs, imgs, counts, metas)
    """
    first = batch[0]

    if len(first) == 3:
        # Hi-C only
        imgs, counts, metas = zip(*batch)
        imgs   = torch.stack(imgs, dim=0)
        counts = torch.stack(counts, dim=0)
        metas  = [_cast_meta_tuple(m) for m in metas]
        return imgs, counts, metas

    else:
        # Seq + Hi-C
        seqs, imgs, counts, metas = zip(*batch)
        seqs   = torch.stack(seqs, dim=0)
        imgs   = torch.stack(imgs, dim=0)
        counts = torch.stack(counts, dim=0)
        metas  = [_cast_meta_tuple(m) for m in metas]
        return seqs, imgs, counts, metas

def get_hicsubmat_dataloader(
    bedpe_path: str,
    windows_pkl_paths: Union[str, List[str]],
    resolution: int,
    window_height: int,
    window_width: int,
    batch_size: int = 8,
    num_workers: int = 4,
    shuffle: bool = False,
    transform: Optional[torch.nn.Module] = None,
    biosample_names: Optional[List[str]] = None,
) -> DataLoader:
    hic_dataset = HiCSubmatDataset(
        bedpe_path=bedpe_path,
        windows_pkl_paths=windows_pkl_paths,
        resolution=resolution,
        window_height=window_height,
        window_width=window_width,
        transform=transform,
        biosample_names=biosample_names,
    )
    return DataLoader(
        hic_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_keep_meta,
    )

def get_seq_hicsubmat_dataloader(
    bedpe_path: str,
    windows_pkl_paths: Union[str, List[str]],
    fasta_file: str,
    resolution: int,
    window_height: int,
    window_width: int,
    seq_length: Optional[int],
    batch_size: int = 8,
    num_workers: int = 0,
    shuffle: bool = False,
    transform_hic: Optional[torch.nn.Module] = None,
    biosample_names: Optional[List[str]] = None,
    seq_indices_npy: Optional[str] = None,
) -> DataLoader:
    dataset = SeqHiCSubmatDataset(
        bedpe_path=bedpe_path,
        windows_pkl_paths=windows_pkl_paths,
        fasta_file=fasta_file,
        resolution=resolution,
        window_height=window_height,
        window_width=window_width,
        seq_length=seq_length,
        seq_return_indices=False,
        seq_shift_augs=None,
        seq_rc_aug=False,
        transform_hic=transform_hic,
        biosample_names=biosample_names,
        seq_indices_npy=seq_indices_npy,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_keep_meta,
    )

def precompute_k_expected_matrix(H: int, W: int, k_median: np.ndarray, center_offset: int = 384) -> np.ndarray:
    """
    Build a (H,W) matrix M where M[r,c] = k_median[ |(c - r) - center_offset| ].
    Assumes k_median has length 576 for 0..575.
    """
    # r grid [0..H-1], c grid [0..W-1]
    r = np.arange(H, dtype=np.int32)[:, None]
    c = np.arange(W, dtype=np.int32)[None, :]
    k = np.abs((c - r) - int(center_offset))
    k = np.clip(k, 0, len(k_median) - 1)
    return k_median[k]

class PugetPerturbationPairDataset(Dataset):
    def __init__(
        self,
        windows_pkl_path: str,
        targets_df: pd.DataFrame,
        seq_indices_npy: str,
        k_median: Optional[np.ndarray],
        H: int = 192,
        W: int = 960,
        fill_mode: str = "k_expected",
        transform_hic: Optional[torch.nn.Module] = None,
    ):
        super().__init__()
        self.H, self.W = int(H), int(W)
        self.targets = targets_df.reset_index(drop=True).copy()
        self.fill_mode = str(fill_mode)
        assert self.fill_mode in ("k_expected", "zeros")
        with open(windows_pkl_path, "rb") as f:
            self.windows = pickle.load(f)
        if len(self.windows) > 0:
            any_entry = next(iter(self.windows.values()))
            self.total_count = float(np.asarray(any_entry.get("total_count", 0.0), dtype=np.float64))
        else:
            self.total_count = 0.0
        arr = np.load(seq_indices_npy, mmap_mode=None)
        assert arr.ndim == 2 and arr.dtype == np.uint8, "seq_indices_npy must be (N_test, L) uint8"
        self.seq_idx = arr
        self.transform_hic = transform_hic or T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std =[0.229, 0.224, 0.225]),
        ])
        if self.fill_mode == "k_expected":
            assert k_median is not None and k_median.shape == (576,)
            self.expected_mat = precompute_k_expected_matrix(self.H, self.W, k_median, center_offset=384)
        else:
            self.expected_mat = None

    def __len__(self) -> int:
        return len(self.targets)

    def _to_img(self, dense_counts: np.ndarray, is_empty: bool) -> torch.Tensor:
        data_log = np.log10(dense_counts + 1.0).astype(np.float32)
        rgb = _convert_rgb_from_log(data_log, is_empty)
        img = self.transform_hic(rgb)
        return img

    def __getitem__(self, i: int):
        row = self.targets.iloc[i]
        key = row["submat_key"]
        r0, r1 = int(row["r0"]), int(row["r1"])
        c0, c1 = int(row["c0"]), int(row["c1"])
        srow = int(row["seq_row"])
        seq_tokens = torch.from_numpy(self.seq_idx[srow]).long()
        seq = seq_indices_to_one_hot(seq_tokens)
        entry = self.windows.get(key, None)
        if entry is None:
            raise KeyError(f"Missing submat_key in windows: {key}")
        row_arr = np.asarray(entry["row"], dtype=np.int16)
        col_arr = np.asarray(entry["col"], dtype=np.int16)
        dat_arr = np.asarray(entry["data"], dtype=np.float16)
        is_empty = bool(entry.get("is_empty", int(row_arr.size == 0)))
        dense_orig = _dense_from_triplets(self.H, self.W, row_arr, col_arr, dat_arr)
        dense_orig = np.nan_to_num(dense_orig)
        dense_pert = dense_orig.copy()
        if self.fill_mode == "zeros":
            dense_pert[r0:r1, c0:c1] = 0.0
        else:
            dense_pert[r0:r1, c0:c1] = self.expected_mat[r0:r1, c0:c1]
        img_orig = self._to_img(dense_orig, is_empty)
        img_pert = self._to_img(dense_pert, is_empty)
        total_count = torch.tensor(self.total_count, dtype=torch.float32)
        meta = (
            key, r0, r1, c0, c1,
            str(row.get("gene_symbol", "")),
            str(row.get("enhancer_id", row.get("loop_id", ""))),
            int(row.get("Regulated", -1)) if "Regulated" in row else -1,
        )
        return seq, img_orig, img_pert, total_count, meta

def collate_puget_perturbation(batch):
    seqs, imgs_o, imgs_p, counts, metas = zip(*batch)
    seqs   = torch.stack(seqs, dim=0)
    imgs_o = torch.stack(imgs_o, dim=0)
    imgs_p = torch.stack(imgs_p, dim=0)
    counts = torch.stack(counts, dim=0)
    return seqs, imgs_o, imgs_p, counts, list(metas)

def get_puget_perturbation_dataloader(
    windows_pkl_path: str,
    targets: Union[str, pd.DataFrame],
    seq_indices_npy: str,
    k_median: Optional[np.ndarray],
    H: int = 192,
    W: int = 960,
    fill_mode: str = "k_expected", # "k_expected" or "zeros"
    batch_size: int = 4,
    num_workers: int = 0,
    shuffle: bool = False,
    transform_hic: Optional[torch.nn.Module] = None,
) -> DataLoader:
    if isinstance(targets, str):
        targets_df = pd.read_csv(targets)
    else:
        targets_df = targets

    ds = PugetPerturbationPairDataset(
        windows_pkl_path=windows_pkl_path,
        targets_df=targets_df,
        seq_indices_npy=seq_indices_npy,
        k_median=k_median,
        H=H,
        W=W,
        fill_mode=fill_mode,
        transform_hic=transform_hic,
    )

    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_puget_perturbation,
    )
