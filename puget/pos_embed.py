# --------------------------------------------------------
# 2D sine-cosine position embedding
# References:
# Transformer: https://github.com/tensorflow/models/blob/master/official/nlp/transformer/model_utils.py
# MoCo v3: https://github.com/facebookresearch/moco-v3
# MAE: https://github.com/facebookresearch/mae
# HiCFoundation: https://github.com/Noble-Lab/HiCFoundation
# --------------------------------------------------------

from typing import Tuple, List, Optional
import math
import numpy as np
import torch
import torch.nn as nn
from torch import einsum
from einops import rearrange

def get_1d_sincos_pos_embed(embed_dim: int, length: int, cls_token: bool = False) -> np.ndarray:
    """
    Return a [length, D] (or [1+length, D]) matrix of fixed 1-D sine-cosine embeddings.

    Args
    ----
    embed_dim : total embedding dimension D (must be even)
    length    : sequence length N
    cls_token : if True, prepend a zero-vector for a CLS token
    """
    assert embed_dim % 2 == 0, "embed_dim must be divisible by 2"
    half_dim = embed_dim // 2

    # positions: shape (length,)
    pos = np.arange(length, dtype=np.float32)[:, None]      # (length, 1)
    # dim indices: 0,1,...,half_dim-1
    dims = np.arange(half_dim, dtype=np.float32)[None, :]   # (1, half_dim)

    # compute the angle rates
    angle_rates = 1.0 / (10000 ** (2 * dims / embed_dim))   # (1, half_dim)
    angle_rads = pos * angle_rates                          # (length, half_dim)

    # interleave sin and cos to get (length, embed_dim)
    sin_embed = np.sin(angle_rads)
    cos_embed = np.cos(angle_rads)
    pe = np.concatenate([sin_embed, cos_embed], axis=1)     # (length, embed_dim)

    if cls_token:
        cls_row = np.zeros((1, embed_dim), dtype=np.float32)
        pe = np.vstack([cls_row, pe])                       # (1+length, embed_dim)

    return pe

# ---------------------------------------------------------------------------------
# This function is adapted from the HiCFoundation repository.
# Source: https://github.com/Noble-Lab/HiCFoundation/blob/main/model/pos_embed.py
# License: Apache License 2.0
# ---------------------------------------------------------------------------------
def get_2d_sincos_pos_embed_rectangle(embed_dim: int,
                                      grid_hw: Tuple[int, int],
                                      cls_token: bool = False) -> np.ndarray:
    """
    Return a [H*W, D] (or [1+H*W, D]) matrix with fixed 2-D sine-cosine embeddings.

    Args
    ----
    embed_dim : total embedding dimension D
    grid_hw   : (H, W) = (#rows, #cols) of the spatial grid
    cls_token : prepend one row of zeros for a CLS token if True
    """
    h, w = grid_hw
    grid_y = np.arange(h, dtype=np.float32)
    grid_x = np.arange(w, dtype=np.float32)
    grid = np.meshgrid(grid_x, grid_y)              # order: X, Y
    grid = np.stack(grid, axis=0)                   # [2, H, W]
    grid = grid.reshape(2, 1, h, w)

    assert embed_dim % 4 == 0, "embed_dim must be divisible by 4 for current splitting logic"
    emb_h = embed_dim // 2
    emb_w = embed_dim - emb_h

    def _pe(pos, dim):
        i = np.arange(dim // 2, dtype=np.float32)
        denom = 1. / (10000 ** (2 * i / dim))
        pos = pos[..., None] * denom                # [..., dim/2]
        return np.concatenate([np.sin(pos), np.cos(pos)], axis=-1)

    emb_y = _pe(grid[1], emb_h)                     # [H, W, emb_h]
    emb_x = _pe(grid[0], emb_w)                     # [H, W, emb_w]
    pos = np.concatenate([emb_y, emb_x], axis=-1)   # [H, W, D]
    pos = pos.reshape(h * w, embed_dim)             # [H*W, D]

    if cls_token:
        pos = np.vstack([np.zeros((1, embed_dim), dtype=np.float32), pos])
    return pos
