import math
from typing import List, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils as U
import pytorch_lightning as pl
import math
from typing import Optional, List

from scipy.stats import pearsonr, spearmanr
from .pos_embed import get_1d_sincos_pos_embed, get_2d_sincos_pos_embed_rectangle

class Transformer1DRegressor(pl.LightningModule):
    def __init__(
        self,
        input_dim,                  # original embed dim D
        output_dim,                 # regression target dim
        steps_per_train_epoch,
        n_bins,                     # length of 1D sequence
        proj_dim=512,               # projected embedding dimension before transformer
        mlp_ratio=4,
        num_heads=16,
        num_layers=4,
        pool_method="cls",
        dropout_p=0.1,
        lr=1e-3,
        warmup_epochs=1,
        decay_epochs=9,
        min_lr=1e-5,
        weight_decay=1e-4,
        **kwargs
    ):
        """
        1D transformer regressor over 1D sequence (n_bins).
        """
        super().__init__()
        self.save_hyperparameters()

        self.seq_len_after = n_bins
        self.input_proj = nn.Linear(input_dim, proj_dim)

        # CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, proj_dim))
        
        pe = get_1d_sincos_pos_embed(
            embed_dim=proj_dim, length=self.seq_len_after, cls_token=True
        )
        pe_t = torch.from_numpy(pe).float()
        self.register_buffer("pos_embed", pe_t, persistent=True)

        dim_feedforward = int(proj_dim * mlp_ratio)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=proj_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout_p,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.final_norm = nn.LayerNorm(proj_dim)
        self.head = nn.Linear(proj_dim, output_dim)
        self.criterion = nn.MSELoss()
        
        # Buffers for metrics accumulation
        self.validation_step_outputs = []
        self.validation_step_targets = []
        self.test_step_outputs = []
        self.test_step_targets = []

    def forward(self, x):
        """
        x: (batch, n_bins, input_dim)
        returns: (batch, output_dim)
        """
        B, N, D = x.shape
        assert N == self.hparams.n_bins, f"Expected n_bins {self.hparams.n_bins}, got {N}"
        assert D == self.hparams.input_dim, f"Expected input_dim={self.hparams.input_dim}, got {D}"

        # 1. Linear Projection
        x = self.input_proj(x)

        # 2. Add Positional Encoding
        pos = self.pos_embed[1:]  # not taking pos_embed for CLS token
        x = x + pos.to(x.dtype)

        # 3. Prepend CLS token
        cls = self.cls_token.to(x.dtype).expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)               

        # 4. Pass through transformer & Norm
        x = self.transformer(x)
        x = self.final_norm(x)

        # 5. Pooling
        if self.hparams.pool_method == "cls":
            pooled = x[:, 0]                         
        else:                                        
            pooled = x[:, 1:].mean(dim=1)

        out = self.head(pooled)
        return out

    def training_step(self, batch, batch_idx):
        x, y = batch[:2]
        w = batch[2] if len(batch) == 3 else None
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch[:2]
        w = batch[2] if len(batch) == 3 else None
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        # Accumulate predictions and targets.
        self.validation_step_outputs.append(y_hat.detach())
        self.validation_step_targets.append(y.detach())
        return {"val_loss": loss}

    def on_validation_epoch_end(self):
        # Concatenate all predictions/targets
        all_preds = torch.cat(self.validation_step_outputs, dim=0)  # (m_seq, n_biosamples)
        all_targets = torch.cat(self.validation_step_targets, dim=0)
        val_loss = self.criterion(all_preds, all_targets)

        preds_np = all_preds.cpu().numpy()
        targets_np = all_targets.cpu().numpy()
        num_samples, num_biosamples = preds_np.shape

        if num_samples > 1:
            pearson_vals, spearman_vals = [], []
            for i in range(num_biosamples):
                pearson_vals.append(pearsonr(preds_np[:, i], targets_np[:, i])[0])
                spearman_vals.append(spearmanr(preds_np[:, i], targets_np[:, i])[0])
            pearson_corr = float(np.mean(pearson_vals))
            spearman_corr = float(np.mean(spearman_vals))
        else:
            pearson_corr = pearsonr(preds_np[:, 0], targets_np[:, 0])[0]
            spearman_corr = spearmanr(preds_np[:, 0], targets_np[:, 0])[0]

        self.log("val_loss", val_loss, prog_bar=True, sync_dist=True)
        self.log("val_pearson", pearson_corr, prog_bar=True, sync_dist=True)
        self.log("val_spearman", spearman_corr, prog_bar=True, sync_dist=True)

        # clear buffers
        self.validation_step_outputs.clear()
        self.validation_step_targets.clear()
        return {"val_loss": val_loss}

    def test_step(self, batch, batch_idx):
        x, y = batch[:2]
        w = batch[2] if len(batch) == 3 else None
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.test_step_outputs.append(y_hat.detach())
        self.test_step_targets.append(y.detach())
        return {"test_loss": loss}

    def on_test_epoch_end(self):
        all_preds = torch.cat(self.test_step_outputs, dim=0)  # (m_seqs, n_biosamples)
        all_targets = torch.cat(self.test_step_targets, dim=0)
        test_loss = self.criterion(all_preds, all_targets)

        preds_np = all_preds.cpu().numpy()
        targets_np = all_targets.cpu().numpy()
        num_samples, num_biosamples = preds_np.shape

        if num_samples > 1:
            pearson_vals, spearman_vals = [], []
            for i in range(num_biosamples):
                pearson_vals.append(pearsonr(preds_np[:, i], targets_np[:, i])[0])
                spearman_vals.append(spearmanr(preds_np[:, i], targets_np[:, i])[0])
            pearson_corr = float(np.mean(pearson_vals))
            spearman_corr = float(np.mean(spearman_vals))
        else:
            pearson_corr = pearsonr(preds_np[:, 0], targets_np[:, 0])[0]
            spearman_corr = spearmanr(preds_np[:, 0], targets_np[:, 0])[0]

        self.log("test_loss", test_loss, prog_bar=True, sync_dist=True)
        self.log("test_pearson", pearson_corr, prog_bar=True, sync_dist=True)
        self.log("test_spearman", spearman_corr, prog_bar=True, sync_dist=True)

        # clear buffers
        self.test_step_outputs.clear()
        self.test_step_targets.clear()
        return {"test_loss": test_loss}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), 
                                     lr=self.hparams.lr,
                                     weight_decay=self.hparams.weight_decay)

        warmup_steps = self.hparams.warmup_epochs * self.hparams.steps_per_train_epoch
        decay_steps = self.hparams.decay_epochs  * self.hparams.steps_per_train_epoch

        def lr_lambda(step):
            # 1) linear warmup
            if step < warmup_steps:
                return float(step) / max(1.0, warmup_steps)
            # 2) half-cycle cosine decay
            elif step < warmup_steps + decay_steps:
                decay_step = step - warmup_steps
                # cosine from 1→(min_lr/lr)
                cosine_decay = 0.5 * (1.0 + math.cos(math.pi * decay_step / decay_steps))
                return (1.0 - self.hparams.min_lr / self.hparams.lr) * cosine_decay \
                       + (self.hparams.min_lr / self.hparams.lr)
            # 3) after decay period, hold at min_lr
            else:
                return self.hparams.min_lr / self.hparams.lr

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            }
        }

class Transformer2DRegressor(pl.LightningModule):
    def __init__(
        self,
        input_dim,                  # original embed dim D
        output_dim,                 # regression target dim
        steps_per_train_epoch,
        n_rows,                     # R spatial dimension
        n_cols,                     # C spatial dimension
        proj_dim=512,               # projected embedding dimension before transformer
        mlp_ratio=4,
        num_heads=16,
        num_layers=4,
        pool_method="cls",
        dropout_p=0.1,
        lr=1e-3,
        warmup_epochs=1,
        decay_epochs=9,
        min_lr=1e-5,
        weight_decay=1e-4,
    ):
        """
        Transformer regressor over spatial grid (n_cols x n_rows).
        """
        super().__init__()
        self.save_hyperparameters()

        # Projection from input embedding D -> proj_dim
        self.input_proj = nn.Linear(input_dim, proj_dim)

        num_patches = n_rows * n_cols
        pe = get_2d_sincos_pos_embed_rectangle(
            proj_dim, (n_rows, n_cols), cls_token=True
        )                                 # (1+H*W, D)

        pe_t = torch.from_numpy(pe).float()
        self.register_buffer("pos_embed", pe_t, persistent=True)

        # CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, proj_dim))

        # Transformer encoder stack
        dim_feedforward = int(proj_dim * mlp_ratio)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=proj_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout_p,
            activation="gelu",
            batch_first=True,
        )

        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.final_norm = nn.LayerNorm(proj_dim)

        self.head = nn.Linear(proj_dim, output_dim)

        self.criterion = nn.MSELoss()

        # Buffers for metrics accumulation
        self.validation_step_outputs = []
        self.validation_step_targets = []
        self.test_step_outputs = []
        self.test_step_targets = []

    def forward(self, x):
        """
        x: (batch, R=n_rows, C=n_cols, input_dim)
        returns: (batch, output_dim)
        """
        B, R, C, D = x.shape
        assert R == self.hparams.n_rows and C == self.hparams.n_cols, \
            f"Expected spatial ({self.hparams.n_rows}, {self.hparams.n_cols}), got ({R},{C})"
        assert D == self.hparams.input_dim, f"Expected input_dim={self.hparams.input_dim}, got {D}"

        # project embeddings
        x = self.input_proj(x)  # (B, R, C, proj_dim)

        # positional encoding (fixed positional embedding)
        pos = self.pos_embed[1:]  # not taking pos_embed for CLS token
        pos = pos.reshape(R, C, -1)
        x = x + pos.to(x.dtype)

        # flatten
        x = x.contiguous().view(B, R * C, -1) # (B, S, proj_dim)

        # prepend CLS
        cls = self.cls_token.expand(B, -1, -1)       # (B, 1, D)
        x = torch.cat([cls, x], dim=1)               # (B, 1+S, D)

        # transformer encoder
        x = self.transformer(x)  # (B, S, proj_dim)
        x = self.final_norm(x)

        # pooling
        if self.hparams.pool_method == "cls":
            pooled = x[:, 0]                         # (B, D)
        else:                                        # mean pooling over patches
            pooled = x[:, 1:].mean(dim=1)

        out = self.head(pooled)
        return out

    def training_step(self, batch, batch_idx):
        x, y = batch[:2]
        w = batch[2] if len(batch) == 3 else None
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch[:2]
        w = batch[2] if len(batch) == 3 else None
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        # Accumulate predictions and targets.
        self.validation_step_outputs.append(y_hat.detach())
        self.validation_step_targets.append(y.detach())
        return {"val_loss": loss}

    def on_validation_epoch_end(self):
        # Concatenate all predictions/targets
        all_preds = torch.cat(self.validation_step_outputs, dim=0)  # (m_seq, n_biosamples)
        all_targets = torch.cat(self.validation_step_targets, dim=0)
        val_loss = self.criterion(all_preds, all_targets)

        preds_np = all_preds.cpu().numpy()
        targets_np = all_targets.cpu().numpy()
        num_samples, num_biosamples = preds_np.shape

        if num_samples > 1:
            pearson_vals, spearman_vals = [], []
            for i in range(num_biosamples):
                pearson_vals.append(pearsonr(preds_np[:, i], targets_np[:, i])[0])
                spearman_vals.append(spearmanr(preds_np[:, i], targets_np[:, i])[0])
            pearson_corr = float(np.mean(pearson_vals))
            spearman_corr = float(np.mean(spearman_vals))
        else:
            pearson_corr = pearsonr(preds_np[:, 0], targets_np[:, 0])[0]
            spearman_corr = spearmanr(preds_np[:, 0], targets_np[:, 0])[0]

        self.log("val_loss", val_loss, prog_bar=True, sync_dist=True)
        self.log("val_pearson", pearson_corr, prog_bar=True, sync_dist=True)
        self.log("val_spearman", spearman_corr, prog_bar=True, sync_dist=True)

        # clear buffers
        self.validation_step_outputs.clear()
        self.validation_step_targets.clear()
        return {"val_loss": val_loss}

    def test_step(self, batch, batch_idx):
        x, y = batch[:2]
        w = batch[2] if len(batch) == 3 else None
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.test_step_outputs.append(y_hat.detach())
        self.test_step_targets.append(y.detach())
        return {"test_loss": loss}

    def on_test_epoch_end(self):
        all_preds = torch.cat(self.test_step_outputs, dim=0)  # (m_seqs, n_biosamples)
        all_targets = torch.cat(self.test_step_targets, dim=0)
        test_loss = self.criterion(all_preds, all_targets)

        preds_np = all_preds.cpu().numpy()
        targets_np = all_targets.cpu().numpy()
        num_samples, num_biosamples = preds_np.shape

        if num_samples > 1:
            pearson_vals, spearman_vals = [], []
            for i in range(num_biosamples):
                pearson_vals.append(pearsonr(preds_np[:, i], targets_np[:, i])[0])
                spearman_vals.append(spearmanr(preds_np[:, i], targets_np[:, i])[0])
            pearson_corr = float(np.mean(pearson_vals))
            spearman_corr = float(np.mean(spearman_vals))
        else:
            pearson_corr = pearsonr(preds_np[:, 0], targets_np[:, 0])[0]
            spearman_corr = spearmanr(preds_np[:, 0], targets_np[:, 0])[0]

        self.log("test_loss", test_loss, prog_bar=True, sync_dist=True)
        self.log("test_pearson", pearson_corr, prog_bar=True, sync_dist=True)
        self.log("test_spearman", spearman_corr, prog_bar=True, sync_dist=True)

        # clear buffers
        self.test_step_outputs.clear()
        self.test_step_targets.clear()
        return {"test_loss": test_loss}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), 
                                     lr=self.hparams.lr,
                                     weight_decay=self.hparams.weight_decay)

        warmup_steps = self.hparams.warmup_epochs * self.hparams.steps_per_train_epoch
        decay_steps = self.hparams.decay_epochs  * self.hparams.steps_per_train_epoch

        def lr_lambda(step):
            # 1) linear warmup
            if step < warmup_steps:
                return float(step) / max(1.0, warmup_steps)
            # 2) half-cycle cosine decay
            elif step < warmup_steps + decay_steps:
                decay_step = step - warmup_steps
                # cosine from 1→(min_lr/lr)
                cosine_decay = 0.5 * (1.0 + math.cos(math.pi * decay_step / decay_steps))
                return (1.0 - self.hparams.min_lr / self.hparams.lr) * cosine_decay \
                       + (self.hparams.min_lr / self.hparams.lr)
            # 3) after decay period, hold at min_lr
            else:
                return self.hparams.min_lr / self.hparams.lr

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            }
        }

class SharedRowColDownsampleStack(nn.Module):
    """
    Shared 1D stack: for ratios=[r1,r2,...], applies
    [BN1d -> Conv1d(k=stride=ri) -> GELU] sequentially to BOTH row & col
    streams using the SAME weights, then outer-sum (+bias), BN2d, GELU.

    seq_row: (B, R*prod(ratios), D)
    seq_col: (B, C*prod(ratios), D)
    returns: (B, out_ch_last, R, C)
    """
    def __init__(
        self,
        in_ch: int,                 # proj_dim
        ratios: List[int],
        hidden: int,                # mid channels for non-last stages
        out_ch_last: int,           # proj_dim
        use_bn1d: bool = True,
        use_bn2d: bool = True,
    ):
        super().__init__()
        stages = []
        ch_in = in_ch
        for i, r in enumerate(ratios):
            is_last = (i == len(ratios) - 1)
            ch_out  = (out_ch_last if is_last else hidden)
            stages.append(nn.Sequential(
                nn.BatchNorm1d(ch_in) if use_bn1d else nn.Identity(),
                nn.Conv1d(ch_in, ch_out, kernel_size=r, stride=r, padding=0, bias=False),
                nn.GELU(),
            ))
            ch_in = ch_out
        self.stages = nn.ModuleList(stages)

        self.out_ch_last = ch_in
        self.bias = nn.Parameter(torch.zeros(self.out_ch_last))
        self.bn2  = nn.BatchNorm2d(self.out_ch_last) if use_bn2d else nn.Identity()
        self.act  = nn.GELU()

    def forward(self, seq_row: torch.Tensor, seq_col: torch.Tensor) -> torch.Tensor:
        xr = seq_row.transpose(1, 2).contiguous()
        xc = seq_col.transpose(1, 2).contiguous()
        # apply the SAME stack to both streams (shared weights)
        for stage in self.stages:
            xr = stage(xr)
            xc = stage(xc)
        # xr: (B, out_ch, R), xc: (B, out_ch, C)
        out = xr.unsqueeze(-1) + xc.unsqueeze(-2) + self.bias.view(1, -1, 1, 1)  # (B, out_ch, R, C)
        out = self.bn2(out)
        return self.act(out)

class TransformerMulti2DNewRegressor(pl.LightningModule):
    def __init__(
        self,
        hic_input_dim: int,         # D_hic   (per-pixel Hi-C embedding)
        seq_input_dim: int,         # D_seq   (per-bin sequence embedding)
        output_dim: int,
        steps_per_train_epoch: int,
        n_rows: int,                # R  (Hi-C rows)
        n_cols: int,                # C  (Hi-C / seq columns)
        hic2seq_ratio: int,         # length ratio of 1 Hi-C bin to 1 sequence bin
        proj_dim: int = 512,        # token dim given to Transformer
        mlp_ratio: int = 4,
        num_heads: int = 16,
        num_layers: int = 4,
        pool_method: str = "cls",
        dropout_p: float = 0.1,
        lr: float = 1e-3,
        warmup_epochs: int = 1,
        decay_epochs: int = 9,
        min_lr: float = 1e-5,
        weight_decay: float = 1e-4,
        ratios: Optional[List[int]] = None,
        align_to: str = "cols",      # {"cols", "rows"}
        seq_pad_mode: str = "zeros", # {"zeros", "learned"}
        seq_scale: float = 1.0,
        add_layernorm: bool = False,
        ln_elementwise_affine: bool = False,
        **kwargs
    ):
        """
        Multi-modal 2-D Transformer regressor.
        """
        super().__init__()
        self.save_hyperparameters()

        hidden = 512

        if ratios is None:
            ratios = [5, 5, 5]
                 
        assert len(ratios) > 0, "Provide non-empty `ratios` (e.g., [5,5,5])."
        self.ratios = list(ratios)
        ratio_total = 1
        for r in self.ratios:
            assert isinstance(r, int) and r >= 1, f"Each ratio must be int>=1, got {r}"
            ratio_total *= r
        self.ratio_total = ratio_total
        assert self.ratio_total == hic2seq_ratio, (
            f"`hic2seq_ratio` ({hic2seq_ratio}) must equal product of `ratios` ({self.ratio_total})."
        )
        
        def _mlp(in_d: int, out_d: int) -> nn.Sequential:
            """2-layer (Lin → act → drop)x1  +  Lin(out)"""
            layers = []
            dims   = [in_d] + [hidden]*1 + [out_d]
            for i in range(len(dims) - 1):
                layers.append(nn.Linear(dims[i], dims[i+1]))
                if i < len(dims) - 2:
                    layers += [nn.Softplus(), nn.Dropout(dropout_p)]
            return nn.Sequential(*layers)
        
        # per-modality projection heads
        self.hic_mlp  = _mlp(hic_input_dim, proj_dim)
        self.seq_mlp  = _mlp(seq_input_dim, proj_dim)

        # fixed 2-D sin-cos positional embedding (with CLS)
        pe = get_2d_sincos_pos_embed_rectangle(
            proj_dim, (n_rows, n_cols), cls_token=True
        )
        self.register_buffer("pos_embed", torch.from_numpy(pe).float(), persistent=True)

        # CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, proj_dim))

        self.use_learned_pad = (seq_pad_mode == "learned")

        if self.use_learned_pad:
            self.pad_token = nn.Parameter(torch.zeros(proj_dim))  # (D,)
        
        self.seq_stack = SharedRowColDownsampleStack(
            in_ch=proj_dim,
            ratios=self.ratios,
            hidden=hidden,
            out_ch_last=proj_dim,
            use_bn1d=True,
            use_bn2d=True,
        )
        
        self.seq_scale = seq_scale
        
        self.pre_fuse_ln = None
        if add_layernorm:
            affine = False if ln_elementwise_affine is None else bool(ln_elementwise_affine)
            self.pre_fuse_ln = nn.LayerNorm(proj_dim, elementwise_affine=affine)

        # Transformer encoder
        dim_feedforward = int(proj_dim * mlp_ratio)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=proj_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout_p,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.final_norm = nn.LayerNorm(proj_dim)
        self.head = nn.Linear(proj_dim, output_dim)
        self.criterion = nn.MSELoss()

        self.validation_step_outputs = []
        self.validation_step_targets = []
        self.test_step_outputs = []
        self.test_step_targets = []

    def forward(self, hic: torch.Tensor, seq: torch.Tensor) -> torch.Tensor:
        """
        hic : (B, R, C, D_hic)       per-pixel Hi-C embeddings
        seq : (B, n_bins, D_seq)     per-bin sequence embeddings

        Returns
        -------
        (B, output_dim)
        """
        B, R, C, D_hic = hic.shape
        B2, num_bins, D_seq  = seq.shape
        assert B == B2, "Hi-C / sequence batch-size mismatch"
        assert R == self.hparams.n_rows and C == self.hparams.n_cols
        assert D_hic == self.hparams.hic_input_dim
        assert D_seq  == self.hparams.seq_input_dim
        assert R <= C, "Expected R ≤ C"

        ratio_total = self.ratio_total
        exp_cols_bins = C * ratio_total
        exp_rows_bins = R * ratio_total

        # 1. Projections
        hic_tok  = self.hic_mlp(hic)
        seq_proj = self.seq_mlp(seq)
        
        mode = self.hparams.align_to
        if mode == "cols":
            assert num_bins == exp_cols_bins, (
                f"Expected num_bins={exp_cols_bins}, got {num_bins}"
            )
            seq_col = seq_proj
            off_bins = (C - R) * ratio_total
            assert off_bins % 2 == 0, f"(C-R)*ratio_total must be even"
            
            start = off_bins // 2
            seq_row = seq_proj[:, start:start + exp_rows_bins, :]
        
        elif mode == "rows":
            assert num_bins == exp_rows_bins, (
                f"Expected num_bins={exp_rows_bins}, got {num_bins}"
            )
            seq_row = seq_proj
            seq_col = seq_proj
        
        else:
            raise ValueError(f"Unknown align_to={mode}!")

        if self.pre_fuse_ln is not None:
            hic_tok = self.pre_fuse_ln(hic_tok)
        
        # 2. Shared Stack (Outer sum)
        x = self.seq_stack(seq_row, seq_col)
        
        # 3. Padding
        if mode == "rows" and C > R:
            pad_cols = C - R
            pad_left = pad_cols // 2
            pad_right = pad_cols - pad_left
            
            x_unpad = x.permute(0, 2, 3, 1).contiguous()
            
            if self.pre_fuse_ln is not None:
                x_unpad = self.pre_fuse_ln(x_unpad)
        
            if self.use_learned_pad:
                pad_vec = self.pad_token.view(1, 1, 1, -1).to(x_unpad)  # (1, 1, 1, proj_dim)
                left  = pad_vec.expand(x_unpad.size(0), x_unpad.size(1), pad_left, x_unpad.size(3))
                right = pad_vec.expand(x_unpad.size(0), x_unpad.size(1), pad_right, x_unpad.size(3))
                seq_tok = torch.cat([left, x_unpad, right], dim=2)
            else:
                # zeros padding
                zeros_left  = x_unpad.new_zeros((x_unpad.size(0), x_unpad.size(1), pad_left,  x_unpad.size(3)))
                zeros_right = x_unpad.new_zeros((x_unpad.size(0), x_unpad.size(1), pad_right, x_unpad.size(3)))
                seq_tok = torch.cat([zeros_left, x_unpad, zeros_right], dim=2)
            
        else:
            seq_tok = x.permute(0, 2, 3, 1).contiguous()
            if self.pre_fuse_ln is not None:
                seq_tok = self.pre_fuse_ln(seq_tok)

        # 4. Fusion with Hi-C
        tok = hic_tok + self.seq_scale * seq_tok

        # 5. Add positional embedding
        pos = self.pos_embed[1:].reshape(R, C, -1)
        tok = tok + pos.to(tok.dtype)

        tok = tok.reshape(B, R * C, -1)
        cls = self.cls_token.expand(B, -1, -1)
        tok = torch.cat([cls, tok], dim=1)

        # 6. Pass through Transformer
        tok = self.transformer(tok)
        tok = self.final_norm(tok)

        # 7. Pooling
        if self.hparams.pool_method == "cls":
            pooled = tok[:, 0]                                       # (B, proj_dim)
        else:
            pooled = tok[:, 1:].mean(dim=1)

        return self.head(pooled)

    def training_step(self, batch, batch_idx):
        hic, seq, y, w = batch
        y_hat = self(hic, seq)
        loss = self.criterion(y_hat, y)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        hic, seq, y, w = batch
        y_hat = self(hic, seq)
        loss = self.criterion(y_hat, y)
        # Accumulate predictions and targets.
        self.validation_step_outputs.append(y_hat.detach())
        self.validation_step_targets.append(y.detach())
        return {"val_loss": loss}

    def on_validation_epoch_end(self):
        # Concatenate all predictions/targets
        all_preds = torch.cat(self.validation_step_outputs, dim=0)
        all_targets = torch.cat(self.validation_step_targets, dim=0)
        val_loss = self.criterion(all_preds, all_targets)

        preds_np = all_preds.cpu().numpy()
        targets_np = all_targets.cpu().numpy()
        num_samples, num_biosamples = preds_np.shape

        if num_samples > 1:
            pearson_vals, spearman_vals = [], []
            for i in range(num_biosamples):
                pearson_vals.append(pearsonr(preds_np[:, i], targets_np[:, i])[0])
                spearman_vals.append(spearmanr(preds_np[:, i], targets_np[:, i])[0])
            pearson_corr = float(np.mean(pearson_vals))
            spearman_corr = float(np.mean(spearman_vals))
        else:
            pearson_corr = pearsonr(preds_np[:, 0], targets_np[:, 0])[0]
            spearman_corr = spearmanr(preds_np[:, 0], targets_np[:, 0])[0]

        self.log("val_loss", val_loss, prog_bar=True, sync_dist=True)
        self.log("val_pearson", pearson_corr, prog_bar=True, sync_dist=True)
        self.log("val_spearman", spearman_corr, prog_bar=True, sync_dist=True)

        # clear buffers
        self.validation_step_outputs.clear()
        self.validation_step_targets.clear()
        return {"val_loss": val_loss}

    def test_step(self, batch, batch_idx):
        hic, seq, y, w = batch
        y_hat = self(hic, seq)
        loss = self.criterion(y_hat, y)
        self.test_step_outputs.append(y_hat.detach())
        self.test_step_targets.append(y.detach())
        return {"test_loss": loss}

    def on_test_epoch_end(self):
        all_preds = torch.cat(self.test_step_outputs, dim=0)
        all_targets = torch.cat(self.test_step_targets, dim=0)
        test_loss = self.criterion(all_preds, all_targets)

        preds_np = all_preds.cpu().numpy()
        targets_np = all_targets.cpu().numpy()
        num_samples, num_biosamples = preds_np.shape

        if num_samples > 1:
            pearson_vals, spearman_vals = [], []
            for i in range(num_biosamples):
                pearson_vals.append(pearsonr(preds_np[:, i], targets_np[:, i])[0])
                spearman_vals.append(spearmanr(preds_np[:, i], targets_np[:, i])[0])
            pearson_corr = float(np.mean(pearson_vals))
            spearman_corr = float(np.mean(spearman_vals))
        else:
            pearson_corr = pearsonr(preds_np[:, 0], targets_np[:, 0])[0]
            spearman_corr = spearmanr(preds_np[:, 0], targets_np[:, 0])[0]

        self.log("test_loss", test_loss, prog_bar=True, sync_dist=True)
        self.log("test_pearson", pearson_corr, prog_bar=True, sync_dist=True)
        self.log("test_spearman", spearman_corr, prog_bar=True, sync_dist=True)

        # clear buffers
        self.test_step_outputs.clear()
        self.test_step_targets.clear()
        return {"test_loss": test_loss}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), 
                                     lr=self.hparams.lr,
                                     weight_decay=self.hparams.weight_decay)

        warmup_steps = self.hparams.warmup_epochs * self.hparams.steps_per_train_epoch
        decay_steps = self.hparams.decay_epochs  * self.hparams.steps_per_train_epoch

        def lr_lambda(step):
            # 1) linear warmup
            if step < warmup_steps:
                return float(step) / max(1.0, warmup_steps)
            # 2) half-cycle cosine decay
            elif step < warmup_steps + decay_steps:
                decay_step = step - warmup_steps
                # cosine from 1→(min_lr/lr)
                cosine_decay = 0.5 * (1.0 + math.cos(math.pi * decay_step / decay_steps))
                return (1.0 - self.hparams.min_lr / self.hparams.lr) * cosine_decay \
                       + (self.hparams.min_lr / self.hparams.lr)
            # 3) after decay period, hold at min_lr
            else:
                return self.hparams.min_lr / self.hparams.lr

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            }
        }

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        hic, seq, y, w = batch
        hic = hic.to(device, non_blocking=True)
        seq = seq.to(device, non_blocking=True)
        y   = y.to(device, non_blocking=True)
        w   = w.to(device, non_blocking=True)
        return hic, seq, y, w
