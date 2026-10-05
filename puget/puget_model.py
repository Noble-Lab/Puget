import math

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import pearsonr, spearmanr

from .pos_embed import get_1d_sincos_pos_embed


class PointwiseHiCBiasCNN(nn.Module):
    def __init__(self, out_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=1)
        self.conv3 = nn.Conv2d(64, int(out_channels), kernel_size=1)
        self.act = nn.ReLU()
        nn.init.zeros_(self.conv3.weight)
        nn.init.zeros_(self.conv3.bias)

    def forward(self, x):
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        return self.conv3(x)


class Down1D(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential()

    def forward(self, x):
        return self.net(x.transpose(1, 2)).transpose(1, 2)


class BiasTransformerBlock(nn.Module):
    def __init__(self, dim: int, heads: int, mlp_ratio: float = 4.0,
                 dropout: float = 0.2):
        super().__init__()
        if dim % heads:
            raise ValueError(f"dim {dim} is not divisible by heads {heads}")
        self.h = int(heads)
        self.dk = dim // int(heads)
        self.scale = 1.0 / math.sqrt(self.dk)

        self.attn_norm = nn.LayerNorm(dim)
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)

        self.ffn_norm = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, dim))
        self.drop = nn.Dropout(dropout)

    def forward(self, x, bias: torch.Tensor):
        batch, tokens, dim = x.shape
        normalized = self.attn_norm(x)
        q = self.q(normalized).view(
            batch, tokens, self.h, self.dk).transpose(1, 2)
        k = self.k(normalized).view(
            batch, tokens, self.h, self.dk).transpose(1, 2)
        v = self.v(normalized).view(
            batch, tokens, self.h, self.dk).transpose(1, 2)

        logits = (q @ k.transpose(-2, -1)) * self.scale
        logits = logits + bias
        attention = logits.softmax(dim=-1)
        context = (attention @ v).transpose(1, 2).reshape(batch, tokens, dim)

        x = x + self.drop(self.o(context))
        x = x + self.ffn(self.ffn_norm(x))
        return x


class BiasModel(nn.Module):
    def __init__(
        self,
        *,
        seq_input_dim: int,
        n_cols: int = 192,
        proj_dim: int = 512,
        heads: int = 8,
        num_layers: int = 4,
        mlp_ratio: float = 4.0,
        dropout_p: float = 0.2,
        hidden: int = 512,
        promoter_tokens: int = 4,
        input_norm: str = "layernorm",
        output_dim: int = 1,
        output_activation: str = "softplus",
    ):
        super().__init__()
        if input_norm not in ("none", "layernorm"):
            raise ValueError(f"input_norm must be none|layernorm, got {input_norm!r}")
        if output_activation not in ("linear", "softplus"):
            raise ValueError(
                "output_activation must be linear|softplus, "
                f"got {output_activation!r}")

        self.input_norm = str(input_norm)
        self.output_activation = str(output_activation)
        self.N = int(n_cols)
        self.heads = int(heads)
        self.num_layers = int(num_layers)
        self.promoter_tokens = int(promoter_tokens)

        self.p0 = (self.N - self.promoter_tokens) // 2

        self.in_norm = (
            nn.LayerNorm(int(seq_input_dim))
            if self.input_norm == "layernorm" else nn.Identity())
        self.seq_mlp = nn.Sequential(
            nn.Linear(int(seq_input_dim), hidden), nn.GELU(), nn.Dropout(dropout_p),
            nn.Linear(hidden, proj_dim))
        self.down = Down1D()
        position = get_1d_sincos_pos_embed(proj_dim, self.N)
        self.register_buffer(
            "pos_embed", torch.from_numpy(position).float(), persistent=True)

        self.blocks = nn.ModuleList([
            BiasTransformerBlock(proj_dim, self.heads, mlp_ratio, dropout_p)
            for _ in range(self.num_layers)])
        self.hic_cnn = nn.ModuleList([
            PointwiseHiCBiasCNN(self.heads)
            for _ in range(self.num_layers)])
        self.hic_scale = nn.Parameter(torch.ones(self.num_layers))
        self.blur = nn.Identity()  # Paper configuration: blur_sigma = 0.0.
        self.out_norm = nn.LayerNorm(proj_dim)
        self.head = nn.Linear(proj_dim, int(output_dim))

    def forward(self, img: torch.Tensor, seq: torch.Tensor) -> torch.Tensor:
        dtype = self.pos_embed.dtype
        img, seq = img.to(dtype), seq.to(dtype)
        _batch, num_bins, _features = seq.shape
        if num_bins != self.N:
            raise ValueError(
                f"expected {self.N} sequence bins, got {num_bins}")

        tokens = (
            self.down(self.seq_mlp(self.in_norm(seq)))
            + self.pos_embed.to(seq.dtype))
        hic = self.blur(img)

        for layer in range(self.num_layers):
            raw_bias = self.hic_cnn[layer](hic)
            bias = self.hic_scale[layer] * raw_bias
            tokens = self.blocks[layer](tokens, bias)

        pooled = self.out_norm(
            tokens[:, self.p0:self.p0 + self.promoter_tokens, :].mean(dim=1))
        output = self.head(pooled)
        return F.softplus(output) if self.output_activation == "softplus" else output


def mean_corr_per_biosample(preds, targets, bio_idx):
    preds = np.asarray(preds).reshape(-1)
    targets = np.asarray(targets).reshape(-1)
    bio_idx = np.asarray(bio_idx).reshape(-1)
    pear, spear = [], []
    for b in np.unique(bio_idx):
        m = bio_idx == b
        if int(m.sum()) < 2:
            continue
        r = pearsonr(preds[m], targets[m])[0]
        s = spearmanr(preds[m], targets[m])[0]
        if np.isfinite(r):
            pear.append(r)
        if np.isfinite(s):
            spear.append(s)
    if not pear:
        return float("nan"), float("nan")
    return float(np.mean(pear)), float(np.mean(spear))


def mean_corr_per_gene(preds, targets, gene_idx):
    preds = np.asarray(preds).reshape(-1)
    targets = np.asarray(targets).reshape(-1)
    gene_idx = np.asarray(gene_idx).reshape(-1)
    pear, spear = [], []
    for g in np.unique(gene_idx):
        m = gene_idx == g
        if int(m.sum()) < 2:
            continue  # need >= 2 biosamples to define a correlation
        r = pearsonr(preds[m], targets[m])[0]
        s = spearmanr(preds[m], targets[m])[0]
        if np.isfinite(r):
            pear.append(r)
        if np.isfinite(s):
            spear.append(s)
    pm = float(np.mean(pear)) if pear else float("nan")
    sm = float(np.mean(spear)) if spear else float("nan")
    return pm, sm


def make_lr_lambda(warmup_steps: int, decay_steps: int, base_lr: float, min_lr: float):
    def fn(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        t = min(1.0, (step - warmup_steps) / max(1, decay_steps))
        cos = 0.5 * (1.0 + math.cos(math.pi * t))
        return (min_lr + (base_lr - min_lr) * cos) / base_lr
    return fn


class VariantLit(pl.LightningModule):
    def __init__(self, model, *, lr=5e-5, min_lr=1e-7, weight_decay=1e-2,
                 warmup_epochs=2, decay_epochs=18, steps_per_train_epoch=1):
        super().__init__()
        self.model = model
        self.criterion = nn.MSELoss()
        self.save_hyperparameters(ignore=["model"])

        self._val = ([], [], [])    # outputs, targets, meta
        self.val_gene_mask = None

    def forward(self, img, seq):
        return self.model(img, seq)

    def training_step(self, batch, batch_idx):
        img, seq, y, _ = batch
        loss = self.criterion(self(img, seq), y)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True,
                 batch_size=img.shape[0])
        return loss

    def validation_step(self, batch, batch_idx):
        img, seq, y, meta = batch
        y_hat = self(img, seq)
        loss = self.criterion(y_hat, y)
        buf = self._val
        buf[0].append(y_hat.detach())
        buf[1].append(y.detach())
        # (bi, j) = biosample-subset position and gene index, carried so the
        # epoch-end metric can regroup by either axis and survive dropped windows.
        buf[2].append(torch.tensor([[int(m[5]), int(m[6])] for m in meta],
                                   device=self.device, dtype=torch.long))
        return {"val_loss": loss}

    def on_validation_epoch_end(self):
        self._epoch_end(self._val)

    def _gather(self, buf):
        preds = torch.cat(buf[0], 0)
        tgts = torch.cat(buf[1], 0)
        meta = torch.cat(buf[2], 0)
        world = int(getattr(self.trainer, "world_size", 1) or 1)
        if world > 1:
            sizes = self.all_gather(torch.tensor([preds.shape[0]], device=self.device)).reshape(-1)
            max_n = int(sizes.max().item())
            if preds.shape[0] < max_n:
                pad = max_n - preds.shape[0]
                preds = F.pad(preds, (0, 0, 0, pad))
                tgts = F.pad(tgts, (0, 0, 0, pad))
                meta = F.pad(meta, (0, 0, 0, pad))
            gp, gt, gm = self.all_gather(preds), self.all_gather(tgts), self.all_gather(meta)
            keep = [(gp[r, :int(sizes[r])], gt[r, :int(sizes[r])], gm[r, :int(sizes[r])])
                    for r in range(gp.shape[0])]
            preds = torch.cat([k[0] for k in keep], 0)
            tgts = torch.cat([k[1] for k in keep], 0)
            meta = torch.cat([k[2] for k in keep], 0)
        return preds, tgts, meta

    def _epoch_end(self, buf):
        if not buf[0]:
            return
        preds, tgts, meta = self._gather(buf)
        loss = self.criterion(preds, tgts)
        p = preds.float().cpu().numpy().reshape(-1)
        t = tgts.float().cpu().numpy().reshape(-1)
        m = meta.cpu().numpy()
        bios, genes = m[:, 0], m[:, 1]

        pear, spear = mean_corr_per_biosample(p, t, bios)
        self.log("val_loss", loss, prog_bar=True, sync_dist=True, on_epoch=True)
        self.log("val_pearson", pear, prog_bar=True, sync_dist=True, on_epoch=True)
        self.log("val_spearman", spear, prog_bar=False, sync_dist=True, on_epoch=True)

        mask = self.val_gene_mask
        if mask is not None:
            sel = np.asarray(mask, dtype=bool)[genes]
            pear_g, spear_g = (mean_corr_per_gene(p[sel], t[sel], genes[sel])
                               if sel.any() else (float("nan"), float("nan")))
        else:
            pear_g = spear_g = float("nan")
        self.log("val_gene_pearson_bio", pear_g, prog_bar=True, sync_dist=True, on_epoch=True)
        self.log("val_gene_spearman_bio", spear_g, prog_bar=False, sync_dist=True, on_epoch=True)

        for i, value in enumerate(self.model.hic_scale.detach().float().cpu().tolist()):
            self.log(f"val_hic_gate_l{i}", value, prog_bar=False,
                     sync_dist=False, on_epoch=True)

        for lst in buf:
            lst.clear()

    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=float(self.hparams.lr), weight_decay=float(self.hparams.weight_decay))
        spe = int(self.hparams.steps_per_train_epoch)
        assert spe >= 1, "steps_per_train_epoch must be >= 1"
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lr_lambda=make_lr_lambda(int(self.hparams.warmup_epochs) * spe,
                                          int(self.hparams.decay_epochs) * spe,
                                          float(self.hparams.lr), float(self.hparams.min_lr)))
        return {"optimizer": opt,
                "lr_scheduler": {"scheduler": sched, "interval": "step", "frequency": 1}}
