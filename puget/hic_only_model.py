from __future__ import annotations

import math
import os

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

from scipy.stats import rankdata

from .HiCFoundation_model import Vision_Transformer_count as Vision_Transformer
from .HiCFoundation_model.pos_embed import interpolate_pos_embed_inputsize
from .pos_embed import get_2d_sincos_pos_embed_rectangle


class HiCFoundationEncoder(nn.Module):
    def __init__(self, model_name: str, img_size_hw: tuple, patch_size: int):
        super().__init__()
        H, W = img_size_hw
        assert H % patch_size == 0 and W % patch_size == 0, \
            f"img_size must be multiples of patch_size; got {(H, W)} vs {patch_size}"

        self.patch_rows = H // patch_size
        self.patch_cols = W // patch_size

        # Build backbone
        self.backbone = Vision_Transformer.__dict__[model_name](img_size=(H, W))
        self.embed_dim = self.backbone.embed_dim

        self.num_additional_token = 2

    def forward(self, img: torch.Tensor, total_count: torch.Tensor):
        # x_backbone: [B, 2 + L, D]  (cls + count + patch_tokens)
        x_backbone = self.backbone.forward_features(img, total_count)

        # drop [CLS, COUNT] → keep only patch tokens
        x = x_backbone[:, self.num_additional_token:, :]  # [B, L, D]
        B, L, D = x.shape
        assert L == self.patch_rows * self.patch_cols, \
            f"L={L} != patch_rows*patch_cols={self.patch_rows*self.patch_cols}"

        # reshape to 2D patch grid
        x = x.view(B, self.patch_rows, self.patch_cols, D)    # [B, Pr, Pc, D]
        return x

# ------------------------------------------------------------------
# The following function is adapted from the HiCFoundation repository.
# Source: https://github.com/Noble-Lab/HiCFoundation/blob/main/inference/main_worker.py
# License: Apache License 2.0
# ------------------------------------------------------------------
def load_hic_encoder_only(
    checkpoint_path: str,
    model_name: str,
    img_size_hw: tuple,
    patch_size: int,
) -> HiCFoundationEncoder:
    assert os.path.exists(checkpoint_path), f"Missing checkpoint: {checkpoint_path}"

    model = HiCFoundationEncoder(model_name, img_size_hw, patch_size)

    # Load checkpoint dict (supports {'model': ...} or flat state_dict)
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    ckpt_model = ckpt.get("model", ckpt)

    # Drop incompatible classification head params (not used in encoder-only inference)
    state_dict = model.backbone.state_dict()
    for k in ["head.weight", "head.bias"]:
        if k in ckpt_model and k in state_dict and ckpt_model[k].shape != state_dict[k].shape:
            del ckpt_model[k]

    # Interpolate encoder positional embeddings for the rectangular patch grid
    interpolate_pos_embed_inputsize(
        model.backbone, ckpt_model,
        input_size=(model.patch_rows, model.patch_cols),  # (rows, cols) in patches
        use_decoder=False
    )

    # Load encoder weights only
    model.backbone.load_state_dict(ckpt_model, strict=False)
    return model


class Transformer2DRegressorNet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        n_rows: int,
        n_cols: int,
        proj_dim: int = 512,
        mlp_ratio: float = 4.0,
        num_heads: int = 16,
        num_layers: int = 4,
        pool_method: str = "cls",
        dropout_p: float = 0.1,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.n_rows = int(n_rows)
        self.n_cols = int(n_cols)
        self.pool_method = str(pool_method)

        self.input_proj = nn.Linear(self.input_dim, proj_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, proj_dim))

        pe = get_2d_sincos_pos_embed_rectangle(
            proj_dim, (self.n_rows, self.n_cols), cls_token=True,
        )
        self.register_buffer("pos_embed", torch.from_numpy(pe).float(), persistent=True)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=proj_dim,
            nhead=num_heads,
            dim_feedforward=int(proj_dim * mlp_ratio),
            dropout=dropout_p,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.final_norm = nn.LayerNorm(proj_dim)
        self.head = nn.Linear(proj_dim, self.output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, R, C, D = x.shape
        assert R == self.n_rows and C == self.n_cols, \
            f"Expected spatial ({self.n_rows}, {self.n_cols}), got ({R}, {C})"
        assert D == self.input_dim, f"Expected input_dim={self.input_dim}, got {D}"

        x = self.input_proj(x)
        pos = self.pos_embed[1:].reshape(R, C, -1)
        x = x + pos.to(x.dtype)
        x = x.reshape(B, R * C, -1)

        cls = self.cls_token.to(x.dtype).expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = self.transformer(x)
        x = self.final_norm(x)

        if self.pool_method == "cls":
            pooled = x[:, 0]
        else:
            pooled = x[:, 1:].mean(dim=1)
        return self.head(pooled)


def safe_pearson(x: np.ndarray, y: np.ndarray) -> float:
    keep = np.isfinite(x) & np.isfinite(y)
    x, y = x[keep], y[keep]
    if x.size < 2 or np.std(x) <= 1.0e-12 or np.std(y) <= 1.0e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def safe_spearman(x: np.ndarray, y: np.ndarray) -> float:
    keep = np.isfinite(x) & np.isfinite(y)
    x, y = x[keep], y[keep]
    if x.size < 2:
        return float("nan")
    return safe_pearson(rankdata(x), rankdata(y))


def mean_group_correlation(pred, target, groups, correlation) -> float:
    values = []
    for group in np.unique(groups):
        selected = groups == group
        value = correlation(pred[selected], target[selected])
        if np.isfinite(value):
            values.append(value)
    return float(np.mean(values)) if values else float("nan")


def lr_lambda(warmup_steps: int, decay_steps: int, base_lr: float, min_lr: float):
    def schedule(step: int) -> float:
        if warmup_steps and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        fraction = min(1.0, (step - warmup_steps) / max(1, decay_steps))
        cosine = 0.5 * (1.0 + math.cos(math.pi * fraction))
        return (min_lr + (base_lr - min_lr) * cosine) / base_lr

    return schedule

OUTPUT_ACTIVATIONS = ("linear", "softplus")


def apply_output_activation(output: torch.Tensor, activation: str) -> torch.Tensor:
    if activation == "linear":
        return output
    if activation == "softplus":
        return F.softplus(output)
    raise ValueError(
        f"output_activation must be one of {OUTPUT_ACTIVATIONS}, got {activation!r}")


class FrozenHiCFoundationRegressor(nn.Module):
    def __init__(
        self,
        *,
        encoder_ckpt_path: str,
        model_name: str = "vit_large_patch16",
        input_size: int = 512,
        patch_size: int = 16,
        embed_dim: int = 1024,
        grid_rows: int = 32,
        grid_cols: int = 32,
        output_dim: int = 1,
        proj_dim: int = 512,
        mlp_ratio: float = 4.0,
        num_heads: int = 16,
        num_layers: int = 4,
        pool_method: str = "cls",
        dropout: float = 0.1,
        encoder_fp32: bool = False,
        output_activation: str = "softplus",
    ):
        super().__init__()
        input_size = int(input_size)
        patch_size = int(patch_size)
        expected_grid = input_size // patch_size
        if input_size % patch_size:
            raise ValueError(f"input_size={input_size} is not divisible by patch_size={patch_size}")
        if (int(grid_rows), int(grid_cols)) != (expected_grid, expected_grid):
            raise ValueError(
                f"Configured grid {(grid_rows, grid_cols)} does not match the complete "
                f"{input_size} / {patch_size} encoder grid {(expected_grid, expected_grid)}"
            )

        print(f"Loading frozen HiCFoundation encoder from {encoder_ckpt_path}")
        self.encoder = load_hic_encoder_only(
            checkpoint_path=encoder_ckpt_path,
            model_name=model_name,
            img_size_hw=(input_size, input_size),
            patch_size=patch_size,
        )
        if int(self.encoder.embed_dim) != int(embed_dim):
            raise ValueError(
                f"Configured embed_dim={embed_dim}, but {model_name} produces "
                f"embed_dim={self.encoder.embed_dim}"
            )
        self.encoder.requires_grad_(False)
        self.encoder.eval()
        self.encoder_fp32 = bool(encoder_fp32)
        if output_activation not in OUTPUT_ACTIVATIONS:
            raise ValueError(
                f"output_activation must be one of {OUTPUT_ACTIVATIONS}, "
                f"got {output_activation!r}")
        self.output_activation = str(output_activation)
        self.grid_rows = int(grid_rows)
        self.grid_cols = int(grid_cols)

        # This is the same decoder class used by Puget, initialized from
        # scratch for this supervised held-out-cell experiment.
        self.decoder = Transformer2DRegressorNet(
            input_dim=int(embed_dim),
            output_dim=int(output_dim),
            n_rows=self.grid_rows,
            n_cols=self.grid_cols,
            proj_dim=int(proj_dim),
            mlp_ratio=float(mlp_ratio),
            num_heads=int(num_heads),
            num_layers=int(num_layers),
            pool_method=str(pool_method),
            dropout_p=float(dropout),
        )

    def train(self, mode: bool = True):
        # Lightning calls train() recursively at epoch boundaries. Override it
        # so the frozen encoder never enables dropout or other training-time
        # behavior, while the decoder still follows the requested mode.
        super().train(mode)
        self.encoder.eval()
        return self

    def forward(self, image: torch.Tensor, total_count: torch.Tensor) -> torch.Tensor:
        self.encoder.eval()
        # Lightning's outer autocast remains active for the decoder. Temporarily
        # disable it here so the frozen HiCFoundation backbone and its count
        # positional embedding execute in FP32.
        if self.encoder_fp32:
            with torch.autocast(device_type=image.device.type, enabled=False):
                with torch.no_grad():
                    grid = self.encoder(image.float(), total_count.float())
        else:
            with torch.no_grad():
                grid = self.encoder(image, total_count)
        if tuple(grid.shape[1:3]) != (self.grid_rows, self.grid_cols):
            raise RuntimeError(
                f"HiCFoundation returned spatial grid {tuple(grid.shape[1:3])}; expected "
                f"the complete {(self.grid_rows, self.grid_cols)} grid"
            )
        output = self.decoder(grid)
        return apply_output_activation(output, self.output_activation)


class HiCFoundationLit(pl.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        *,
        lr: float,
        min_lr: float,
        weight_decay: float,
        warmup_epochs: int,
        decay_epochs: int,
        steps_per_epoch: int,
        optimizer: str = "adam",
    ):
        super().__init__()
        self.model = model
        self.loss_fn = nn.MSELoss()
        self.save_hyperparameters(ignore=["model"])
        self.val_buffers = ([], [], [])

    def forward(self, image: torch.Tensor, total_count: torch.Tensor) -> torch.Tensor:
        return self.model(image, total_count)

    def on_train_epoch_start(self) -> None:
        self.model.encoder.eval()

    def training_step(self, batch, batch_idx):
        image, total_count, target, _ = batch
        prediction = self(image, total_count)
        loss = self.loss_fn(prediction, target)
        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True,
            batch_size=image.shape[0],
        )
        return loss

    def validation_step(self, batch, batch_idx):
        image, total_count, target, metadata = batch
        prediction = self(image, total_count)
        loss = self.loss_fn(prediction, target)
        output_buffer, target_buffer, meta_buffer = self.val_buffers
        output_buffer.append(prediction.detach().float().cpu())
        target_buffer.append(target.detach().float().cpu())
        meta_buffer.extend((int(meta[1]), int(meta[2])) for meta in metadata)
        return {"val_loss": loss}

    def on_validation_epoch_end(self):
        outputs, targets, metadata = self.val_buffers
        if not outputs:
            return
        prediction = torch.cat(outputs).numpy().reshape(-1)
        target = torch.cat(targets).numpy().reshape(-1)
        metadata_array = np.asarray(metadata, dtype=np.int64)
        biosamples, genes = metadata_array[:, 0], metadata_array[:, 1]

        loss = float(np.mean((prediction - target) ** 2))
        pearson = mean_group_correlation(prediction, target, biosamples, safe_pearson)
        spearman = mean_group_correlation(prediction, target, biosamples, safe_spearman)
        gene_pearson = mean_group_correlation(prediction, target, genes, safe_pearson)
        gene_spearman = mean_group_correlation(prediction, target, genes, safe_spearman)

        self.log("val_loss", loss, prog_bar=True, on_epoch=True)
        self.log("val_pearson", pearson, prog_bar=True, on_epoch=True)
        self.log("val_spearman", spearman, on_epoch=True)
        self.log(
            "val_gene_pearson_bio", gene_pearson,
            prog_bar=True, on_epoch=True,
        )
        self.log("val_gene_spearman_bio", gene_spearman, on_epoch=True)

        outputs.clear()
        targets.clear()
        metadata.clear()

    def configure_optimizers(self):
        if any(parameter.requires_grad for parameter in self.model.encoder.parameters()):
            raise RuntimeError("Frozen HiCFoundation encoder unexpectedly has trainable parameters")
        # Only decoder parameters enter Adam/AdamW, so neither weight decay nor
        # optimizer state is ever applied to the frozen encoder.
        trainable = [
            parameter for parameter in self.model.decoder.parameters() if parameter.requires_grad
        ]
        optimizer_name = str(self.hparams.optimizer).lower()
        optimizer_cls = {"adam": torch.optim.Adam, "adamw": torch.optim.AdamW}.get(optimizer_name)
        if optimizer_cls is None:
            raise ValueError(f"Unsupported optimizer {optimizer_name!r}; use adam or adamw")
        optimizer = optimizer_cls(
            trainable,
            lr=float(self.hparams.lr),
            weight_decay=float(self.hparams.weight_decay),
        )
        steps = int(self.hparams.steps_per_epoch)
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lr_lambda(
                int(self.hparams.warmup_epochs) * steps,
                int(self.hparams.decay_epochs) * steps,
                float(self.hparams.lr),
                float(self.hparams.min_lr),
            ),
        )
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}


def parameter_count(model: nn.Module) -> tuple[int, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return total, trainable
