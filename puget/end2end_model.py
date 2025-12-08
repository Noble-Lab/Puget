import torch
import torch.nn as nn
from enformer_pytorch import Enformer
from .decoder_model import Transformer1DRegressor, Transformer2DRegressor, TransformerMulti2DNewRegressor
from .hicfoundation_encoder import load_hic_encoder_only

class CenterCrop(nn.Module):
    def __init__(self, target_length: int):
        super().__init__()
        self.target_length = target_length

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, dim = x.shape
        if seq_len < self.target_length:
            raise ValueError(
                f"Input length {seq_len} is smaller than target crop length {self.target_length}"
            )
        
        crop_start = (seq_len - self.target_length) // 2
        crop_end = crop_start + self.target_length
        return x[:, crop_start:crop_end, :]

class CenterCrop2D(nn.Module):
    def __init__(self, target_rows: int, target_cols: int):
        super().__init__()
        self.target_rows = int(target_rows)
        self.target_cols = int(target_cols)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"CenterCrop2D expects 4D [B,R,C,D], got {tuple(x.shape)}")
        B, R, C, D = x.shape
        if R < self.target_rows or C < self.target_cols:
            raise ValueError(
                f"Input spatial {(R, C)} is smaller than target crop {(self.target_rows, self.target_cols)}"
            )
        r0 = (R - self.target_rows) // 2
        c0 = (C - self.target_cols) // 2
        return x[:, r0:r0 + self.target_rows, c0:c0 + self.target_cols, :]

class SequenceEmbeddingGenerator(nn.Module):
    """
    Wraps Enformer + CenterCrop for generating offline embeddings.
    Loads weights, freezes parameters, and returns cropped embeddings.
    """
    def __init__(self, enformer_ckpt_path: str, crop_seq_bins: int):
        super().__init__()
        print(f"Loading Enformer from {enformer_ckpt_path}...")
        self.enformer = Enformer.from_hparams()
        
        # Load weights
        state = torch.load(enformer_ckpt_path, map_location="cpu")
        if hasattr(self.enformer, "module"):
            self.enformer.module.load_state_dict(state)
        else:
            self.enformer.load_state_dict(state)
        
        self.enformer.eval()
        for p in self.enformer.parameters():
            p.requires_grad = False
        
        self.center_crop = CenterCrop(crop_seq_bins)
        print(f"Initialized CenterCrop with target length: {crop_seq_bins}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, L, 4) One-Hot encoded sequence
        Returns: (B, crop_seq_bins, 3072)
        """
        embeddings = self.enformer(x, return_only_embeddings=True)
        cropped_embeddings = self.center_crop(embeddings)
        return cropped_embeddings

class EnformerGenePredictor(nn.Module):
    def __init__(self, enformer_ckpt_path: str, regressor_ckpt_path: str):
        super().__init__()

        # 1. Load the pre-trained Enformer model
        print("Loading Enformer...")
        self.enformer = Enformer.from_hparams()
        state = torch.load(enformer_ckpt_path, map_location="cpu")
        self.enformer.load_state_dict(state)
        del state
        
        # 2. Load Regressor ckpt
        print("Loading Transformer1DRegressor...")
        self.regressor = Transformer1DRegressor.load_from_checkpoint(
            checkpoint_path=regressor_ckpt_path,
            map_location="cpu"
        )

        # 3. Create the intermediate center crop layer
        self.n_bins_for_regressor = self.regressor.hparams.n_bins
        self.center_crop = CenterCrop(self.n_bins_for_regressor)
        print(f"Initialized center crop to extract {self.n_bins_for_regressor} bins.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs an end-to-end prediction from a one-hot DNA sequence.
        x: one-hot encoded DNA sequence, shape (B, L, 4)
        """   
        # Step 1: Get embeddings from Enformer
        embeddings = self.enformer(x, return_only_embeddings=True)
        
        # Step 2: Crop the center of the embeddings
        cropped_embeddings = self.center_crop(embeddings)
        
        # Step 3: Get final predictions from the regressor
        predictions = self.regressor(cropped_embeddings)
        
        return predictions

class HiCEmbeddingGenerator(nn.Module):
    """
    Wraps HiCFoundation Encoder + CenterCrop2D for generating offline embeddings.
    """
    def __init__(
        self,
        encoder_ckpt_path: str,
        model_name: str,
        img_size_hw: tuple,
        patch_size: int,
        crop_rows: int,
        crop_cols: int
    ):
        super().__init__()
        print("Loading HiCFoundation encoder...")
        self.encoder = load_hic_encoder_only(
            checkpoint_path=encoder_ckpt_path,
            model_name=model_name,
            img_size_hw=img_size_hw,
            patch_size=patch_size,
        )
        
        self.encoder.eval()
        for p in self.encoder.parameters():
            p.requires_grad = False

        self.center_crop_2d = CenterCrop2D(crop_rows, crop_cols)
        print(f"Initialized 2D CenterCrop to ({crop_rows}, {crop_cols})")

    def forward(self, img: torch.Tensor, total_count: torch.Tensor) -> torch.Tensor:
        """
        img: [B, 3, H, W]
        Returns: [B, crop_rows, crop_cols, D]
        """
        enc_grid = self.encoder(img, total_count)
        cropped = self.center_crop_2d(enc_grid)
        return cropped
    
class HiCFoundationGenePredictor(nn.Module):
    def __init__(
        self,
        encoder_ckpt_path: str,
        model_name: str,
        img_size_hw: tuple,
        patch_size: int,
        regressor_ckpt_path: str,
        init_decoder="pretrained"
    ):
        super().__init__()

        # 1) Encoder
        print("Loading HiCFoundation encoder...")
        self.encoder = load_hic_encoder_only(
            checkpoint_path=encoder_ckpt_path,
            model_name=model_name,
            img_size_hw=img_size_hw,
            patch_size=patch_size,
        )

        # 2) Regressor
        print("Loading Transformer2DRegressor...")
        self.regressor = Transformer2DRegressor.load_from_checkpoint(
            checkpoint_path=regressor_ckpt_path,
            map_location="cpu",
        )

        if init_decoder == "scratch":
            def _reset(m):
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif isinstance(m, nn.LayerNorm):
                    nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
            self.regressor.apply(_reset)
            if hasattr(self.regressor, "cls_token"):
                nn.init.normal_(self.regressor.cls_token, std=0.02)
            if hasattr(self.regressor, "head") and isinstance(self.regressor.head, nn.Linear):
                nn.init.xavier_uniform_(self.regressor.head.weight)
                if self.regressor.head.bias is not None:
                    nn.init.zeros_(self.regressor.head.bias)

        # 3) Center-crop to regressor's expected spatial size
        self.crop_rows = int(self.regressor.hparams.n_rows)
        self.crop_cols = int(self.regressor.hparams.n_cols)
        self.center_crop_2d = CenterCrop2D(self.crop_rows, self.crop_cols)
        print(f"Initialized 2D center crop to ({self.crop_rows}, {self.crop_cols}).")

    def forward(self, img: torch.Tensor, total_count: torch.Tensor) -> torch.Tensor:
        """
        img:         [B, 3, H, W]  (normalized as in backbone)
        total_count: [B] float tensor (global Hi-C count; backbone handles log10 internally)
        returns:     [B, output_dim]
        """
        # Encoder → [B, Pr, Pc, D]
        enc_grid = self.encoder(img, total_count)

        # Center-crop spatial grid → [B, n_rows, n_cols, D]
        cropped = self.center_crop_2d(enc_grid)

        # Regressor → [B, output_dim]
        predictions = self.regressor(cropped)
        return predictions

class PugetGenePredictor(nn.Module):
    def __init__(
        self,
        enformer_ckpt_path: str,
        hic_encoder_ckpt_path: str,
        hic_model_name: str,
        img_size_hw: tuple,
        patch_size: int,
        regressor_ckpt_path: str,
    ):
        super().__init__()

        # 1) Sequence encoder (Enformer) → embeddings
        print("Loading Enformer...")
        self.enformer = Enformer.from_hparams()
        enformer_state = torch.load(enformer_ckpt_path, map_location="cpu")
        if hasattr(self.enformer, "module"):
            self.enformer.module.load_state_dict(enformer_state)
        else:
            self.enformer.load_state_dict(enformer_state)
        del enformer_state

        # 2) Hi-C encoder (HiCFoundation)
        print("Loading HiCFoundation encoder...")
        self.hic_encoder = load_hic_encoder_only(
            checkpoint_path=hic_encoder_ckpt_path,
            model_name=hic_model_name,
            img_size_hw=img_size_hw,
            patch_size=patch_size,
        )

        # 3) Multi-2D regressor loading
        print("Loading TransformerMulti2DNewRegressor...")
        self.regressor: TransformerMulti2DNewRegressor = TransformerMulti2DNewRegressor.load_from_checkpoint(
            checkpoint_path=regressor_ckpt_path,
            map_location="cpu",
        )

        # 4) CROP modules
        self.n_rows = int(self.regressor.hparams.n_rows)
        self.n_cols = int(self.regressor.hparams.n_cols)
        self.hic2seq_ratio = int(self.regressor.hparams.hic2seq_ratio)
        self.align_to = str(self.regressor.hparams.align_to)

        if self.align_to == "cols":
            target_seq_bins = self.n_cols * self.hic2seq_ratio
        elif self.align_to == "rows":
            target_seq_bins = self.n_rows * self.hic2seq_ratio
        else:
            raise ValueError(f"Unknown align_to={self.align_to}")

        self.center_crop_seq = CenterCrop(target_seq_bins)
        print(f"Initialized 1D center crop to extract {target_seq_bins} bins.")
        self.center_crop_hic = CenterCrop2D(self.n_rows, self.n_cols)
        print(f"Initialized 2D center crop to ({self.n_rows}, {self.n_cols}).")

    def forward(
        self,
        seq_tokens: torch.Tensor,      # (B, L, 4) or (B, L)
        img: torch.Tensor,             # (B, 3, H, W)
        total_count: torch.Tensor,     # (B,)
    ) -> torch.Tensor:
        # Sequence branch → embeddings (B, N_seq_full, D_seq_full)
        seq_emb = self.enformer(seq_tokens, return_only_embeddings=True)   # (B, N, D_seq)
        seq_emb = self.center_crop_seq(seq_emb)                            # (B, target_seq_bins, D_seq)

        # Hi-C branch → grid embeddings
        hic_grid = self.hic_encoder(img, total_count)                      # (B, Pr, Pc, D_hic)
        hic_grid = self.center_crop_hic(hic_grid)                          # (B, n_rows, n_cols, D_hic)

        # Regress
        predictions = self.regressor(hic_grid, seq_emb)
        return predictions
