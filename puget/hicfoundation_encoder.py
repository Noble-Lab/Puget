import os
import torch
import torch.nn as nn

from .HiCFoundation_model import Vision_Transformer_count as Vision_Transformer
from .HiCFoundation_model.pos_embed import interpolate_pos_embed_inputsize

class HiCFoundationEncoder(nn.Module):
    """
    Encoder-only wrapper (embed_depth = 0).
    Produces patch-grid embeddings from the ViT encoder; no decoder is used.
    """
    def __init__(self, model_name: str, img_size_hw: tuple, patch_size: int):
        """
        model_name: e.g. 'vit_base_patch16' (a symbol in Vision_Transformer.__dict__)
        img_size_hw: (H, W) image size fed to ViT (must match your submat window)
        patch_size:  e.g., 16 (must match the chosen backbone)
        """
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
        """
        img:         [B, 3, H, W] (already normalized like the original pipeline)
        total_count: [B] float tensor (global Hi-C count; backbone takes log10 internally)

        Returns:     [B, patch_rows, patch_cols, D] encoder grid embeddings
                      (CLS + COUNT tokens are dropped)
        """
        # x_backbone: [B, 2 + L, D]  (cls + count + patch_tokens)
        x_backbone = self.backbone.forward_features(img, total_count)

        # drop [CLS, COUNT] → keep only patch tokens
        x = x_backbone[:, self.num_additional_token:, :]                              # [B, L, D]
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
    """
    Build the encoder, resize its positional embeddings to (patch_rows, patch_cols),
    load encoder weights (drop head.* if incompatible), and return the model.
    """
    assert os.path.exists(checkpoint_path), f"Missing checkpoint: {checkpoint_path}"

    model = HiCFoundationEncoder(model_name, img_size_hw, patch_size)

    # Load checkpoint dict (supports {'model': ...} or flat state_dict)
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    ckpt_model = ckpt.get("model", ckpt)

    # Drop incompatible classification head params (not used in encoder-only inference)
    state_dict = model.backbone.state_dict()
    for k in ["head.weight", "head.bias"]:
        if k in ckpt_model and k in state_dict and ckpt_model[k].shape != state_dict[k].shape:
            # print(f"Removing key {k} from pretrained checkpoint")
            del ckpt_model[k]

    # Interpolate encoder positional embeddings for the rectangular patch grid
    interpolate_pos_embed_inputsize(
        model.backbone, ckpt_model,
        input_size=(model.patch_rows, model.patch_cols),  # (rows, cols) in patches
        use_decoder=False
    )

    # Load encoder weights only
    msg = model.backbone.load_state_dict(ckpt_model, strict=False)
    # print("Loaded encoder state (strict=False):", msg)

    # model.to(device)
    # model.eval()
    return model
