# ---------------------------------------------------------------------------------
# This file is adapted from the HiCFoundation repository.
# Source: https://github.com/Noble-Lab/HiCFoundation/blob/main/model/Vision_Transformer_count.py
# License: Apache License 2.0
# ---------------------------------------------------------------------------------

from functools import partial

import torch
import torch.nn as nn

import timm.models.vision_transformer

from .pos_embed import convert_count_to_pos_embed_cuda

class VisionTransformer(timm.models.vision_transformer.VisionTransformer):
    """ Vision Transformer with support for global average pooling
    """
    def __init__(self,  **kwargs):
        super(VisionTransformer, self).__init__(**kwargs)
        self.patch_size = kwargs['patch_size']
        self.in_chans = kwargs['in_chans']
        self.embed_dim = kwargs['embed_dim']

    def forward_features(self, x,total_count):
        B = x.shape[0]
        x = self.patch_embed(x)

        total_count = torch.log10(total_count)
        count_embed = convert_count_to_pos_embed_cuda(total_count, self.embed_dim)
        count_embed = count_embed.unsqueeze(1)# (N, 1, D)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        #cls tokens is simply a learnable parameter and 0 positional embedding is added
        x = x + self.pos_embed[:, 1:, :]

        x = torch.cat((cls_tokens,count_embed,  x), dim=1)
        
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)

        return x

def vit_base_patch16(**kwargs):
    model = VisionTransformer(in_chans=3,
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def vit_large_patch16(**kwargs):
    model = VisionTransformer(in_chans=3,
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def vit_huge_patch14(**kwargs):
    model = VisionTransformer(in_chans=3,
        patch_size=14, embed_dim=1280, depth=32, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model
