import math
from functools import partial
from typing import Literal

import torch
import torch.nn as nn
from mmcv.cnn.bricks.drop import build_dropout
from mmengine.model import BaseModule, ModuleList
from mmengine.model.weight_init import trunc_normal_
from mmengine.runner.checkpoint import _load_checkpoint

from mmseg.registry import MODELS


class RMSNorm(nn.Module):

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x * rms * self.weight


class LayerScale(nn.Module):

    def __init__(self, dim, init_values=1e-5):
        super().__init__()
        self.gamma = nn.Parameter(torch.empty(dim))
        self.init_values = init_values
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.constant_(self.gamma, self.init_values)

    def forward(self, x):
        return x * self.gamma


class DINOv3PatchEmbed(nn.Module):

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        if isinstance(img_size, int):
            img_size = (img_size, img_size)
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.Identity()

    def forward(self, x):
        x = self.proj(x)
        h, w = x.shape[2], x.shape[3]
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, (h, w)


class RopePositionEmbedding(nn.Module):

    def __init__(self,
                 embed_dim,
                 *,
                 num_heads,
                 base=100.0,
                 min_period=None,
                 max_period=None,
                 normalize_coords: Literal['min', 'max', 'separate'] =
                 'separate',
                 dtype=torch.float32):
        super().__init__()
        assert embed_dim % (4 * num_heads) == 0
        both_periods = min_period is not None and max_period is not None
        if (base is None and not both_periods) or (base is not None
                                                   and both_periods):
            raise ValueError('Either base or min_period+max_period is needed.')

        self.base = base
        self.min_period = min_period
        self.max_period = max_period
        self.normalize_coords = normalize_coords
        d_head = embed_dim // num_heads
        self.d_head = d_head
        self.dtype = dtype
        self.register_buffer('periods', torch.empty(d_head // 4, dtype=dtype))
        self._init_weights()

    def _init_weights(self):
        if self.base is not None:
            periods = self.base**(
                2 * torch.arange(
                    self.d_head // 4, device=self.periods.device, dtype=self.dtype)
                / (self.d_head // 2))
        else:
            base = self.max_period / self.min_period
            exponents = torch.linspace(
                0, 1, self.d_head // 4, device=self.periods.device, dtype=self.dtype)
            periods = (base**exponents) / base * self.max_period
        self.periods.data = periods

    def forward(self, *, h, w):
        dd = dict(device=self.periods.device, dtype=self.dtype)
        if self.normalize_coords == 'max':
            max_hw = max(h, w)
            coords_h = torch.arange(0.5, h, **dd) / max_hw
            coords_w = torch.arange(0.5, w, **dd) / max_hw
        elif self.normalize_coords == 'min':
            min_hw = min(h, w)
            coords_h = torch.arange(0.5, h, **dd) / min_hw
            coords_w = torch.arange(0.5, w, **dd) / min_hw
        else:
            coords_h = torch.arange(0.5, h, **dd) / h
            coords_w = torch.arange(0.5, w, **dd) / w

        coords = torch.stack(
            torch.meshgrid(coords_h, coords_w, indexing='ij'),
            dim=-1).flatten(0, 1)
        coords = 2.0 * coords - 1.0

        angles = 2 * math.pi * coords[:, :, None] / self.periods[None, None, :]
        angles = angles.flatten(1, 2).tile(2)
        sin = torch.sin(angles)
        cos = torch.cos(angles)
        return sin, cos


class SelfAttention(nn.Module):

    def __init__(self,
                 dim,
                 num_heads=8,
                 qkv_bias=False,
                 proj_bias=True,
                 attn_drop=0.,
                 proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    @staticmethod
    def _rope_rotate_half(x):
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat([-x2, x1], dim=-1)

    def _apply_rope(self, q, k, rope):
        sin, cos = rope
        q_dtype, k_dtype = q.dtype, k.dtype
        q, k = q.to(sin.dtype), k.to(sin.dtype)
        n = q.shape[-2]
        prefix = n - sin.shape[-2]
        q_prefix, k_prefix = q[:, :, :prefix, :], k[:, :, :prefix, :]
        q_rope = (q[:, :, prefix:, :] * cos) + (self._rope_rotate_half(
            q[:, :, prefix:, :]) * sin)
        k_rope = (k[:, :, prefix:, :] * cos) + (self._rope_rotate_half(
            k[:, :, prefix:, :]) * sin)
        q = torch.cat((q_prefix, q_rope), dim=-2).to(q_dtype)
        k = torch.cat((k_prefix, k_rope), dim=-2).to(k_dtype)
        return q, k

    def forward(self, x, rope=None):
        b, n, c = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, c // self.num_heads)
        q, k, v = torch.unbind(qkv, 2)
        q, k, v = [t.transpose(1, 2) for t in (q, k, v)]
        if rope is not None:
            q, k = self._apply_rope(q, k, rope)
        x = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        x = x.transpose(1, 2).reshape(b, n, c)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Mlp(nn.Module):

    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 bias=True,
                 drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class SelfAttentionBlock(nn.Module):

    def __init__(self,
                 dim,
                 num_heads,
                 ffn_ratio=4.0,
                 qkv_bias=True,
                 proj_bias=True,
                 ffn_bias=True,
                 drop=0.0,
                 attn_drop=0.0,
                 drop_path=0.0,
                 init_values=None,
                 norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = SelfAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            attn_drop=attn_drop,
            proj_drop=drop)
        self.ls1 = LayerScale(dim, init_values=init_values
                              ) if init_values else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * ffn_ratio),
            out_features=dim,
            bias=ffn_bias,
            drop=drop)
        self.ls2 = LayerScale(dim, init_values=init_values
                              ) if init_values else nn.Identity()
        self.drop_path = nn.Identity() if drop_path <= 0 else build_dropout(
            dict(type='DropPath', drop_prob=drop_path))

    def forward(self, x, rope=None):
        x = x + self.drop_path(self.ls1(self.attn(self.norm1(x), rope=rope)))
        x = x + self.drop_path(self.ls2(self.mlp(self.norm2(x))))
        return x


NORM_LAYER_DICT = {
    'layernorm': partial(nn.LayerNorm, eps=1e-6),
    'layernormbf16': partial(nn.LayerNorm, eps=1e-5),
    'rmsnorm': RMSNorm,
}

ROPE_DTYPE_DICT = {
    'fp32': torch.float32,
    'fp16': torch.float16,
    'bf16': torch.bfloat16,
}


@MODELS.register_module()
class DINOv3ViT(BaseModule):
    """DINOv3 Vision Transformer backbone adapted for MMSeg/UPerNet."""

    def __init__(self,
                 img_size=224,
                 patch_size=16,
                 in_channels=3,
                 embed_dims=768,
                 num_layers=12,
                 num_heads=12,
                 mlp_ratio=4.0,
                 out_indices=(3, 5, 7, 11),
                 qkv_bias=True,
                 proj_bias=True,
                 ffn_bias=True,
                 drop_rate=0.0,
                 attn_drop_rate=0.0,
                 drop_path_rate=0.0,
                 init_values=None,
                 n_storage_tokens=0,
                 norm_layer='layernorm',
                 pos_embed_rope_base=100.0,
                 pos_embed_rope_min_period=None,
                 pos_embed_rope_max_period=None,
                 pos_embed_rope_normalize_coords='separate',
                 pos_embed_rope_dtype='bf16',
                 pretrained=None,
                 init_cfg=None):
        if pretrained is not None:
            init_cfg = dict(type='Pretrained', checkpoint=pretrained)
        super().__init__(init_cfg=init_cfg)

        if isinstance(out_indices, int):
            out_indices = [out_indices]
        self.out_indices = tuple(out_indices)
        self.embed_dims = embed_dims
        self.num_layers = num_layers
        self.n_storage_tokens = n_storage_tokens

        norm_layer_cls = NORM_LAYER_DICT[norm_layer]
        dpr = torch.linspace(0, drop_path_rate, num_layers).tolist()

        self.patch_embed = DINOv3PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_channels,
            embed_dim=embed_dims)
        self.cls_token = nn.Parameter(torch.empty(1, 1, embed_dims))
        if n_storage_tokens > 0:
            self.storage_tokens = nn.Parameter(
                torch.empty(1, n_storage_tokens, embed_dims))
        self.mask_token = nn.Parameter(torch.empty(1, embed_dims))
        self.rope_embed = RopePositionEmbedding(
            embed_dim=embed_dims,
            num_heads=num_heads,
            base=pos_embed_rope_base,
            min_period=pos_embed_rope_min_period,
            max_period=pos_embed_rope_max_period,
            normalize_coords=pos_embed_rope_normalize_coords,
            dtype=ROPE_DTYPE_DICT[pos_embed_rope_dtype])
        self.blocks = ModuleList([
            SelfAttentionBlock(
                dim=embed_dims,
                num_heads=num_heads,
                ffn_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                proj_bias=proj_bias,
                ffn_bias=ffn_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                init_values=init_values,
                norm_layer=norm_layer_cls) for i in range(num_layers)
        ])
        self.norm = norm_layer_cls(embed_dims)
        self.head = nn.Identity()

    def init_weights(self):
        trunc_normal_(self.cls_token, std=.02)
        nn.init.zeros_(self.mask_token)
        if self.n_storage_tokens > 0:
            trunc_normal_(self.storage_tokens, std=.02)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
            elif isinstance(m, RMSNorm):
                nn.init.constant_(m.weight, 1.0)

        if isinstance(self.init_cfg, dict) and self.init_cfg.get(
                'type') == 'Pretrained':
            checkpoint = _load_checkpoint(
                self.init_cfg['checkpoint'], logger=None, map_location='cpu')
            if isinstance(checkpoint, dict):
                for key in ('state_dict', 'model', 'teacher', 'student'):
                    if key in checkpoint and isinstance(checkpoint[key], dict):
                        checkpoint = checkpoint[key]
                        break
            self.load_state_dict(checkpoint, strict=False)

    def _prepare_tokens_with_masks(self, x):
        x, hw_shape = self.patch_embed(x)
        b = x.shape[0]
        cls_token = self.cls_token + 0 * self.mask_token
        if self.n_storage_tokens > 0:
            storage_tokens = self.storage_tokens
        else:
            storage_tokens = torch.empty(
                1, 0, cls_token.shape[-1], device=cls_token.device, dtype=cls_token.dtype)
        x = torch.cat([
            cls_token.expand(b, -1, -1),
            storage_tokens.expand(b, -1, -1), x
        ],
                      dim=1)
        return x, hw_shape

    def forward(self, x):
        x, (h, w) = self._prepare_tokens_with_masks(x)
        outs = []
        for idx, blk in enumerate(self.blocks):
            rope = self.rope_embed(h=h, w=w) if self.rope_embed is not None else None
            x = blk(x, rope=rope)
            if idx in self.out_indices:
                x_norm = self.norm(x)
                patch_tokens = x_norm[:, self.n_storage_tokens + 1:, :]
                patch_tokens = patch_tokens.reshape(-1, h, w, self.embed_dims).permute(
                    0, 3, 1, 2).contiguous()
                outs.append(patch_tokens)
        return tuple(outs)
