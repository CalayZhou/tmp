import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.layers import DropPath, to_2tuple


class LayerScale(nn.Module):
    """DINOv3-style LayerScale."""

    def __init__(self, dim, init_values=1e-5):
        super().__init__()
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return self.gamma * x


class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
        bias=True,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    """Attention path aligned with DINOv3 SelfAttention implementation.

    Kept backward-compatible return_attention outputs for UNIP distillation code.
    """

    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        proj_bias=True,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5
        self.drop_rate = attn_drop

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, return_attention=False, temperature=1.0):
        B, N, C = x.shape

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        q, k, v = torch.unbind(qkv, 2)
        q, k, v = [t.transpose(1, 2) for t in [q, k, v]]

        if return_attention:
            q_rgb, q_ir = q[: B // 2], q[B // 2 :]
            k_rgb, k_ir = k[: B // 2], k[B // 2 :]
            attn_rgbt_q_rgb_k_rgb = (q_rgb @ k_rgb.transpose(-2, -1)) * self.scale
            attn_rgbt_q_rgb_k_ir = (q_rgb @ k_ir.transpose(-2, -1)) * self.scale
            qk = (q @ k.transpose(-2, -1)) * self.scale / temperature

        with torch.backends.cuda.sdp_kernel(enable_math=False):
            x = F.scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=self.drop_rate if self.training else 0.0,
            )
        x = x.transpose(1, 2).reshape(B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)

        if return_attention:
            return x, qk, attn_rgbt_q_rgb_k_rgb, attn_rgbt_q_rgb_k_ir
        return x


class Block(nn.Module):
    """DINOv3-style transformer block with optional LayerScale."""

    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        init_values=1e-5,
        proj_bias=True,
        ffn_bias=True,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            proj_bias=proj_bias,
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
            bias=ffn_bias,
        )
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()

    def forward(self, x, return_attention=False, temperature=1.0):
        if return_attention:
            tmp_x, attn_softmax_qk, attn_rgbt_q_rgb_k_ir_softmax, attn_rgbt_q_ir_k_rgb_softmax = self.attn(
                self.norm1(x),
                return_attention=True,
                temperature=temperature,
            )
            x = x + self.drop_path(self.ls1(tmp_x))
        else:
            x = x + self.drop_path(self.ls1(self.attn(self.norm1(x))))

        x = x + self.drop_path(self.ls2(self.mlp(self.norm2(x))))

        if return_attention:
            return x, attn_softmax_qk, attn_rgbt_q_rgb_k_ir_softmax, attn_rgbt_q_ir_k_rgb_softmax
        return x


class PatchEmbed(nn.Module):
    """DINOv3-style PatchEmbed: supports patch projection norm and relaxed image size check."""

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        norm_layer=None,
        flatten_embedding=True,
    ):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        patch_grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patch_grid_size
        self.num_patches = patch_grid_size[0] * patch_grid_size[1]

        self.flatten_embedding = flatten_embedding
        self.embed_dim = embed_dim
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        x = self.proj(x)
        H, W = x.size(2), x.size(3)
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        if not self.flatten_embedding:
            x = x.reshape(-1, H, W, self.embed_dim)
        return x
