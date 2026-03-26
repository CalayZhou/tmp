import torch
import torch.nn as nn

from timm.layers import DropPath, to_2tuple
import torch.nn.functional as F


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5
        self.drop_rate = attn_drop
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, return_attention=False, temperature=1.0):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)  
        q, k, v = qkv[0], qkv[1], qkv[2] 

        if return_attention:

            q_rgb, q_ir = q[:B//2,:,:,:],q[B//2:,:,:,:] # B H L C
            k_rgb, k_ir = k[:B//2,:,:,:],k[B//2:,:,:,:] # C
            
            #attn_rgbt_q_rgb_k_rgb = (q_rgb @ k_rgb.transpose(-2, -1)) * self.scale
            #attn_rgbt_q_ir_k_ir = (q_ir @ k_ir.transpose(-2, -1)) * self.scale
            
            #attn_rgbt_q_rgb_k_ir = (q_rgb @ k_ir.transpose(-2, -1)) * self.scale
            #attn_rgbt_q_ir_k_rgb = (q_ir @ k_rgb.transpose(-2, -1)) * self.scale

            attn_rgbt_q_rgb_k_rgb = (q_rgb @ k_rgb.transpose(-2, -1)) * self.scale # B H L C * # B H C L
            attn_rgbt_q_rgb_k_ir = (q_rgb @ k_ir.transpose(-2, -1)) * self.scale


            # attn_rgbt_q_ir_k_rgb = (q_ir @ k_rgb.transpose(-2, -1)) * self.scale
            # attn_rgbt_q_ir_k_ir = (q_ir @ k_ir.transpose(-2, -1)) * self.scale


            # attn_list1, attn_list2 = [],[]
            #attn_list1.append(attn_rgbt_q_rgb_k_rgb)
            #attn_list1.append(attn_rgbt_q_rgb_k_ir)
            # attn_list1.append(attn_rgbt_q_rgb_k_rgb)
            # attn_list1.append(attn_rgbt_q_ir_k_rgb)
            #attn_list2.append(attn_rgbt_q_ir_k_ir)
            #attn_list2.append(attn_rgbt_q_ir_k_rgb)
            # attn_list2.append(attn_rgbt_q_rgb_k_ir)
            # attn_list2.append(attn_rgbt_q_ir_k_ir)

            #attn_rgbt_q_rgb_k_ir_softmax = attn_rgbt_q_rgb_k_ir.softmax(dim=-1)
            #attn_rgbt_q_ir_k_rgb_softmax = attn_rgbt_q_ir_k_rgb.softmax(dim=-1)
            
            '''
            attn = (q @ k.transpose(-2, -1)) * self.scale
            qk = (q @ k.transpose(-2, -1)) * self.scale / temperature
            attn = attn.softmax(dim=-1)
            #attn_softmax_qk = attn
            attn = self.attn_drop(attn)
            x = (attn @ v).transpose(1, 2).reshape(B, N, C)
            '''
            qk = (q @ k.transpose(-2, -1)) * self.scale / temperature
            with torch.backends.cuda.sdp_kernel(enable_math=False):
                x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.drop_rate)
                x = x.transpose(1, 2).reshape(B, N, C)
        else:
            # use flash attention for speeding up training
            with torch.backends.cuda.sdp_kernel(enable_math=False): 
                x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.drop_rate)
                x = x.transpose(1, 2).reshape(B, N, C)
 
        x = self.proj(x)
        x = self.proj_drop(x)
        
        if return_attention:
            return x, qk, attn_rgbt_q_rgb_k_rgb, attn_rgbt_q_rgb_k_ir#attn_list1, attn_list2 #attn_rgbt_q_rgb_k_ir,attn_rgbt_q_ir_k_rgb
        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)


    def forward(self, x, return_attention=False, temperature=1.0):
        if return_attention:
            tmp_x, attn_softmax_qk, attn_rgbt_q_rgb_k_ir_softmax,attn_rgbt_q_ir_k_rgb_softmax = self.attn(self.norm1(x), return_attention=True, temperature=temperature)
            x = x + self.drop_path(tmp_x)
        else:
            x = x + self.drop_path(self.attn(self.norm1(x)))

        x = x + self.drop_path(self.mlp(self.norm2(x)))
        
        if return_attention:
            return x, attn_softmax_qk, attn_rgbt_q_rgb_k_ir_softmax,attn_rgbt_q_ir_k_rgb_softmax
        return x
    

class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x
