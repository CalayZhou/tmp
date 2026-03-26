# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from functools import partial

import torch
import torch.nn as nn

from vit import PatchEmbed, Block
import torch.nn.functional as F
from util.pos_embed import get_2d_sincos_pos_embed

# import os
# from PIL import Image
# import cv2
# import matplotlib.pyplot as plt
# import seaborn as sns
# import numpy as np
# save_num = 6

class MaskedAutoencoderViT(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3,
                 embed_dim=1024, depth=24, num_heads=16, intermediate=18,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, 
                 temperature=1.0):
        super().__init__()
        self.dim = embed_dim
        self.temperature = temperature

        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)
        self.intermediate=intermediate
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        self.initialize_weights()


    def initialize_weights(self):
        # initialization
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def CMSS_Similarity(self, tensor_1, tensor_2):
        normalized_tensor_1 = tensor_1 / tensor_1.norm(dim=-1, keepdim=True)
        normalized_tensor_2 = tensor_2 / tensor_2.norm(dim=-1, keepdim=True)
        r_num = torch.sum(normalized_tensor_1 * normalized_tensor_2, dim=1)
        r_num = (r_num + 1.0) * 0.5
        var_tensor_1 = torch.var(tensor_1, dim=1)
        var_tensor_2 = torch.var(tensor_2, dim=1)
        r_num = torch.sqrt(r_num)
        # r_num = torch.ones_like(r_num)
        measure = r_num / (var_tensor_1 * var_tensor_2)
        var_rgbt = var_tensor_1 * var_tensor_2

        measure = measure / torch.max(measure)  # *10
        # measure = (measure - torch.min(measure)) / (torch.max(measure)-torch.min(measure))
        return var_rgbt


    def cmss_weight(self, x):
        # ############################
        B, N, C = x.shape
        x_rgb, x_ir = x[:B // 2, :, :], x[B // 2:, :, :]
        x_rgb_flat = x_rgb.clone().detach().contiguous().view(B // 2 * N, C)
        x_ir_flat = x_ir.clone().detach().contiguous().view(B // 2 * N, C)
        var_rgbt = self.CMSS_Similarity(x_rgb_flat, x_ir_flat)
        cmss_weight_value = var_rgbt.view(B // 2, N) #B//2 N
        return cmss_weight_value

    def compute_cross_modal_mi_kl(self, x, temperature=0.1):

        """
        基于KL散度的互信息估计
        x_rgb: [B, N, C]
        x_ir: [B, N, C]
        返回: [B, N] 值域在0~1之间
        """

        x_rgb, x_ir = x[:x.shape[0] // 2, :, :], x[x.shape[0] // 2:, :, :]

        B, N, C = x_rgb.shape

        # 计算相似度矩阵（逐patch计算）
        # 归一化特征
        x_rgb_norm = F.normalize(x_rgb, dim=-1)  # [B, N, C]
        x_ir_norm = F.normalize(x_ir, dim=-1)  # [B, N, C]

        # 计算每个patch与所有patch的相似度
        sim_matrix = torch.bmm(x_rgb_norm, x_ir_norm.transpose(1, 2))  # [B, N, N]
        sim_matrix = sim_matrix / temperature

        # 计算条件分布
        p_rgb_given_ir = F.softmax(sim_matrix, dim=-1)  # P(x_rgb|x_ir) [B, N, N]
        p_ir_given_rgb = F.softmax(sim_matrix.transpose(1, 2), dim=-1)  # P(x_ir|x_rgb) [B, N, N]

        # 边缘分布（均匀分布）
        p_rgb = torch.ones(B, N, N, device=x_rgb.device) / N  # [B, N, N]
        p_ir = torch.ones(B, N, N, device=x_rgb.device) / N  # [B, N, N]

        # 计算KL散度作为互信息估计
        # MI(x_rgb, x_ir) = KL(P(x_rgb|x_ir) || P(x_rgb))
        mi_rgb_ir = F.kl_div(
            torch.log(p_rgb_given_ir + 1e-8),
            p_rgb,
            reduction='none'
        ).sum(dim=-1)  # [B, N]

        mi_ir_rgb = F.kl_div(
            torch.log(p_ir_given_rgb + 1e-8),
            p_ir,
            reduction='none'
        ).sum(dim=-1)  # [B, N]

        # 对称互信息
        mi = (mi_rgb_ir + mi_ir_rgb) / 2

        # 归一化到0~1
        # 互信息总是非负的，我们可以通过sigmoid或min-max归一化
        mi_normalized = torch.sigmoid(mi)

        return mi_normalized



    def output_mi_vis(self, cmss_map):
        # ############################
        # B, N= x_mi.shape
        # var_rgbt = var_rgbt.view(B,N-1)

        save_path = "/home/calay/calay/UNIP/UNIP_RGBT_pretraining_transform_v3.8/VIS_RESULT/"
        num = (len(os.listdir(save_path))) // save_num  # rgb ir attn1 attn2 cmss mask

        # x_mi= x_mi[:, 1:]/var_rgbt
        # x_mi = x_mi/torch.max(x_mi)
        msdi_value_patch = cmss_map[0, :]
        msdi_value_patch = msdi_value_patch.view(14, 14)
        msdi_value_patch_np = msdi_value_patch.cpu().numpy()
        msdi_value_patch_np = (msdi_value_patch_np * 255).astype(np.uint8)
        msdi_value_patch_np_resize = cv2.resize(msdi_value_patch_np, (256, 256), interpolation=cv2.INTER_NEAREST)

        # msdi_value_patch_np_resize_color = cv2.applyColorMap(msdi_value_patch_np_resize, cv2.COLORMAP_HOT)
        msdi_value_patch_np_resize_color = msdi_value_patch_np_resize
        cv2.imwrite(save_path + str(num) + '_mi.png', msdi_value_patch_np_resize_color)



    def forward_encoder(self, x_concat):

        ##################
        # B, N, H, W = x_concat.shape
        # x_rgb, x_ir = x_concat[:B // 2, :, :, :], x_concat[B // 2:, :, :, :]
        # x_rgb = x_rgb * 0.1871+0.3801
        # x_ir = x_ir * 0.1871+0.3801
        # x_rgb_np = x_rgb[0,:,:,:].cpu().numpy().transpose((1,2,0))
        # x_ir_np = x_ir[0,:,:,:].cpu().numpy().transpose((1,2,0))
        # x_rgb_np = (x_rgb_np * 255).astype(np.uint8)
        # x_ir_np = (x_ir_np * 255).astype(np.uint8)
        # save_path = "/home/calay/calay/UNIP/UNIP_RGBT_pretraining_transform_v3.8/VIS_RESULT/"
        # num = (len(os.listdir(save_path)))  // save_num # rgb ir attn1 attn2 cmss mask
        # # x_rgb_resized = x_rgb.resize((256, 256))
        # # x_ir_resized = x_ir.resize((256, 256))
        # x_rgb_np_pil = Image.fromarray(x_rgb_np)
        # x_ir_np_pil = Image.fromarray(x_ir_np)
        # x_rgb_np_pil.save(save_path+str(num)+'_x_rgb.png')
        # x_ir_np_pil.save(save_path+str(num)+'_x_ir.png')
        ##################

        x_cmss = self.patch_embed(x_concat)
        # return cmss map
        var_rgbt = self.cmss_weight(x_cmss) #B//2 N
        

        B, C, H, W = x_concat.shape
        x = x_concat[:B//2,:,:,:]
        
        # embed patches
        x = self.patch_embed(x)

        # return cmss map
        #cmss_map = self.cmss_weight(x) #B//2 N

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
            
        # apply Transformer blocks
        for i, blk in enumerate(self.blocks):
            if i == self.intermediate - 1:
                # x, qk = blk(x, return_attention=True)
                x, qk,attn_rgbt_q_rgb_k_ir_softmax,attn_rgbt_q_ir_k_rgb_softmax = blk(x, return_attention=True)

                # # mi cmss
                # x_mi =  self.compute_cross_modal_mi_kl(x_cmss)
                # B, N = x_mi.shape
                # var_rgbt = var_rgbt.view(B, N)
                # x_mi = x_mi / var_rgbt
                # cmss_map = x_mi / torch.max(x_mi)
                # cmss_map = cmss_map.view(B // 2, N-1)  # B//2 N
                # self.output_mi_vis(cmss_map)

                return qk,var_rgbt #v3.3
            else:
                x = blk(x)
        
        return qk


    def forward(self, imgs):
        qk = self.forward_encoder(imgs)
        return qk
    
    
    
def vit_tiny(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=192, depth=12, num_heads=6,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def vit_small(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=384, depth=12, num_heads=6,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def vit_base(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def vit_large(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

