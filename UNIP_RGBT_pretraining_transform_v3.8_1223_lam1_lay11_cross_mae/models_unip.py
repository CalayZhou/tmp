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
from util.pos_embed import get_2d_sincos_pos_embed
import torch.nn.functional as F
import numpy as np
import cv2
import os
class UNIP(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, drop_path=0.1,
                 embed_dim=1024, depth=24, num_heads=16, last_heads=12, distill_layers=[7,11], #calay v3.7
                 mlp_ratio=4., norm_layer=nn.LayerNorm, loss_type='KL'):
        super().__init__()
        
        self.depth = depth
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.loss_type = loss_type
        self.last_heads = last_heads
        self.distill_layers = distill_layers

        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)  # fixed sin-cos embedding
        
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, drop_path=drop_path, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(depth-1)]+[Block(embed_dim, self.last_heads, mlp_ratio, drop_path=drop_path, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)])
        self.norm = norm_layer(embed_dim)

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
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
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
 
    def compute_cross_modal_mi_kl(self, x, temperature=0.1):

        """
        基于KL散度的互信息估计
        x_rgb: [B, N, C]
        x_ir: [B, N, C]
        返回: [B, N] 值域在0~1之间
        """
        x = x.detach()
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

    def forward_encoder(self, x, var_rgbt):
        # embed patches
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        attn_rgbt_q_rgb_k_rgb_list, attn_rgbt_q_rgb_k_ir_list = [], []
        cmss_map_list = []
        # apply Transformer blocks
        for i, blk in enumerate(self.blocks):

            # if i == self.depth - 1:
            #     # x, qk = blk(x, return_attention=True)
            #     x, attn_softmax_qk, attn_rgbt_q_rgb_k_ir_softmax,attn_rgbt_q_ir_k_rgb_softmax = blk(x, return_attention=True)
            #
            #     # nmi_qk = self.calculate_normalized_mutual_information(qk) #nmi calay
            #     return attn_softmax_qk,attn_rgbt_q_rgb_k_ir_softmax,attn_rgbt_q_ir_k_rgb_softmax

            if i in self.distill_layers: # v3.7 calay
                # x, qk = blk(x, return_attention=True)
                x, attn_softmax_qk, attn_rgbt_q_rgb_k_rgb,attn_rgbt_q_rgb_k_ir = blk(x, return_attention=True)
                attn_rgbt_q_rgb_k_rgb_list.append(attn_rgbt_q_rgb_k_rgb)
                attn_rgbt_q_rgb_k_ir_list.append(attn_rgbt_q_rgb_k_ir)
                # nmi_qk = self.calculate_normalized_mutual_information(qk) #nmi calay

                # mi cmss  ####v3.8 s3####
                x_mi = self.compute_cross_modal_mi_kl(x[:, 1:])
                B, N = x_mi.shape
                var_rgbt = var_rgbt.view(B, N)
                x_mi = x_mi / var_rgbt
                cmss_map_list.append(x_mi)
                # cmss_map = x_mi / torch.max(x_mi)
                # # cmss_map = cmss_map.view(B // 2, N-1)  # B//2 N

                if i == self.depth - 1:
                    max_vals = [x.max() for x in cmss_map_list]
                    global_max = torch.stack(max_vals).max()
                    cmss_map_list = [x /global_max  for x in cmss_map_list]
                    # self.output_mi_vis(cmss_map_list)
                    return attn_softmax_qk, attn_rgbt_q_rgb_k_rgb_list, attn_rgbt_q_rgb_k_ir_list,cmss_map_list  ####v3.8 s6####
            else:
                x = blk(x)

    def output_mi_vis(self, cmss_map_list):
        # ############################
        # B, N= x_mi.shape
        # var_rgbt = var_rgbt.view(B,N-1)
        for i in range(len(cmss_map_list)):
            cmss_map = cmss_map_list[i]
            save_path = "/home/calay/calay/UNIP/UNIP_RGBT_pretraining_transform_v3.8/VIS_RESULT/"
            num = (len(os.listdir(save_path))) // 6 #save_num  # rgb ir attn1 attn2 cmss mask

            # x_mi= x_mi[:, 1:]/var_rgbt
            # x_mi = x_mi/torch.max(x_mi)
            msdi_value_patch = cmss_map[0, :]
            msdi_value_patch = msdi_value_patch.view(14, 14)
            msdi_value_patch_np = msdi_value_patch.detach().cpu().numpy()
            msdi_value_patch_np = (msdi_value_patch_np * 255).astype(np.uint8)
            msdi_value_patch_np_resize = cv2.resize(msdi_value_patch_np, (256, 256), interpolation=cv2.INTER_NEAREST)

            msdi_value_patch_np_resize_color = cv2.applyColorMap(msdi_value_patch_np_resize, cv2.COLORMAP_HOT)
            # msdi_value_patch_np_resize_color = msdi_value_patch_np_resize
            cv2.imwrite(save_path + str(num) + 'layer'+str(i)+'_mi.png', msdi_value_patch_np_resize_color)

    def forward_kd_loss(self, pred, teacher_out):
        if self.loss_type == "CE":
            loss = nn.CrossEntropyLoss(reduction='none')(pred, torch.softmax(teacher_out, dim=-1)).sum(-1)   # [B, H]
        elif self.loss_type == 'KL':
            loss = nn.KLDivLoss(reduction="none", log_target=True)(torch.log_softmax(pred, dim=-1), torch.log_softmax(teacher_out, dim=-1)).sum(-1) # [B, H]
        else:
            raise ValueError(f"Loss type {self.loss_type} is not supported!")
        return loss.mean()

    def forward_kd_cmss_loss(self, pred, teacher_out, cmss_map):
        #pred, teacher_out [B, H, query(L), key(L)]

        cmss_cls_token = torch.ones(cmss_map.size(0), 1, device=cmss_map.device)  # (B,1)

        cmss_map_mask= torch.cat([cmss_cls_token, cmss_map], dim=1)  # (B, L+1)
        cmss_map_mask = 1 - cmss_map_mask[:, None, :] #+ 1.0  #!!12.6
        cmss_map_mask =  torch.exp(cmss_map_mask)#12.9 v3.6.3

        #loss_type == 'KL':
        loss = nn.KLDivLoss(reduction="none", log_target=True)(torch.log_softmax(pred, dim=-1),
                                                                   torch.log_softmax(teacher_out, dim=-1)).sum( -1)  # [B, H]
        loss_cmss_weight = loss*cmss_map_mask
        return loss_cmss_weight.mean()

    def forward(self, imgs_concat, teacher_qk=None, var_rgbt=None):

        student_qk,attn_rgbt_q_rgb_k_rgb_list,attn_rgbt_q_rgb_k_ir_list,cmss_map_list = self.forward_encoder(imgs_concat,var_rgbt)
        # if teacher_qk==None or sutdent_ir_qk==None:
        #     return student_qk#10.17
        B, _, _, _ = student_qk.shape
        student_qk = student_qk[:B//2,:,:,:]

        assert student_qk.shape == teacher_qk.shape, "The outputs of student and teachers unmatched!"
        #assert attn_rgbt_q_rgb_k_ir_softmax.shape == attn_rgbt_q_ir_k_rgb_softmax.shape, "The outputs of student and teachers unmatched!"
        qk_loss_teacher = self.forward_kd_loss(student_qk.float(), teacher_qk.float())

        # ORI KL LOSS
        # qk_loss_rgbt0 = self.forward_kd_loss(attn_rgbt_q_rgb_k_ir_softmax[0].float(), attn_rgbt_q_ir_k_rgb_softmax[0].float())#10.17
        # qk_loss_rgbt1 = self.forward_kd_loss(attn_rgbt_q_rgb_k_ir_softmax[1].float(), attn_rgbt_q_ir_k_rgb_softmax[1].float())
        # qk_loss_rgbt = (qk_loss_rgbt0 + qk_loss_rgbt1) / 2

        # CMSS KL LOSS
        qk_losses = []
        qk_losses_weight = [1.0, 1.0, 1.0, 1.0]

        for i, (attn_rgbt_q_rgb_k_rgb_i, attn_rgbt_q_rgb_k_ir_i) in enumerate(
                zip(attn_rgbt_q_rgb_k_rgb_list, attn_rgbt_q_rgb_k_ir_list)):
            qk_loss = self.forward_kd_cmss_loss(
                attn_rgbt_q_rgb_k_rgb_i.float(),
                attn_rgbt_q_rgb_k_ir_i.float(),
                cmss_map_list[i]
            )
            weighted_qk_loss = qk_loss.mean() * qk_losses_weight[i]
            qk_losses.append(weighted_qk_loss)
        print("qk_loss:",qk_losses)
        qk_loss_rgbt = torch.stack(qk_losses).mean()
        # qk_loss_rgbt = self.forward_kd_cmss_loss(attn_rgbt_q_rgb_k_ir_softmax[0].float(), attn_rgbt_q_ir_k_rgb_softmax[0].float(), cmss_map)#10.17
        #qk_loss_rgbt1 = self.forward_kd_cmss_loss(attn_rgbt_q_rgb_k_ir_softmax[1].float(), attn_rgbt_q_ir_k_rgb_softmax[1].float(), cmss_map)
        #qk_loss_rgbt = (qk_loss_rgbt0 + qk_loss_rgbt1) / 2

        # ## BI KL LOSS
        
        #qk_loss_rgbt0 = self.forward_kd_loss(attn_rgbt_q_rgb_k_ir_softmax[0].float(), attn_rgbt_q_ir_k_rgb_softmax[0].float())#10.17
        #qk_loss_rgbt1 = self.forward_kd_loss(attn_rgbt_q_ir_k_rgb_softmax[0].float(), attn_rgbt_q_rgb_k_ir_softmax[0].float())#10.17
        #qk_loss_rgbt2 = self.forward_kd_loss(attn_rgbt_q_rgb_k_ir_softmax[1].float(), attn_rgbt_q_ir_k_rgb_softmax[1].float())
        #qk_loss_rgbt3 = self.forward_kd_loss(attn_rgbt_q_ir_k_rgb_softmax[1].float(), attn_rgbt_q_rgb_k_ir_softmax[1].float())
        #qk_loss_rgbt = (qk_loss_rgbt0+qk_loss_rgbt1)/2#+qk_loss_rgbt2+qk_loss_rgbt3)/4
        

        return qk_loss_teacher, qk_loss_rgbt

    # calay 1012
    def calculate_normalized_mutual_information(self, attention_weights):
        # attention weights: (batch_size, num_heads, num_patch, num_patch) / (B, H, L, L)
        query_prob_sum = torch.sum(attention_weights, dim=-1, keepdims=True)  # (B, H, L, 1)
        query_prob = query_prob_sum / torch.sum(query_prob_sum, dim=-2, keepdims=True)  # (B, H, L, 1)

        key_prob_sum = torch.sum(attention_weights, dim=-2, keepdims=True)  # (B, H, 1, L)
        key_prob = key_prob_sum / torch.sum(key_prob_sum, dim=-1, keepdims=True)  # (B, H, 1, L)

        H_query = - torch.sum(query_prob * torch.log2(query_prob), dim=(-1, -2))  # (B, H)
        H_key = - torch.sum(key_prob * torch.log2(key_prob), dim=(-1, -2))  # (B, H)

        qk_joint_prob = attention_weights * query_prob  # (B, H, L, L)
        I = torch.sum(qk_joint_prob * torch.log2(qk_joint_prob / (query_prob * key_prob)), axis=(-1, -2))  # (B, H)
        normalized_I = I / torch.sqrt(H_query * H_key)  # (B, H)
        normalized_I = torch.mean(normalized_I, dim=-1)  # (B)
        return normalized_I


def unip_vit_tiny_patch16(**kwargs):
    model = UNIP(
        patch_size=16, embed_dim=192, depth=12, num_heads=3, drop_path=0.1,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def unip_vit_small_patch16(**kwargs):
    model = UNIP(
        patch_size=16, embed_dim=384, depth=12, num_heads=6,drop_path=0.1,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def unip_vit_base_patch16(**kwargs):
    model = UNIP(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, drop_path=0.1,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model



