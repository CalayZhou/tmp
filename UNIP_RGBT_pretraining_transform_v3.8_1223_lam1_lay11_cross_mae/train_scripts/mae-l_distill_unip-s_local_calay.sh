CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch  --nproc_per_node 1 --master_port 29590 main_pretrain_quick.py \
    --batch_size 2 --accum_iter 8 --epochs 100 --warmup_epochs 5 --blr 1.5e-4 --weight_decay 0.05 --pin_mem \
    --model unip_vit_small_patch16 --last_heads 16  \
    --infpre_path /home/calay/dataset/InfMix/ \
    --per_cls_num 200 --rgb_gray \
    --teacher_path /home/calay/calay/FM_VIT-B/mae/mae_pretrain_vit_large.pth \
    --teacher_model vit_large \
    --intermediate 18 \
    --output_dir=../mae_large_layer18_distill_unip_small_100ep_infmix \
    --log_dir=../mae_large_layer18_distill_unip_small_100ep_infmix