CUDA_VISIBLE_DEVICES=2,3 python -m torch.distributed.launch  \
    --nproc_per_node 2 --master_port 29593 main_pretrain.py \
    --batch_size 256 --accum_iter 8 --epochs 100 --warmup_epochs 5 --blr 1.5e-4 \
    --weight_decay 0.05 --pin_mem \
    --model unip_vit_small_patch16 --last_heads 16 \
    --infpre_path /home/dataset/RGBT_ALL/ \
    --in1k_path /home/dataset/ImageNet-1K/train/ \
    --coco_path /home/dataset/COCO/ \
    --use_coco --use_in1k --per_cls_num 200 \
    --teacher_path /home/unip/mae_pretrain_vit_large.pth \
    --resume ./mae_large_layer18_distill_unip_small_100ep_infmix/checkpoint-40.pth \
    --start_epoch 40 \
    --teacher_model vit_large \
    --intermediate 18 \
    --output_dir=./mae_large_layer18_distill_unip_small_100ep_infmix \
    --log_dir=./mae_large_layer18_distill_unip_small_100ep_infmix

