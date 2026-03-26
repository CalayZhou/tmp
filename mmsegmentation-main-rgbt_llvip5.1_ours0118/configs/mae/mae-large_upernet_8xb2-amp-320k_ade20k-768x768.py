_base_ = [
    '../_base_/models/upernet_mae_classbalance.py', '../_base_/datasets/ade20k_768x768.py', 
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_320k.py'
]
crop_size_value = 768
crop_size = (crop_size_value, crop_size_value)#(512, 512)
data_preprocessor = dict(size=crop_size)
model = dict(
    data_preprocessor=data_preprocessor,
    # pretrained='./pretrain/M-SpecGene_VIT-B_seg_transform.pth',
    pretrained='./pretrain/VIT-L/deit_3_large_224_1k_transform_fixed.pth',
    # ibot_vitl_checkpoint_teacher_transform.pth',
    #ibot_vitl22k_checkpoint_student_transform.pth',#mae_pretrain_vit_large_transform.pth',
    #pretrained='./pretrain/transform_model/VIT-L/deit_3_large_224_1k_transform_fixed.pth',
    # pretrained='./pretrain/transform_model/VIT-L/ibot_vitl_checkpoint_teacher_transform.pth',
    # pretrained='./pretrain/transform_model/VIT-L/ibot_vitl22k_checkpoint_student_transform.pth',
    # pretrained='./pretrain/transform_model/VIT-L/deit_3_large_224_21k_transform_fixed.pth',

    backbone=dict(
        type='MAE',
        img_size=(crop_size_value,crop_size_value),#(768,768),
        patch_size=16,
        embed_dims=1024,#768,
        num_layers=24,#12,
        num_heads=16,#12,
        mlp_ratio=4,
        init_values=1.0,
        drop_path_rate=0.2,#0.1,
        out_indices=[7, 11, 15, 23]),#[5, 11, 17, 23]),#[3, 5, 7, 11]),
    neck=dict(embed_dim=1024*2, rescales=[4, 2, 1, 0.5]),
    decode_head=dict(
        in_channels=[1024*2, 1024*2, 1024*2, 1024*2], num_classes=13, channels=1024),#13 15
    auxiliary_head=dict(in_channels=1024*2, num_classes=13),#13 15
    test_cfg=dict(mode='slide', crop_size=(crop_size_value, crop_size_value), stride=(341, 341)))

optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW', lr=1e-4, betas=(0.9, 0.999), weight_decay=0.05),
    paramwise_cfg=dict(num_layers=12, layer_decay_rate=0.65),
    constructor='LayerDecayOptimizerConstructor')

param_scheduler = [
    dict(
        type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=1500),
    dict(
        type='PolyLR',
        eta_min=0.0,
        power=1.0,
        begin=1500,
        end=320000,#160000
        by_epoch=False,
    )
]

# mixed precision
fp16 = dict(loss_scale='dynamic')

# By default, models are trained on 8 GPUs with 2 images per GPU
train_dataloader = dict(batch_size=1)
val_dataloader = dict(batch_size=1)
test_dataloader = val_dataloader
