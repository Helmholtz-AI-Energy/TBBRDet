from common_vars import custom_imports, img_norm_cfg

_base_ = [
    '../_base_/models/mask_rcnn_r50_fpn.py',
    '../_base_/datasets/coco_instance.py',
    '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py'
]

model = dict(
    type='MaskRCNN',
    backbone=dict(
        _delete_=True,
        type='SwinTransformer',
        embed_dims=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.2,
        patch_norm=True,
        out_indices=(0, 1, 2, 3),
        with_cp=False,
        convert_weights=True,
        # This sets the weights to use or not use for pretraining
        # init_cfg=pretrained_init_cfg,
        # This is to set to our hyperspectral input
        in_channels=5,
    ),
    neck=dict(in_channels=[96, 192, 384, 768]),
    # We also need to change the num_classes in head to match the dataset's annotation
    roi_head=dict(
        bbox_head=dict(num_classes=1),
        mask_head=dict(num_classes=1)),
    # pretrained=None,
)


# model = dict(
#     roi_head=dict(
#         bbox_head=dict(num_classes=1),
#         mask_head=dict(num_classes=1)),
#     # pretrained=None,
#     backbone=dict(
#         in_channels=5,
#         # norm_eval=False
#     ),
# )

# augmentation strategy originates from DETR / Sparse RCNN
train_pipeline = [
    dict(type='LoadNumpyImageFromFile', drop_height=False),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(
        type='AutoAugment',
        policies=[
            [
                dict(
                    type='Resize',
                    img_scale=[(int(3.35 * 480), 3370), (int(3.35 * 512), 3370), (int(3.35 * 544), 3370), (int(3.35 * 576), 3370),
                               (int(3.35 * 608), 3370), (int(3.35 * 640), 3370), (int(3.35 * 672), 3370), (int(3.35 * 704), 3370),
                               (int(3.35 * 736), 3370), (int(3.35 * 768), 3370), (int(3.35 * 800), 3370)],
                    multiscale_mode='value',
                    keep_ratio=True)
            ],
            [
                dict(
                    type='Resize',
                    img_scale=[(int(3.35 * 400), 3370), (int(3.35 * 500), 3370), (int(3.35 * 600), 3370)],
                    multiscale_mode='value',
                    keep_ratio=True),
                dict(
                    type='RandomCrop',
                    crop_type='absolute_range',
                    crop_size=(int(3.35 * 384), int(3.35 * 600)),
                    allow_negative_crop=True),
                dict(
                    type='Resize',
                    img_scale=[(int(3.35 * 480), 3370), (int(3.35 * 512), 3370), (int(3.35 * 544), 3370),
                               (int(3.35 * 576), 3370), (int(3.35 * 608), 3370), (int(3.35 * 640), 3370),
                               (int(3.35 * 672), 3370), (int(3.35 * 704), 3370), (int(3.35 * 736), 3370),
                               (int(3.35 * 768), 3370), (int(3.35 * 800), 3370)],
                    multiscale_mode='value',
                    override=True,
                    keep_ratio=True
                )
            ]
        ]),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks']),
]

# This is from _base_/datasets/coco_instance.py
test_pipeline = [
    dict(type='LoadNumpyImageFromFile', drop_height=False),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(3370, 2680),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

# Set data location and pipeline
# Other settings from _base_/datasets/coco_instance.py
data = dict(
    # Seems to be really memory intensive
    samples_per_gpu=1,
    workers_per_gpu=32,
    train=dict(
        pipeline=train_pipeline,
    ),
    val=dict(
        pipeline=test_pipeline,
    ),
    test=dict(
        pipeline=test_pipeline,
    ),
)

optimizer = dict(
    _delete_=True,
    type='AdamW',
    lr=0.0001,
    betas=(0.9, 0.999),
    weight_decay=0.05,
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'relative_position_bias_table': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.)
        }))
lr_config = dict(warmup_iters=1000, step=[27, 33])
runner = dict(max_epochs=36)
