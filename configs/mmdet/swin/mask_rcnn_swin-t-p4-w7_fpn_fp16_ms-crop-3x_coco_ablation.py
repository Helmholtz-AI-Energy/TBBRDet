# We inherit from the fp16 training and overwrite anything to do with input channels
_base_ = './mask_rcnn_swin-t-p4-w7_fpn_fp16_ms-crop-3x_coco.py'

# Truncate these to handle 4 input channels
img_norm_cfg = dict(
    mean=[130., 135., 135., 118.],  # 118.],
    std=[44., 40., 40., 30.],  # 21.],
    to_rgb=False
)

model = dict(
    backbone=dict(
        # This is to set to our hyperspectral input
        in_channels=4,
    ),
)

# These have to be copied over in full
# augmentation strategy originates from DETR / Sparse RCNN
train_pipeline = [
    dict(type='LoadNumpyImageFromFile', drop_height=True),
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
    dict(type='LoadNumpyImageFromFile', drop_height=True),
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
