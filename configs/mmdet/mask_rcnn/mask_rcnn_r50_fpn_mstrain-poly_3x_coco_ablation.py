# This config sets up the ablation training on the default config
_base_ = [
    './mask_rcnn_r50_fpn_mstrain-poly_3x_coco.py',
]

# Truncate these to handle 4 input channels
img_norm_cfg = dict(
    mean=[130., 135., 135., 118.],  # 118.],
    std=[44., 40., 40., 30.],  # 21.],
    to_rgb=False
)

model = dict(
    backbone=dict(
        # This is to set to our RGBT input
        in_channels=4,
    ),
)

# Need to copy these in full to modify the pipeline
# In mstrain 3x config, img_scale=[(1333, 640), (1333, 800)],
# multiscale_mode='range'
train_pipeline = [
    dict(type='LoadNumpyImageFromFile', drop_height=True),
    dict(
        type='LoadAnnotations',
        with_bbox=True,
        with_mask=True,
        poly2mask=False),
    dict(
        type='Resize',
        img_scale=[(3370, int(3.35 * 640)), (3370, int(3.35 * 800))],
        multiscale_mode='range',
        keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks']),
]
test_pipeline = [
    dict(type='LoadNumpyImageFromFile', drop_height=True),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(3370, int(3.35 * 800)),
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

data = dict(
    train=dict(
        dataset=dict(pipeline=train_pipeline)),
    val=dict(pipeline=test_pipeline),
    test=dict(pipeline=test_pipeline),
)
