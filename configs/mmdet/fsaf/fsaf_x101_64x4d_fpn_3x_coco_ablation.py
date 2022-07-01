# Import our own loader pipeline
custom_imports = dict(imports=['numpy_loader'], allow_failed_imports=False)

_base_ = './fsaf_x101_64x4d_fpn_3x_coco.py'

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
    )
)

# This is copied from the coco_detection file, need to modify the normalisation
train_pipeline = [
    dict(type='LoadNumpyImageFromFile', drop_height=False),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(3370, (3.35 * 800)), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadNumpyImageFromFile', drop_height=False),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(3370, (3.35 * 800)),
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
    samples_per_gpu=1,
    train=dict(pipeline=train_pipeline),
    val=dict(pipeline=test_pipeline),
    test=dict(pipeline=test_pipeline),
)
