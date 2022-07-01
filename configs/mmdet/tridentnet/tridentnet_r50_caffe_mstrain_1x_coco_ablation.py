from common_vars import custom_imports

_base_ = './tridentnet_r50_caffe_1x_coco_ablation.py'

# Truncate these to handle 4 input channels
img_norm_cfg = dict(
    mean=[130., 135., 135., 118.],  # 118.],
    std=[44., 40., 40., 30.],  # 21.],
    to_rgb=False
)

train_pipeline = [
    dict(type='LoadNumpyImageFromFile', drop_height=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='Resize',
        img_scale=[(3370, int(3.35 * 640)), (3370, int(3.35 * 672)), (3370, int(3.35 * 704)), (3370, int(3.35 * 736)),
                   (3370, int(3.35 * 768)), (3370, int(3.35 * 800))],
        multiscale_mode='value',
        keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]

data = dict(train=dict(pipeline=train_pipeline))
