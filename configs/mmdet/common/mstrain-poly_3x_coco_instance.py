from common_vars import custom_imports

_base_ = '../_base_/default_runtime.py'
# dataset settings
dataset_type = 'CocoDataset'
classes = ('Thermal bridge',)

data_root = '/hkfs/work/workspace/scratch/cd4062-wahn22/drone_dataset/Flug1_merged/'
train_img_prefix = 'train/images/'
train_ann_file = 'train/annotations/Flug1_onelabel_coco.json'
test_img_prefix = 'test/images/'
test_ann_file = 'test/annotations/Flug1_onelabel_coco.json'

img_norm_cfg = dict(
    mean=[130., 135., 135., 118., 118.],
    std=[44., 40., 40., 30., 21.],
    to_rgb=False
)

# In mstrain 3x config, img_scale=[(1333, 640), (1333, 800)],
# multiscale_mode='range'
train_pipeline = [
    dict(type='LoadNumpyImageFromFile', drop_height=False),
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
    dict(type='LoadNumpyImageFromFile', drop_height=False),
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

# Use RepeatDataset to speed up training
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=32,
    train=dict(
        type='RepeatDataset',
        # This is what makes it a 3x training schedule
        times=3,
        dataset=dict(
            type=dataset_type,
            ann_file=data_root + train_ann_file,
            img_prefix=data_root + train_img_prefix,
            classes=classes,
            pipeline=train_pipeline)
    ),
    # val=None,
    val=dict(
        type=dataset_type,
        ann_file=data_root + test_ann_file,
        img_prefix=data_root + test_img_prefix,
        classes=classes,
        pipeline=test_pipeline,
    ),
    test=dict(
        type=dataset_type,
        ann_file=data_root + test_ann_file,
        img_prefix=data_root + test_img_prefix,
        classes=classes,
        pipeline=test_pipeline
    ),
)
# This is saving the best model according to average recall @ 100 detections
# We never have more in the image than this.
evaluation = dict(
    interval=1,
    metric=['proposal', 'bbox', 'segm'],
    proposal_nums=[1, 10, 100],
    save_best='AR@1000'
)

# optimizer
optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)

# learning policy
# Experiments show that using step=[9, 11] has higher performance
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[9, 11])
runner = dict(type='EpochBasedRunner', max_epochs=12)
