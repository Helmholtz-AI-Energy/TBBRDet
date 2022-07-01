# Import our own loader pipeline
custom_imports = dict(imports=['numpy_loader'], allow_failed_imports=False)

# dataset settings
dataset_type = 'CocoDataset'
data_root = '/hkfs/work/workspace/scratch/cd4062-wahn22/drone_dataset/Flug1_merged/'
classes = ('Thermal bridge',)

train_img_prefix = 'train/images/'
train_ann_file = 'train/annotations/Flug1_onelabel_coco.json'
test_img_prefix = 'test/images/'
test_ann_file = 'test/annotations/Flug1_onelabel_coco.json'

img_norm_cfg = dict(
    mean=[130., 135., 135., 118., 118.],
    std=[44., 40., 40., 30., 21.],
    to_rgb=False
)
train_pipeline = [
    dict(type='LoadNumpyImageFromFile', drop_height=False),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='Resize', img_scale=(3370, (3.35 * 800)), keep_ratio=True),
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
    samples_per_gpu=2,
    workers_per_gpu=32,
    train=dict(
        type=dataset_type,
        img_prefix=data_root + train_img_prefix,
        ann_file=data_root + train_ann_file,
        pipeline=train_pipeline,
        classes=classes,
    ),
    # We use the test set to evaluate iteratively
    # val=None,
    val=dict(
        type=dataset_type,
        img_prefix=data_root + test_img_prefix,
        ann_file=data_root + test_ann_file,
        pipeline=test_pipeline,
        classes=classes,
    ),
    test=dict(
        type=dataset_type,
        img_prefix=data_root + test_img_prefix,
        ann_file=data_root + test_ann_file,
        pipeline=test_pipeline,
        classes=classes,
    ),
)
# This is saving the best model according to average recall @ 100 detections
# We never have more in the image than this.
# evaluation = dict(metric=['bbox', 'segm'])
# evaluation = dict(interval=1, metric=['proposal', 'bbox', 'segm'], save_best='AR@100')
evaluation = dict(
    interval=1,
    metric=['proposal', 'bbox', 'segm'],
    proposal_nums=[1, 10, 100],
    save_best='AR@1000'
)
