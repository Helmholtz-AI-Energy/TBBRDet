from common_vars import custom_imports, img_norm_cfg

_base_ = [
    '../_base_/models/faster_rcnn_r50_caffe_c4.py',
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

model = dict(
    type='TridentFasterRCNN',
    backbone=dict(
        type='TridentResNet',
        trident_dilations=(1, 2, 3),
        num_branch=3,
        test_branch_idx=1,
        # init_cfg=dict(
        #     type='Pretrained',
        #     checkpoint='open-mmlab://detectron2/resnet50_caffe')),
    ),
    roi_head=dict(type='TridentRoIHead', num_branch=3, test_branch_idx=1),
    train_cfg=dict(
        rpn_proposal=dict(max_per_img=500),
        rcnn=dict(
            sampler=dict(num=128, pos_fraction=0.5,
                         add_gt_as_proposals=False)
        )
    )
)

train_pipeline = [
    dict(type='LoadNumpyImageFromFile', drop_height=False),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(3370, int(3.35 * 800)), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
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
            dict(type='Collect', keys=['img'])
        ])
]
data = dict(
    samples_per_gpu=2,
    train=dict(pipeline=train_pipeline),
    val=dict(
        pipeline=test_pipeline,
        samples_per_gpu=2,
    ),
    test=dict(
        pipeline=test_pipeline,
        samples_per_gpu=2,
    ))
