# This is copied from the MaskRCNN R50 config used in experiments
# The changes are removing augmentations from the pipeline and setting the test
# set as the train set, this lets you use browse_dataset.py to visualise the test set.
checkpoint_config = dict(interval=1)
log_config = dict(
    interval=50,
    hooks=[dict(type='TextLoggerHook'),
           dict(type='TensorboardLoggerHook')])
custom_hooks = [dict(type='NumClassCheckHook')]
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
mlflow_tracking_uri = 'sqlite:////hkfs/work/workspace/scratch/cd4062-wahn22/drone_dataset/Flug1_merged/mlflow/mlruns.db'
mlflow_artifact_root = '/hkfs/work/workspace/scratch/cd4062-wahn22/drone_dataset/Flug1_merged/mlflow/mlartifacts/'
custom_imports = dict(imports=['numpy_loader'], allow_failed_imports=False)
dataset_type = 'CocoDataset'
classes = ('Thermal bridge', )
data_root = '/hkfs/work/workspace/scratch/cd4062-wahn22/drone_dataset/Flug1_merged/'
train_img_prefix = 'train/images/'
train_ann_file = 'train/annotations/Flug1_onelabel_coco.json'
test_img_prefix = 'test/images/'
test_ann_file = 'test/annotations/Flug1_onelabel_coco.json'
img_norm_cfg = dict(
    mean=[130.0, 135.0, 135.0, 118.0, 118.0],
    std=[44.0, 40.0, 40.0, 30.0, 21.0],
    to_rgb=False)
train_pipeline = [
    dict(type='LoadNumpyImageFromFile', drop_height=False),
    dict(
        type='LoadAnnotations',
        with_bbox=True,
        with_mask=True,
        poly2mask=False),
    # dict(
    #     type='Resize',
    #     img_scale=[(3370, 2144), (3370, 2680)],
    #     multiscale_mode='range',
    #     keep_ratio=True),
    # dict(type='RandomFlip', flip_ratio=0.5),
    dict(
        type='Normalize',
        mean=[130.0, 135.0, 135.0, 118.0, 118.0],
        std=[44.0, 40.0, 40.0, 30.0, 21.0],
        to_rgb=False),
    # dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks'])
]
test_pipeline = [
    dict(type='LoadNumpyImageFromFile', drop_height=False),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(3370, 2680),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            # dict(type='RandomFlip'),
            dict(
                type='Normalize',
                mean=[130.0, 135.0, 135.0, 118.0, 118.0],
                std=[44.0, 40.0, 40.0, 30.0, 21.0],
                to_rgb=False),
            # dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=16,
    train=dict(
        type='RepeatDataset',
        times=3,
        dataset=dict(
            type='CocoDataset',
            ann_file=
            '/hkfs/work/workspace/scratch/cd4062-wahn22/drone_dataset/Flug1_merged/test/annotations/Flug1_onelabel_coco.json',
            img_prefix=
            '/hkfs/work/workspace/scratch/cd4062-wahn22/drone_dataset/Flug1_merged/test/images/',
            classes=('Thermal bridge', ),
            pipeline=[
                dict(type='LoadNumpyImageFromFile', drop_height=False),
                dict(
                    type='LoadAnnotations',
                    with_bbox=True,
                    with_mask=True,
                    poly2mask=False),
                # dict(
                #     type='Resize',
                #     img_scale=[(3370, 2144), (3370, 2680)],
                #     multiscale_mode='range',
                #     keep_ratio=True),
                # dict(type='RandomFlip', flip_ratio=0.5),
                dict(
                    type='Normalize',
                    mean=[130.0, 135.0, 135.0, 118.0, 118.0],
                    std=[44.0, 40.0, 40.0, 30.0, 21.0],
                    to_rgb=False),
                dict(type='Pad', size_divisor=32),
                dict(type='DefaultFormatBundle'),
                dict(
                    type='Collect',
                    keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks'])
            ])),
    val=dict(
        type='CocoDataset',
        ann_file=
        '/hkfs/work/workspace/scratch/cd4062-wahn22/drone_dataset/Flug1_merged/test/annotations/Flug1_onelabel_coco.json',
        img_prefix=
        '/hkfs/work/workspace/scratch/cd4062-wahn22/drone_dataset/Flug1_merged/test/images/',
        classes=('Thermal bridge', ),
        pipeline=[
            dict(type='LoadNumpyImageFromFile', drop_height=False),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(3370, 2680),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    # dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[130.0, 135.0, 135.0, 118.0, 118.0],
                        std=[44.0, 40.0, 40.0, 30.0, 21.0],
                        to_rgb=False),
                    dict(type='Pad', size_divisor=32),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]),
    test=dict(
        type='CocoDataset',
        ann_file=
        '/hkfs/work/workspace/scratch/cd4062-wahn22/drone_dataset/Flug1_merged/test/annotations/Flug1_onelabel_coco.json',
        img_prefix=
        '/hkfs/work/workspace/scratch/cd4062-wahn22/drone_dataset/Flug1_merged/test/images/',
        classes=('Thermal bridge', ),
        pipeline=[
            dict(type='LoadNumpyImageFromFile', drop_height=False),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(3370, 2680),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    # dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[130.0, 135.0, 135.0, 118.0, 118.0],
                        std=[44.0, 40.0, 40.0, 30.0, 21.0],
                        to_rgb=False),
                    dict(type='Pad', size_divisor=32),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]))
evaluation = dict(
    interval=1,
    metric=['proposal', 'bbox', 'segm'],
    proposal_nums=[1, 10, 100],
    save_best='AR@1000')
optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[9, 11])
runner = dict(type='EpochBasedRunner', max_epochs=12)
model = dict(
    type='MaskRCNN',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=False,
        in_channels=5,
        style='pytorch',
        init_cfg=dict()),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5),
    rpn_head=dict(
        type='RPNHead',
        in_channels=256,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[8],
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[0.0, 0.0, 0.0, 0.0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
    roi_head=dict(
        type='StandardRoIHead',
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=dict(
            type='Shared2FCBBoxHead',
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=1,
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0.0, 0.0, 0.0, 0.0],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            reg_class_agnostic=False,
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
        mask_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=14, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        mask_head=dict(
            type='FCNMaskHead',
            num_convs=4,
            in_channels=256,
            conv_out_channels=256,
            num_classes=1,
            loss_mask=dict(
                type='CrossEntropyLoss', use_mask=True, loss_weight=1.0))),
    train_cfg=dict(
        rpn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.7,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                match_low_quality=True,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=256,
                pos_fraction=0.5,
                neg_pos_ub=-1,
                add_gt_as_proposals=False),
            allowed_border=-1,
            pos_weight=-1,
            debug=False),
        rpn_proposal=dict(
            nms_pre=2000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.5,
                neg_iou_thr=0.5,
                min_pos_iou=0.5,
                match_low_quality=True,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=512,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True),
            mask_size=28,
            pos_weight=-1,
            debug=False)),
    test_cfg=dict(
        rpn=dict(
            nms_pre=1000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            score_thr=0.05,
            nms=dict(type='nms', iou_threshold=0.5),
            max_per_img=100,
            mask_thr_binary=0.5)))
fp16 = dict(loss_scale=dict(init_scale=512))
work_dir = '/hkfs/work/workspace/scratch/cd4062-wahn22/drone_dataset/Flug1_merged/output/mmdet/MaskRCNN/mask_rcnn_r50_fpn_mstrain-poly_3x_coco.scratch/1645963'
auto_resume = False
gpu_ids = range(0, 4)
