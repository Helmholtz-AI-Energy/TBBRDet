from common_vars import custom_imports, img_norm_cfg


_base_ = '../swin/mask_rcnn_swin-t-p4-w7_fpn_ms-crop-3x_coco.py'
# you need to set mode='dynamic' if you are using pytorch<=1.5.0
fp16 = dict(loss_scale=dict(init_scale=512))

train_pipeline = [dict(type='LoadNumpyImageFromFile')]
test_pipeline = [dict(type='LoadNumpyImageFromFile')]

# We also need to change the num_classes in head to match the dataset's annotation
model = dict(
    roi_head=dict(
        bbox_head=dict(num_classes=1),
        mask_head=dict(num_classes=1)),
    # pretrained=None,
    backbone=dict(
        in_channels=5,
        # norm_eval=False
    ),
)

# From the swin config, modified for our images
# augmentation strategy originates from DETR / Sparse RCNN
train_pipeline.extend([
    dict(type='LoadImageFromFile'),
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
                    keep_ratio=True)
            ]
        ]),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks']),
])

# This is from _base_/datasets/coco_instance.py
# Will likely need the same in all config files
test_pipeline.extend([
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
])

# Set data location and pipeline
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=32,
    train=dict(
        # img_prefix=data_root + train_img_prefix,
        # ann_file=data_root + train_ann_file,
        pipeline=train_pipeline,
    ),
    val=dict(
        pipeline=test_pipeline,
    ),
    test=dict(
        # img_prefix=data_root + test_img_prefix,
        # ann_file=data_root + test_ann_file,
        pipeline=test_pipeline,
    ),
)

# #### END OF END COPY SECTION ####
