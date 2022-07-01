# #### COPY INTO BEGINNING OF CONFIG ####
from common_vars import (custom_imports, dataset_type, classes, img_norm_cfg, data_root,
                         train_img_prefix, train_ann_file, test_img_prefix, test_ann_file,
                         mlflow_tracking_uri, mlflow_artifact_root,
                         log_config, mmdet_config_base)

# #### Copy the following into all configs ####
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
        norm_eval=False
    ),
)
# #### END OF BEGINNING COPY SECTION ####

# ###########################################################################

# #### COPY INTO END OF CONFIG ####

# Set FP16 precision
# fp16 settings
fp16 = dict(loss_scale=512.)

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
    workers_per_gpu=16,
    train=dict(
        img_prefix=data_root + train_img_prefix,
        ann_file=data_root + train_ann_file,
        classes=classes,
        pipeline=train_pipeline,
    ),
    val=None,
    # val=dict(
    #     img_prefix='/hkfs/work/workspace/scratch/cd4062-wahn22/drone_dataset/Flug1_merged/test/images',
    #     classes=classes,
    #     ann_file='/hkfs/work/workspace/scratch/cd4062-wahn22/drone_dataset/Flug1_merged/test/annotations/Flug1_onelabel_coco.json'),
    test=dict(
        img_prefix=data_root + test_img_prefix,
        ann_file=data_root + test_ann_file,
        classes=classes,
        pipeline=test_pipeline,
    ),
)

# #### END OF END COPY SECTION ####
