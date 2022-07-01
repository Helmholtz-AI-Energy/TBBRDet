from common_vars import (custom_imports, dataset_type, classes, img_norm_cfg, data_root,
                         train_img_prefix, train_ann_file, test_img_prefix, test_ann_file,
                         mlflow_tracking_uri, mlflow_artifact_root,
                         workers_per_gpu,
                         log_config, mmdet_config_base)
# #### End of copy section ####

# The new config inherits a base config to highlight the necessary modification
_base_ = [
    f'{mmdet_config_base}/mask_rcnn/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco.py',
]

# #### Copy the following into all configs ####
train_pipeline = [dict(type='LoadNumpyImageFromFile', drop_height=False)]
test_pipeline = [dict(type='LoadNumpyImageFromFile', drop_height=False)]

# We also need to change the num_classes in head to match the dataset's annotation
model = dict(
    roi_head=dict(
        bbox_head=dict(num_classes=1),
        mask_head=dict(num_classes=1)),
    pretrained=None,
    backbone=dict(
        in_channels=5,
        norm_eval=False
    ),
)


# Extend each pipeline with whatever the model needs
train_pipeline.extend([
    dict(
        type='LoadAnnotations',
        with_bbox=True,
        with_mask=True,
        poly2mask=False),
    dict(
        type='Resize',
        # These set the maximum long and short edge sizes, keep_ratio
        # means the real sizes will conform to the image ratio
        # Using th eshort edge only here to effectively scale everything
        img_scale=[(3370, 2400), (3370, 2450), (3370, 2500), (3370, 2550),
                   (3370, 2600), (3370, 2680)],
        multiscale_mode='value',
        keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks']),
])
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

# We can use the pre-trained Mask RCNN model to obtain higher performance
# load_from = 'checkpoints/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco_bbox_mAP-0.408__segm_mAP-0.37_20200504_163245-42aa3d00.pth'
load_from = ''


# #### COPY INTO END OF CONFIG ####
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=workers_per_gpu,
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

# slurm_jobid = os.environ['SLURM_JOBID']
# log_config = dict(
#     interval=50,
#     hooks=[
#         dict(type='TextLoggerHook'),
#         dict(type='TensorboardLoggerHook'),
#         dict(
#             type='MlflowLoggerHook',
#             exp_name='mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_thermals',
#             tags=dict(SLURM_JOBID=0)
#         ),
#     ])
# #### END OF END COPY SECTION ####
