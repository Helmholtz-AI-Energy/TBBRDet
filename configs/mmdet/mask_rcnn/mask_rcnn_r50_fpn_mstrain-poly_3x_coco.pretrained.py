_base_ = [
    './mask_rcnn_r50_fpn_mstrain-poly_3x_coco.py',
]

# Set the model to load with first layer/norm deleted
load_from = '/hkfs/work/workspace/scratch/cd4062-wahn22/mmdetection/mask_rcnn/mask_rcnn_r50_fpn_mstrain-poly_3x_coco_20210524_201154-21b550bb.truncated.pth'
# model = dict(
#     backbone=dict(
#         # This sets the weights to use or not use for pretraining
#         init_cfg=dict(
#             type='Pretrained',
#             checkpoint='/hkfs/work/workspace/scratch/cd4062-wahn22/mmdetection/mask_rcnn/mask_rcnn_r50_fpn_mstrain-poly_3x_coco_20210524_201154-21b550bb.truncated.pth',
#         ),
#     ),
# )

# DEPRECATED: Using 3x with checkpointing now
# Set a 1x training schedule to finetune
# data = dict(
#     train=dict(
#         # type='RepeatDataset',
#         # Turn this down to make a 1x training schedule
#         # In the inherited config it's times=3
#         times=1,
#     ),
# )
