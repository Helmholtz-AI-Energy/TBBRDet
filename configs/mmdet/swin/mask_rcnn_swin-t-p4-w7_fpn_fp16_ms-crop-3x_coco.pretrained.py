_base_ = './mask_rcnn_swin-t-p4-w7_fpn_fp16_ms-crop-3x_coco.py'

load_from = '/hkfs/work/workspace/scratch/cd4062-wahn22/mmdetection/swin/mask_rcnn_swin-t-p4-w7_fpn_fp16_ms-crop-3x_coco_20210908_165006-90a4008c.pth'

# All the below was only using a pretrained backbone, we need the full model
# This was the original but the first layer's input dims don't match (it's 3 for RGB instead of 5)
# pretrained = 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth'  # noqa
# This has the first layer's weights removed so we can finetune it
# pretrained = '/hkfs/work/workspace/scratch/cd4062-wahn22/mmdetection/swin/swin_tiny_patch4_window7_224.truncated.pth'  # noqa
# pretrained_init_cfg = dict(type='Pretrained', checkpoint=pretrained)

# model = dict(
#     backbone=dict(
#         # This sets the weights to use or not use for pretraining
#         init_cfg=pretrained_init_cfg,
#     ),
# )
