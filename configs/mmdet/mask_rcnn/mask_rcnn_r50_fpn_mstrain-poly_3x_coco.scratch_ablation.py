# This is the default with ablation setup
# I removed the pretrained model from the maskrcnn _base_ config already
_base_ = [
    './mask_rcnn_r50_fpn_mstrain-poly_3x_coco_ablation.py',
]
