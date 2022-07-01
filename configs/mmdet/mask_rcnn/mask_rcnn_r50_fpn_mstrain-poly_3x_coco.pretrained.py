_base_ = [
    './mask_rcnn_r50_fpn_mstrain-poly_3x_coco.py',
]

# Set the model to load with first layer/norm deleted
load_from = '/path/to/mask_rcnn/mask_rcnn_r50_fpn_mstrain-poly_3x_coco_20210524_201154-21b550bb.truncated.pth'
