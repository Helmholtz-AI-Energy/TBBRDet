_base_ = [
    '../common/mstrain-poly_3x_coco_instance.py',
    '../_base_/models/mask_rcnn_r50_fpn.py'
]

# you need to set mode='dynamic' if you are using pytorch<=1.5.0
fp16 = dict(loss_scale=dict(init_scale=512))
