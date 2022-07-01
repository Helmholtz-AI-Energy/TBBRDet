_base_ = './retinanet_r50_fpn_1x_coco.py'
# learning policy
lr_config = dict(step=[28, 34])
runner = dict(type='EpochBasedRunner', max_epochs=36)
