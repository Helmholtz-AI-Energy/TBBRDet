_base_ = './tridentnet_r50_caffe_mstrain_1x_coco_ablation.py'

lr_config = dict(step=[28, 34])
runner = dict(type='EpochBasedRunner', max_epochs=36)

evaluation = dict(
    # Trident seems to need longer to produce any predictions initially
    start=10,
)
