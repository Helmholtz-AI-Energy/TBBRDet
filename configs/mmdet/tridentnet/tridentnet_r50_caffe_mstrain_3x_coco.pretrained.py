_base_ = './tridentnet_r50_caffe_mstrain_3x_coco.py'

# Doesn't work, try ImageNet1k backbone instead
load_from = '/hkfs/work/workspace/scratch/cd4062-wahn22/mmdetection/tridentnet/tridentnet_r50_caffe_mstrain_3x_coco_20201130_100539-46d227ba.pth'
# model = dict(
#     backbone=dict(
#         init_cfg=dict(
#             type='Pretrained',
#             checkpoint='open-mmlab://detectron2/resnet50_caffe'),
#     ),
# )

# Turn this down from 0.02 to avoid NaNs and let it converge
optimizer = dict(type='SGD', lr=0.002, momentum=0.9, weight_decay=0.0001)

evaluation = dict(
    # Trident seems to need longer to produce any predictions initially
    start=20,
)
