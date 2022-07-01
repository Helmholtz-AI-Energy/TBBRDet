# Import our own loader pipeline
custom_imports = dict(imports=['numpy_loader'], allow_failed_imports=False)

# Modify dataset related settings
dataset_type = 'CocoDataset'
classes = ('Thermal bridge',)
# workers_per_gpu = 16

# Overwriting this to set values from previous paper
img_norm_cfg = dict(
    mean=[130., 135., 135., 118., 118.],
    std=[44., 40., 40., 30., 21.],
    to_rgb=False
)

data_root = '/path/to/dataset/'
train_img_prefix = 'train/images/'
train_ann_file = 'train/annotations/Flug1_onelabel_coco.json'
test_img_prefix = 'test/images/'
test_ann_file = 'test/annotations/Flug1_onelabel_coco.json'


mmdet_config_base = '/path/to/configs/mmdet/'
