# Copyright (c) OpenMMLab. All rights reserved.
import argparse
# import os
from collections import Sequence
from pathlib import Path
from joblib import Parallel, delayed

from tqdm import tqdm
# import mmcv
from mmcv import Config, DictAction

from mmdet.core.utils import mask2ndarray
from mmdet.core.visualization import imshow_det_bboxes
from mmdet.datasets.builder import build_dataset


def parse_args():
    parser = argparse.ArgumentParser(description='Browse a dataset')
    parser.add_argument('config', help='train config file path')
    parser.add_argument(
        '--skip-type',
        type=str,
        nargs='+',
        default=['DefaultFormatBundle', 'Normalize', 'Collect'],
        help='skip some useless pipeline')
    parser.add_argument(
        '--output-dir',
        default=None,
        type=str,
        help='If there is no display interface, you can save it')
    parser.add_argument(
        "--channel",
        type=str,
        default='RGB',
        choices=['RGB', 'Thermal', 'Depth'],
        help="Which colour channels to plot"
    )
    parser.add_argument('--not-show', default=False, action='store_true')
    parser.add_argument(
        '-n',
        '--num-processes',
        type=int,
        default=1,
        help='Number of parallel processes to use')
    parser.add_argument(
        '--show-interval',
        type=float,
        default=2,
        help='the interval of show (s)')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    args = parser.parse_args()
    return args


def retrieve_data_cfg(config_path, skip_type, cfg_options):

    def skip_pipeline_steps(config):
        config['pipeline'] = [
            x for x in config.pipeline if x['type'] not in skip_type
        ]

    cfg = Config.fromfile(config_path)
    if cfg_options is not None:
        cfg.merge_from_dict(cfg_options)
    train_data_cfg = cfg.data.train
    while 'dataset' in train_data_cfg and train_data_cfg[
            'type'] != 'MultiImageMixDataset':
        train_data_cfg = train_data_cfg['dataset']

    if isinstance(train_data_cfg, Sequence):
        [skip_pipeline_steps(c) for c in train_data_cfg]
    else:
        skip_pipeline_steps(train_data_cfg)

    return cfg


def process_image(item, args, class_names):
    """ Process and plot/save a single image
    """
    # Set the output filename (None = not saving?)
    filename = None
    if args.output_dir is not None:
        in_file = Path(item['filename'])
        # Since it's numpy, change the extension to jpg so cv2.imwrite
        filename = Path(
            args.output_dir,
            in_file.parts[-2],
            args.channel,
            in_file.name
        ).with_suffix('.jpg')
        # print(filename)

    gt_masks = item.get('gt_masks', None)
    if gt_masks is not None:
        gt_masks = mask2ndarray(gt_masks)

    # Extract the image channel we need
    if args.channel == 'RGB':
        img = item['img'][:, :, :3]
        # Colorblind friendly colours
        # Need to stand out on red rooftops
        bbox_color = (26, 255, 26)
        mask_color = (75, 0, 146)
        # bbox_color = (17, 119, 51)
        # mask_color = (136, 204, 238)
        thickness = 3
    elif args.channel == 'Thermal':
        img = item['img'][:, :, 3]
        # Colorblind friendly colours
        bbox_color = (220, 50, 32)
        mask_color = (0, 90, 181)
        thickness = 4
    elif args.channel == 'Depth':
        img = item['img'][:, :, 4]
        # Colorblind friendly colours
        bbox_color = (220, 50, 32)
        mask_color = (0, 90, 181)
        thickness = 4

    imshow_det_bboxes(
        img=img,
        bboxes=item['gt_bboxes'],
        labels=item['gt_labels'],
        segms=gt_masks,
        class_names=class_names,
        show=not args.not_show,
        wait_time=args.show_interval,
        out_file=str(filename),
        thickness=thickness,
        text_color=(255, 102, 61),
        # Colorblind friendly colours
        bbox_color=bbox_color,
        mask_color=mask_color,
        # bbox_color=(255, 102, 61),
        # text_color=(255, 102, 61),
        font_size=1,
    )


def main():
    args = parse_args()
    cfg = retrieve_data_cfg(args.config, args.skip_type, args.cfg_options)

    dataset = build_dataset(cfg.data.train)
    class_names = dataset.CLASSES

    # progress_bar = mmcv.ProgressBar(len(dataset))

    Parallel(n_jobs=args.num_processes)(delayed(process_image)(item, args, class_names) for item in tqdm(dataset))


if __name__ == '__main__':
    main()
