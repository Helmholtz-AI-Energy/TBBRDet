#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.

import argparse
import json
import numpy as np
from collections import defaultdict
import cv2
import tqdm
from pathlib import Path

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import Boxes, BoxMode, Instances
from detectron2.utils.file_io import PathManager
from detectron2.utils.logger import setup_logger
from detectron2.utils.visualizer import Visualizer
from detectron2.data.datasets import register_coco_instances


def create_instances(predictions, image_size):
    ret = Instances(image_size)

    score = np.asarray([x["score"] for x in predictions])
    chosen = (score > args.conf_threshold).nonzero()[0]
    score = score[chosen]
    bbox = np.asarray([predictions[i]["bbox"] for i in chosen]).reshape(-1, 4)
    bbox = BoxMode.convert(bbox, BoxMode.XYWH_ABS, BoxMode.XYXY_ABS)

    labels = np.asarray([dataset_id_map(predictions[i]["category_id"]) for i in chosen])

    ret.scores = score
    ret.pred_boxes = Boxes(bbox)
    ret.pred_classes = labels

    try:
        ret.pred_masks = [predictions[i]["segmentation"] for i in chosen]
    except KeyError:
        pass
    return ret


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="A script that visualizes the json predictions from COCO or LVIS dataset."
    )
    parser.add_argument("--input", required=True, help="JSON file produced by the model")
    parser.add_argument("--output", required=True, help="output directory")
    parser.add_argument("--dataset", help="name of the dataset", default="thermals_test")
    parser.add_argument("--channel", type=str, default='RGB', choices=['RGB', 'Thermal', 'Depth'], help="Which colour channels to plot")
    parser.add_argument("--conf-threshold", default=0.5, type=float, help="confidence threshold")
    args = parser.parse_args()

    logger = setup_logger()

    with PathManager.open(args.input, "r") as f:
        predictions = json.load(f)

    pred_by_image = defaultdict(list)
    for p in predictions:
        pred_by_image[p["image_id"]].append(p)

    # Register the dataset
    top_path = Path('/path/to/dataset/')
    register_coco_instances(
        'thermals_test',
        {},
        top_path / 'test' / 'annotations' / 'Flug1_onelabel_coco.json',
        top_path / 'test' / 'images',
    )

    dicts = list(DatasetCatalog.get(args.dataset))
    metadata = MetadataCatalog.get(args.dataset)
    if hasattr(metadata, "thing_dataset_id_to_contiguous_id"):

        def dataset_id_map(ds_id):
            return metadata.thing_dataset_id_to_contiguous_id[ds_id]

    elif "lvis" in args.dataset:
        # LVIS results are in the same format as COCO results, but have a different
        # mapping from dataset category id to contiguous category id in [0, #categories - 1]
        def dataset_id_map(ds_id):
            return ds_id - 1

    else:
        raise ValueError("Unsupported dataset: {}".format(args.dataset))

    output_dir = Path(args.output, args.channel)
    output_dir.mkdir(parents=True, exist_ok=True)

    for dic in tqdm.tqdm(dicts):
        # img = cv2.imread(dic["file_name"], cv2.IMREAD_COLOR)[:, :, ::-1]
        img = np.load(dic["file_name"])  # (H,W,C) originally in BGR mode?
        # This will get back to RGB
        if args.channel == 'RGB':
            img = img[:, :, 2::-1]
        elif args.channel == 'Thermal':
            img = img[:, :, 3]
        elif args.channel == 'Depth':
            img = img[:, :, 4]
        else:
            raise ValueError(f"Channel argument {args.channel} invalid")

        # Will use same filename, but in the default case it's a jpg and we have numpy
        # basename = os.path.basename(dic["file_name"])
        basename = Path(dic["file_name"]).stem

        predictions = create_instances(pred_by_image[dic["image_id"]], img.shape[:2])
        vis = Visualizer(img, metadata)
        vis_pred = vis.draw_instance_predictions(predictions).get_image()

        vis = Visualizer(img, metadata)
        vis_gt = vis.draw_dataset_dict(dic).get_image()

        concat = np.concatenate((vis_pred, vis_gt), axis=1)
        # cv2.imwrite(os.path.join(args.output, basename), concat[:, :, ::-1])
        cv2.imwrite(
            str(output_dir / f"{basename}.jpg"),
            concat[:, :, ::-1]
        )
