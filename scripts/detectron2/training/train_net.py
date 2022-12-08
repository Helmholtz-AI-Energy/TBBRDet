#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
"""
Official script from the updated Detectron2 repo,
I've adapted it to work with our dataset.


A main training script.

This scripts reads a given config file and runs the training or evaluation.
It is an entry point that is made to train standard models in detectron2.

In order to let one script support training of many models,
this script contains logic that are specific to these built-in models and therefore
may not be suitable for your own project.
For example, your research project perhaps only needs a single "evaluator".

Therefore, we recommend you to use detectron2 as an library and take
this file as an example of how to use the library.
You may want to write your own script with your datasets and other customizations.
"""

import logging
import os
from collections import OrderedDict
from pathlib import Path

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, hooks, launch
from detectron2.evaluation import (
    CityscapesInstanceEvaluator,
    CityscapesSemSegEvaluator,
    COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    PascalVOCDetectionEvaluator,
    SemSegEvaluator,
    verify_results,
)
from detectron2.modeling import GeneralizedRCNNWithTTA
from detectron2.data.datasets import register_coco_instances
# from detectron2.projects.panoptic_deeplab import add_panoptic_deeplab_config
from detectron2 import model_zoo
from detectron2.data import build_detection_train_loader, build_detection_test_loader
from detectron2.solver import build_lr_scheduler

from thermal_dataset_mapper import ThermalDatasetMapper


def build_evaluator(cfg, dataset_name, output_folder=None):
    """
    Create evaluator(s) for a given dataset.
    This uses the special metadata "evaluator_type" associated with each builtin dataset.
    For your own dataset, you can simply create an evaluator manually in your
    script and do not have to worry about the hacky if-else logic here.
    """
    if output_folder is None:
        output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
    evaluator_list = []
    evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
    if evaluator_type in ["sem_seg", "coco_panoptic_seg"]:
        evaluator_list.append(
            SemSegEvaluator(
                dataset_name,
                distributed=True,
                output_dir=output_folder,
            )
        )
    if evaluator_type in ["coco", "coco_panoptic_seg"]:
        evaluator_list.append(COCOEvaluator(dataset_name, output_dir=output_folder))
    if evaluator_type == "coco_panoptic_seg":
        evaluator_list.append(COCOPanopticEvaluator(dataset_name, output_folder))
    if evaluator_type == "cityscapes_instance":
        return CityscapesInstanceEvaluator(dataset_name)
    if evaluator_type == "cityscapes_sem_seg":
        return CityscapesSemSegEvaluator(dataset_name)
    elif evaluator_type == "pascal_voc":
        return PascalVOCDetectionEvaluator(dataset_name)
    elif evaluator_type == "lvis":
        return LVISEvaluator(dataset_name, output_dir=output_folder)
    if len(evaluator_list) == 0:
        raise NotImplementedError(
            "no Evaluator for the dataset {} with the type {}".format(dataset_name, evaluator_type)
        )
    elif len(evaluator_list) == 1:
        return evaluator_list[0]
    return DatasetEvaluators(evaluator_list)


class Trainer(DefaultTrainer):
    """
    We use the "DefaultTrainer" which contains pre-defined default logic for
    standard training workflow. They may not work for you, especially if you
    are working on a new research project. In that case you can write your
    own training loop. You can use "tools/plain_train_net.py" as an example.
    """

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        return build_evaluator(cfg, dataset_name, output_folder)

    @classmethod
    def test_with_TTA(cls, cfg, model):
        logger = logging.getLogger("detectron2.trainer")
        # In the end of training, run an evaluation with TTA
        # Only support some R-CNN models.
        logger.info("Running inference with test-time augmentation ...")
        model = GeneralizedRCNNWithTTA(cfg, model)
        evaluators = [
            cls.build_evaluator(
                cfg, name, output_folder=os.path.join(cfg.OUTPUT_DIR, "inference_TTA")
            )
            for name in cfg.DATASETS.TEST
        ]
        res = cls.test(cfg, model, evaluators)
        res = OrderedDict({k + "_TTA": v for k, v in res.items()})
        return res

    @classmethod
    def build_train_loader(cls, cfg):
        # The aspect_ratio_grouping flag should be true if images are different sizes
        return build_detection_train_loader(
            cfg,
            mapper=ThermalDatasetMapper(
                is_train=True,
                augmentations=[],
                use_instance_mask=True,
                ablation=cfg.ABLATION if "ABLATION" in cfg else False,
            ),
            aspect_ratio_grouping=False,
            num_workers=cfg.DATALOADER.NUM_WORKERS,
        )

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        return build_detection_test_loader(
            cfg,
            dataset_name,
            mapper=ThermalDatasetMapper(
                is_train=False,
                augmentations=[],
                use_instance_mask=True,
                ablation=cfg.ABLATION if "ABLATION" in cfg else False,
            ),
            num_workers=cfg.DATALOADER.NUM_WORKERS,
        )


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()

    # Model configs
    # you can choose alternative models as backbone here
    # if args.base_model == "DeepLab":
    #     add_panoptic_deeplab_config(cfg)
    if args.base_model == "Mask-RCNN":
        cfg.merge_from_file(model_zoo.get_config_file(
            "Misc/scratch_mask_rcnn_R_50_FPN_3x_gn.yaml"
            # "Cityscapes/mask_rcnn_R_50_FPN.yaml"
        ))

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)

    # Adding for ablation
    cfg.ABLATION = args.ablation

    cfg.freeze()

    # Check if the output dir exists, if so only proceed if we're resuming training
    if not args.eval_only and not args.resume:
        assert not Path(cfg.OUTPUT_DIR).exists(), f"{cfg.OUTPUT_DIR} exists, training can only proceed if resume flag is set"

    default_setup(cfg, args)

    return cfg


def main(args):

    # the data has to be registered within detectron2, once for the train and once for the test data
    top_path = Path('/path/to/dataset/')
    for d in ['train', 'test']:
        register_coco_instances(
            f'thermals_{d}',
            {},
            top_path / d / 'annotations' / 'Flug1_onelabel_coco.json',
            top_path / d / 'images',
        )

    cfg = setup(args)

    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model)
        if cfg.TEST.AUG.ENABLED:
            res.update(Trainer.test_with_TTA(cfg, model))
        if comm.is_main_process():
            verify_results(cfg, res)
        return res

    """
    If you'd like to do anything fancier than the standard training logic,
    consider writing your own training loop (see plain_train_net.py) or
    subclassing the trainer.
    """
    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    if cfg.TEST.AUG.ENABLED:
        trainer.register_hooks(
            [hooks.EvalHook(0, lambda: trainer.test_with_TTA(cfg, trainer.model))]
        )
    return trainer.train()


if __name__ == "__main__":

    parser = default_argument_parser()
    # Going to add some extra command line arguments
    parser.add_argument("--base-model", type=str, default='Mask-RCNN', choices=['Mask-RCNN', 'DeepLab'], help="Base model to inherit parameters from")
    parser.add_argument("--ablation", action="store_true", help="Flag to leave of height map for ablation study")

    args = parser.parse_args()
    print("Command Line Args:", args)

    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
