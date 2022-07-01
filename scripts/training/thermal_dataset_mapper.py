import copy
from typing import List, Union

import numpy as np
import torch

from detectron2.data import detection_utils, transforms, DatasetMapper


class ThermalDatasetMapper(DatasetMapper):
    ''' The default DatasetMapper with the image loading changed to load a numpy file

    Note that this is identical to the default DatasetMapper, I've just changed the file loading line to load a numpy file
    instead of a JPG as is default.
    '''
    def __init__(
        self,
        is_train: bool,
        *args,
        augmentations: List[Union[transforms.Augmentation, transforms.Transform]],
        image_format=None,
        ablation=False,
        **kwargs
    ):

        self.ablation = ablation

        super().__init__(
            is_train=is_train,
            *args,
            augmentations=augmentations,
            image_format=image_format,
            **kwargs,
        )

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        # USER: Write your own image loading if it's not from a file
#         image = utils.read_image(dataset_dict["file_name"], format=self.image_format)
        image = np.load(dataset_dict["file_name"])  # (H,W,C) in BGR mode?

        if self.ablation:
            image = image[:, :, :4]

        detection_utils.check_image_size(dataset_dict, image)

        # USER: Remove if you don't do semantic/panoptic segmentation.
        if "sem_seg_file_name" in dataset_dict:
            sem_seg_gt = detection_utils.read_image(dataset_dict.pop("sem_seg_file_name"), "L").squeeze(2)
        else:
            sem_seg_gt = None

        aug_input = transforms.AugInput(image, sem_seg=sem_seg_gt)
        aug_transforms = self.augmentations(aug_input)
        image, sem_seg_gt = aug_input.image, aug_input.sem_seg

        image_shape = image.shape[:2]  # h, w
        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
        if sem_seg_gt is not None:
            dataset_dict["sem_seg"] = torch.as_tensor(sem_seg_gt.astype("long"))

        # USER: Remove if you don't use pre-computed proposals.
        # Most users would not need this feature.
        if self.proposal_topk is not None:
            detection_utils.transform_proposals(
                dataset_dict, image_shape, aug_transforms, proposal_topk=self.proposal_topk
            )

        if not self.is_train:
            # USER: Modify this if you want to keep them for some reason.
            dataset_dict.pop("annotations", None)
            dataset_dict.pop("sem_seg_file_name", None)
            return dataset_dict

        if "annotations" in dataset_dict:
            # USER: Modify this if you want to keep them for some reason.
            for anno in dataset_dict["annotations"]:
                if not self.use_instance_mask:
                    anno.pop("segmentation", None)
                if not self.use_keypoint:
                    anno.pop("keypoints", None)

            # USER: Implement additional transformations if you have other types of data
            annos = [
                detection_utils.transform_instance_annotations(
                    obj, aug_transforms, image_shape, keypoint_hflip_indices=self.keypoint_hflip_indices
                )
                for obj in dataset_dict.pop("annotations")
                if obj.get("iscrowd", 0) == 0
            ]
            instances = detection_utils.annotations_to_instances(
                annos, image_shape, mask_format=self.instance_mask_format
            )

            # After transforms such as cropping are applied, the bounding box may no longer
            # tightly bound the object. As an example, imagine a triangle object
            # [(0,0), (2,0), (0,2)] cropped by a box [(1,0),(2,2)] (XYXY format). The tight
            # bounding box of the cropped triangle should be [(1,0),(2,1)], which is not equal to
            # the intersection of original bounding box and the cropping box.
            if self.recompute_boxes:
                instances.gt_boxes = instances.gt_masks.get_bounding_boxes()
            dataset_dict["instances"] = detection_utils.filter_empty_instances(instances)
        return dataset_dict
