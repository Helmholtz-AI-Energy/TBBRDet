import os.path as osp
import numpy as np

from mmdet.datasets import PIPELINES


@PIPELINES.register_module()
class LoadNumpyImageFromFile:
    """Load a NumPy image from file.

    Required keys are "img_prefix" and "img_info" (a dict that must contain the
    key "filename"). Added or updated keys are "filename", "img", "img_shape",
    "ori_shape" (same as `img_shape`), "pad_shape" (same as `img_shape`),
    "scale_factor" (1.0) and "img_norm_cfg" (means=0 and stds=1).

    Args:
        to_float32 (bool): Whether to convert the loaded image to a float32
            numpy array. If set to False, the loaded image is an uint8 array.
            Defaults to False.
        drop_height (bool): Whether to drop the height channel. This assumes a
            5-channel input and will drop the last.
            Defaults to False.
    """

    def __init__(
        self,
        to_float32=False,
        drop_height=False,
    ):
        self.to_float32 = to_float32
        self.drop_height = drop_height

    def __call__(self, results):
        """Call functions to load image and get image meta information.

        Args:
            results (dict): Result dict from :obj:`mmdet.CustomDataset`.

        Returns:
            dict: The dict contains loaded image and meta information.
        """

        if results['img_prefix'] is not None:
            filename = osp.join(results['img_prefix'],
                                results['img_info']['filename'])
        else:
            filename = results['img_info']['filename']

        img = np.load(filename)

        # Channels are last, we assume the height map is the final channel
        if self.drop_height:
            assert img.shape[-1] == 5, f"Image must have 5 colour channels to for dropping height map, found {img.shape[-1]}"
            img = img[..., :4]

        if self.to_float32:
            img = img.astype(np.float32)

        results['filename'] = filename
        results['ori_filename'] = results['img_info']['filename']
        results['img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        results['img_fields'] = ['img']
        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'to_float32={self.to_float32}, '
                    f"color_type='{self.color_type}', "
                    f'file_client_args={self.file_client_args})')
        return repr_str
