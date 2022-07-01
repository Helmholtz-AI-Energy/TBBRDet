# mmdetection scripts

Since `mmdetection` requires importing custom pipeline modules, all the scripts are in one place.

## Visualisation

For plotting the training dataset, I've set up a copy of the config used for MaskRCNN R50 training but with augmentations removed.
You can plot it with something like
```bash
python browse_dataset.py --output-dir /path/to.output/ --channel RGB --not-show ~/Wahn/configs/mmdet/common/plotting_config.py
```

## Training

## Testing

## Analyzing results

You should set `TOPDIR` to the top-level work directory of the training, then run
```bash
python analyze_results.py $TOPDIR/mask_*.py $TOPDIR/evaluation/eval_results.pkl $TOPDIR/evaluation/images_0.3/ --show-score-thr 0.3 --topk 50
```
