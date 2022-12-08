# MMDetection scripts

For MMDetection instructions see the [MMDetection README](mmdet/README.md).

Because MMDetection requires importing the pipeline class defined `numpy_loader.py` to load inputs,
all MMDetection scripts are within a single directory.
Similarly, `common_vars.py` defines common config parameters.

All other scripts are adapted from the `tools/` scripts provided in the [MMDetection repository](https://github.com/open-mmlab/mmdetection).

## Structure

```
├── README.md                       <- This file
├── analyze_results.py              <- Plots loss/mAP curves
├── browse_dataset.py               <- Visualise annotations and dataset
├── combine_evaluation_scores.py    <- Calculate mean/std of evaluation scores
├── common_vars.py                  <- Common configurations
├── numpy_loader.py                 <- Pipeline class to load NumPy inputs
├── slurm_bulk_test.sh              <- Submit multiple evaluation jobs
├── slurm_submit.sh                 <- Submit a single training job
├── submit_all_seeds.sh             <- Run multiple trainings with different seeds
├── test.py                         <- Evaluate a trained model
└── train.py                        <- Train a model
```


## Visualisation

For plotting the training dataset, I've set up a copy of the config used for MaskRCNN R50 training but with augmentations removed.
You can plot it with something like
```bash
python browse_dataset.py --output-dir /path/to.output/ --channel RGB --not-show ~/Wahn/configs/mmdet/common/plotting_config.py
```

## Training

TBD

## Testing

TBD

## Analyzing results

You should set `TOPDIR` to the top-level work directory of the training, then run
```bash
python analyze_results.py $TOPDIR/mask_*.py $TOPDIR/evaluation/eval_results.pkl $TOPDIR/evaluation/images_0.3/ --show-score-thr 0.3 --topk 50
```
