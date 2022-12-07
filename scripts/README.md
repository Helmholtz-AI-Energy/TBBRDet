# Scripts

This contains all scripts for preprocessing, training, and evaluating models, as well as some useful utilities.

The experiments described in the paper used two libraries: `Detectron2` and `MMDetection`.
Their individual scripts are kept in their respective directories.

# Preprocessing

# Detectron2 scripts

## Training

Detectron2 training scripts come from the [Detectron2 repository](https://github.com/facebookresearch/detectron2) tools directory.
See that for explanations of their function and arguments.

For this work, the `--ablation` flag was added to `train_net.py` to execute trainings without the height map inputs.

Additionally, the slurm submission scripts were adapted from Detectron2 to work with our particular cluster and will likely need to be
modified to work with your own, so we provide it simply as a template for users.
This includes a convenience script to submit embarrassingly parallel trainings with different seeds.

## Evaluation

The evaluation scripts `analyze_model.py` and `visualize_json_results.py` were similarly adapted from Detectron2 to work with our 5-channel inputs.

An additional set of scripts to combine evaluation scores and print the LaTeX code for table rows of mean and standard
deviation for the corresponding paper is included.


# MMDetection scripts

For MMDetection instructions see the [MMDetection README](mmdet/README.md).

Because MMDetection requires importing the pipeline class defined `numpy_loader.py` to load inputs,
all MMDetection scripts are within a single directory.
Similarly, `common_vars.py` defines common config parameters.

All other scripts are adapted from the `tools/` scripts provided in the [MMDetection repository](https://github.com/open-mmlab/mmdetection).


## Training


## Testing


