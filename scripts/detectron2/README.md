# Detectron2

These are all scripts for running Detectron2 experiments.


## Structure

```
├── README.md                             <- This file
├── evaluation                            <- Evaluation scripts
│   ├── analyze_model.py                  <- Analyze a Detectrons2 model
│   ├── combine_evaluation_scores.py      <- Detectron2 training scripts
│   ├── combine_evaluation_scores_csv.py  <- Detectron2 training scripts
│   └── visualize_json_results.py         <- Visualize json instance detection/segmentation results
└── training                              <- Training scripts
    ├── slurm_submit.sh                   <- Example submission script to train on Slurm cluster
    ├── submit_all_seeds.sh               <- Convenience script to submit all experiments
    ├── thermal_dataset_mapper.py         <- Class to load NumPy images as inputs
    └── train_net.py                      <- Training script
```
    

## Training

Detectron2 training scripts come from the [Detectron2 repository](https://github.com/facebookresearch/detectron2) tools directory.
See that for explanations of their function and arguments.

For this work, the `--ablation` flag was added to `train_net.py` to execute trainings without the height map inputs.

Additionally, the slurm submission scripts were adapted from Detectron2 to work with our particular cluster and will likely need to be
modified to work with your own, so we provide it simply as a template for users.
This includes the convenience script `submit_all_sees.sh` to submit embarrassingly parallel trainings with different seeds.

## Evaluation

The evaluation scripts `analyze_model.py` and `visualize_json_results.py` were similarly adapted from Detectron2 to work with our 5-channel inputs.

An additional set of scripts to combine evaluation scores and print the LaTeX code for table rows of mean and standard
deviation for the corresponding paper is included.

