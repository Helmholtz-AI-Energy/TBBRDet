# Thermal Bridges on Building Rooftops Detection (TBBRDet)

## Installation

### Detectron2

To install you'll need a Python virtual environment with Detectron2 and PyTorch.
We used CUDA 11.1 for the paper.
```bash
# Set up virtual environment
python3 -m venv ~/venvs/tbbrdet_det2
. ~/venvs/tbbrdet_det2/bin/activate

# Install dependencies
pip install --upgrade pip
pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html
python -m pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu111/torch1.8/index.html
```
We have provided a requirements file, we recommend using that to set up the remainder of the environment correctly.


### MMDetection)

Create a separate venv for this.
In it install mmdetection according to the instructions on their repository (we used version `2.21.0`), and also install 
tensorboard and mlflow.
```bash
pip install future tensorboard mlflow
```
Again, we provide a requirements file, though we recommend first setting up mmdetection and pytorch first and then installing the remaining requirements with the requirements file.

#### Using MMDetection pretrained models

To train with pretrained weights provided by MMDetection it's advised to download the corresponding weight files.
You will find the name of the pretrained file used in our work within the corresponding config file in `config/mmdet/`.
You can then find the file within the MMDetection github repository `configs/` directory of that model.

## TBBR Dataset

You can find the latest version of the TBBR dataset here on Zenodo: https://doi.org/10.5281/zenodo.4767771

## Usage

Paths to datasets/MLFlow/outputs/etc. need to be set within the config files. 
You can search for the string `/path/to/` to know which to replace.

For library usage instructions, see either the [MMDetection README](scripts/mmdet/README.md) or the [Detectron2 README](scripts/README.md).


## Structure
```
├── README.md                           <- This file
├── configs                             <- Directory for configurations of model & application
│   ├── detectron2                      <- Detectron2 experiment configs
│   └── mmdet                           <- MMDetection experiment configs
└── scripts                             <- Python and Slurm scripts for running experiments
    ├── evaluation                      <- Detectron2 evaluation scripts
    ├── mmdet                           <- All MMDetection experiment scripts
    ├── training                        <- Detectron2 training scripts
    └── utils                           <- Miscellaneous utility scripts
```