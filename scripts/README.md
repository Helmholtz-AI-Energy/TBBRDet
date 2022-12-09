# Scripts

This contains all scripts for preprocessing, training, and evaluating models, as well as some useful utilities.

The experiments described in the paper used two libraries: `Detectron2` and `MMDetection`.
Their individual scripts are kept in their respective directories.

For all scripts, running `python <script> --help` will show all options.

## Structure

```
├── README.md                           <- This file
├── alignment                           <- Image merging and alignment
├── detectron2                          <- Detectron2 scripts
│   ├── training                        <- Detectron2 training scripts
│   └── evaluation                      <- Detectron2 evaluation scripts
├── mmdet                               <- All MMDetection experiment scripts
└── utils                               <- Miscellaneous utilies
```
