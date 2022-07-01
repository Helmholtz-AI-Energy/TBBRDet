#!/bin/bash
#
#SBATCH --nodes=1
#SBATCH --cpus-per-task=128
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x.%j.out
#SBATCH --error=logs/%x.%j.err
#SBATCH --partition=<add your partition>
#SBATCH --gres=gpu:4
# # Need the following if we want to do energy measurements
#SBATCH --exclusive

# First check only one arg was given
if [[ "$#" -ne 3 ]]; then
        echo "Illegal number of parameters"
        echo "Usage: sbatch $0 <config file> <seed> <output dir>"
        exit 1
fi
# Then check the file exists
if [ ! -f "$1" ]; then
        echo "$1 does not exist."
        echo "Usage: sbatch $0 <config file> <seed> <output dir>"
        exit 1
fi
# Print config file name for stdout records
echo "Config file: $1";
echo "Random seed: $2";
echo "Output dir: $3";


source /path/to/python/venv

if grep -qi "ablation" <<< $1
then
    echo "Running ablation training"
    python3 /path/to/scripts/training/train_net.py --config-file $1 --num-gpus 4 --ablation SEED $2 OUTPUT_DIR $3
else
    echo "Running full input training"
    python3 /path/to/scripts/training/train_net.py --config-file $1 --num-gpus 4 SEED $2 OUTPUT_DIR $3
fi
