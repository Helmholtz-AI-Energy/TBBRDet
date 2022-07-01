#!/bin/bash

# First check only two args were given
if [[ "$#" -ne 2 ]]; then
        echo "Illegal number of parameters"
        echo "Usage: $0 <config file> <job name>"
        exit 1
fi
# Then check the config file exists
if [ ! -f "$1" ]; then
        echo "$1 does not exist."
        echo "Usage: $0 <config file> <job name>"
        exit 1
fi
# Print config file name for stdout records
echo "Config file: $1";

for seed in 3000 10117 10001 20770001 1008111
do
    echo "Submitting job: ${2}_${seed}";
    sbatch -J ${2}_${seed} ./slurm_submit.sh $1 ${seed}
done
