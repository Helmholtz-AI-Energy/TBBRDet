#!/bin/bash
#
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-gpu=1
#SBATCH --cpus-per-gpu=32
#SBATCH --time=05:00:00
#SBATCH --output=logs/%x.%j.out
#SBATCH --error=logs/%x.%j.err
#SBATCH --partition=<add your partition>
# # Need the following if we want to do energy measurements
#SBATCH --exclusive

# First check only one arg was given
if [[ "$#" -ne 2 ]]; then
        echo "Illegal number of parameters"
        echo "Usage: $0 <config file> <seed>"
        exit 1
fi
# Then check the file exists
if [ ! -f "$1" ]; then
        echo "$1 does not exist."
        echo "Usage: $0 <config file> <seed>"
        exit 1
fi
# Print config file name for stdout records
echo "Config file: $1";
echo "Random seed: $2";

# The following will depend on your cluster
SRUN_PARAMS=(
  --mpi="pmi2"
  --label
#  --cpu-bind="none"
  --cpus-per-task=16
  --unbuffered
)

export NCCL_IB_TIMEOUT=30
export SHARP_COLL_LOG_LEVEL=3
export OMPI_MCA_coll_hcoll_enable=0
export NCCL_SOCKET_IFNAME="ib0"
export NCCL_COLLNET_ENABLE=0
export NCCL_DEBUG_SUBSYS="INIT,GRAPH"

source /path/to/python/venv


# Comment out seed and deterministic lines for randomised trainings
srun "${SRUN_PARAMS[@]}" \
    python ~/Wahn/scripts/mmdet/train.py \
    --launcher="slurm" \
    --seed $2 \
    --deterministic \
    $1
