#!/bin/bash
#
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-gpu=1
#SBATCH --cpus-per-gpu=32
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x.%j.out
#SBATCH --error=logs/%x.%j.err
#SBATCH --partition=<add your partition>
# # Need the following if we want to do energy measurements
# #SBATCH --exclusive

# First check only one arg was given
if [[ "$#" -ne 2 ]]; then
        echo "Illegal number of parameters"
        echo "Usage: sbatch $0 <top dir> <metric>"
        echo "E.g.: sbatch $0 /path/to/topdir/ \"best_AR@1000\""
        exit 1
fi
# Print config file name for stdout records
echo "Top directory: $1";
echo "Metric: $2";

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


for jobid in $(ls $1)
do
    TOPDIR=${1}/${jobid}
    echo Evaluating $TOPDIR
    srun "${SRUN_PARAMS[@]}" \
        python /path/to/Wahn/scripts/mmdet/test.py \
        ${TOPDIR}/*.py \
        ${TOPDIR}/${2}*.pth \
        --work-dir ${TOPDIR}/evaluation/ \
        --out ${TOPDIR}/evaluation/eval_results.pkl \
        --eval proposal bbox segm \
        --launcher slurm
done
