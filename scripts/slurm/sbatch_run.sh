#!/bin/bash -l
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

CMD=
IMAGE_VER=6
IMAGE=/netscratch/$USER/envs/xai_torch_v$IMAGE_VER.sqsh
WORK_DIR=$SCRIPT_DIR/../../
DATA_ROOT_DIR=/ds-sds/
MOUNTS=/netscratch/$USER:/netscratch/$USER,/ds-sds:/ds-sds,/home/$USER/:/home/$USER/
CACHE_DIR=/netscratch/$USER/cache
PYTHON_PATH=$WORK_DIR/src:$WORK_DIR/external/torchfusion/src
TORCH_FUSION_OUTPUT_DIR=/netscratch/$USER/torchfusion
EXPORTS="TERM=linux,NCCL_SOCKET_IFNAME=bond,NCCL_IB_HCA=mlx5,ROOT_DIR=/,USER_DIR=$USER,DATA_ROOT_DIR=$DATA_ROOT_DIR,TORCH_FUSION_CACHE_DIR=$CACHE_DIR,TORCH_FUSION_OUTPUT_DIR=$TORCH_FUSION_OUTPUT_DIR,PYTHONPATH=$PYTHON_PATH,TORCH_HOME=$CACHE_DIR/pretrained"

NODES=1
TASKS=1
GPUS_PER_TASK=1
CPUS_PER_TASK=8
PARTITION=batch
MEMORY=40
ARRAY=0
JOB_NAME=

usage()
{
    echo "Usage:"
    echo "./sbatch_run.sh --cmd=<cmd>"
    echo ""
    echo " --cmd : Command to run. "
    echo " --image: Container image to use. "
    echo " --work-dir: Path to work directory. "
    echo " --mounts: Directories to mount. "
    echo " --nodes : Number of nodes."
    echo " --tasks : Number of tasks per node."
    echo " --gpus_per_task : Number of GPUs per task."
    echo " --cpus_per_task : Number of GPUs per task."
    echo " --array : Number of tasks in the array."
    echo " -p | --partition : GPU partition"
    echo " -m | --memory : Total memory"
    echo " -h | --help : Displays the help"
    echo ""
}

while [ "$1" != "" ]; do
    PARAM=`echo $1 | awk -F= '{print $1}'`
    VALUE=`echo $1 | awk -F= '{print $2}'`
    case $PARAM in
        -h | --help)
            usage
            exit
            ;;
        --cmd)
            CMD=$VALUE
            ;;
        --image)
            IMAGE=$VALUE
            ;;
        --work-dir)
            WORK_DIR=$VALUE
            ;;
        --mounts)
            MOUNTS="$MOUNTS,$VALUE"
            ;;
        --nodes)
            NODES=$VALUE
            ;;
        --tasks)
            TASKS=$VALUE
            ;;
        --gpus_per_task )
            GPUS_PER_TASK=$VALUE
            ;;
        --cpus_per_task )
            CPUS_PER_TASK=$VALUE
            ;;
	    -p  | --partition )
	        PARTITION=$VALUE
	        ;;
	    -m  | --memory )
	        MEMORY=$VALUE
	        ;;
	    -a  | --array )
	        ARRAY=$VALUE
	        ;;
	    -jn  | --job-name )
	        JOB_NAME=$VALUE
	        ;;
        *)
            echo "ERROR: unknown parameter \"$PARAM\""
            usage
            exit 1
            ;;
    esac
    shift
done

if [ "$JOB_NAME" = "" ]; then
  usage
  exit 1
fi

if [ "$CMD" = "" ]; then
  usage
  exit 1
fi

if [ "$TASKS" -gt "1" ]; then
    WORLD_SIZE=$(($GPUS_PER_TASK * $TASKS))
    EXPORTS="$EXPORTS,MASTER_PORT=12345,WORLD_SIZE=$WORLD_SIZE"
fi

# debugging flags (optional)
export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1

echo $PARTITION
MEMORY=$(($MEMORY * $TASKS))
if [ $GPUS_PER_TASK == 0 ]; then
    sbatch --array=$ARRAY --job-name=$JOB_NAME --nodes=$NODES \
        --ntasks-per-node=$(($TASKS / $NODES)) \
        --ntasks=$TASKS \
        --time=2-0 \
        --cpus-per-task=$CPUS_PER_TASK \
        --mem="${MEMORY}G" \
        -p $PARTITION \
        --output="$TORCH_FUSION_OUTPUT_DIR/slurm_logs/${JOB_NAME}_%a.txt" \
        --wrap "srun \
                --container-image=$IMAGE \
                --container-workdir=$WORK_DIR \
                --container-mounts=$MOUNTS \
                --export=$EXPORTS \
                --task-prolog=$WORK_DIR/scripts/slurm/install.sh \
                -K \
                $CMD"
else
    sbatch --array=$ARRAY --job-name=$JOB_NAME --nodes=$NODES \
        --ntasks-per-node=$(($TASKS / $NODES)) \
        --ntasks=$TASKS \
        --time=2-0 \
        --cpus-per-task=$CPUS_PER_TASK \
        --gpus-per-task=$GPUS_PER_TASK \
        --mem="${MEMORY}G" \
        -p $PARTITION \
        --output="$TORCH_FUSION_OUTPUT_DIR/slurm_logs/${JOB_NAME}_%a.txt" \
        --wrap "srun \
                --container-image=$IMAGE \
                --container-workdir=$WORK_DIR \
                --container-mounts=$MOUNTS \
                --export=$EXPORTS \
                --task-prolog=$WORK_DIR/scripts/slurm/install.sh \
                -K \
                $CMD"
fi
