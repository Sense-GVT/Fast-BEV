#!/usr/bin/env bash
T=`date +%m%d%H%M`

MMDET3D=${MMDET3D:-/mnt/cache/huangbin1/workspace/m2bev}
SRUN_ARGS=${SRUN_ARGS:-""}
echo MMDET3D: $MMDET3D
echo SRUN_ARGS: $SRUN_ARGS

function slurm_vis {
    PARTITION=$1
    GPUS=$2
    EXPNAME=$3
    JOB_NAME=uniconv-vis-${4:-`basename $EXPNAME`}
    RESUME=${5:-work_dirs/uniconv/exp/$EXPNAME/latest.pth}
    GPUS_PER_NODE=$(($GPUS<8?$GPUS:8))

    echo FUNCTION: slurm_vis
    echo PARTITION: $PARTITION
    echo GPUS: $GPUS
    echo EXPNAME: $EXPNAME
    echo JOB_NAME: $JOB_NAME
    echo RESUME: $RESUME
    sleep 0.5s

    MMDET3D=$MMDET3D \
    SRUN_ARGS=$SRUN_ARGS \
    GPUS=$GPUS GPUS_PER_NODE=$GPUS_PER_NODE \
    sh ./tools/slurm_vis.sh \
        $PARTITION \
        $JOB_NAME \
        configs/uniconv/exp/$EXPNAME.py \
        $RESUME \
        --out work_dirs/uniconv/exp/$EXPNAME/results/results.pkl \
        --eval bbox \
        --show --show-dir work_dirs/uniconv/exp/$EXPNAME/results/vis
}

PARTITION=$1

MASTER_PORT=29581 slurm_vis $PARTITION 1 nuscenes/uniconv_v0.1_e48_s256x704_v200x200x6_ibaug_v2
