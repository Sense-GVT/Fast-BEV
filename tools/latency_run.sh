#!/usr/bin/env bash
T=`date +%m%d%H%M`

MMDET3D=${MMDET3D:-/mnt/cache/huangbin1/workspace/m2bev}
SRUN_ARGS=${SRUN_ARGS:-""}
echo MMDET3D: $MMDET3D
echo SRUN_ARGS: $SRUN_ARGS

function spring_slurm_train {
    PARTITION=$1
    GPUS=$2
    JOB_TYPE=$3
    EXPNAME=$4
    JOB_NAME=latency-spring_slurm_train-${5:-`basename $EXPNAME`}
    GPUS_PER_NODE=$(($GPUS<8?$GPUS:8))

    echo spring_slurm_train; sleep 0.5s
    MMDET3D=$MMDET3D \
    SRUN_ARGS=$SRUN_ARGS \
    GPUS=$GPUS GPUS_PER_NODE=$GPUS_PER_NODE \
    sh ./tools/spring_slurm_train.sh \
        $PARTITION \
        $JOB_TYPE \
        $JOB_NAME \
        configs/uniconv/exp/$EXPNAME.py \
        work_dirs/uniconv/exp/$EXPNAME \
        --resume-from work_dirs/uniconv/exp/$EXPNAME/latest.pth
}

function slurm_train {
    PARTITION=$1
    GPUS=$2
    EXPNAME=$3
    JOB_NAME=latency-slurm_train-${4:-`basename $EXPNAME`}
    GPUS_PER_NODE=$(($GPUS<8?$GPUS:8))

    echo slurm_train; sleep 0.5s
    MMDET3D=$MMDET3D \
    SRUN_ARGS=$SRUN_ARGS \
    GPUS=$GPUS GPUS_PER_NODE=$GPUS_PER_NODE \
    sh ./tools/slurm_train.sh \
        $PARTITION \
        $JOB_NAME \
        configs/uniconv/exp/$EXPNAME.py \
        work_dirs/uniconv/exp/$EXPNAME \
        --resume-from work_dirs/uniconv/exp/$EXPNAME/latest.pth
}

function slurm_test {
    PARTITION=$1
    GPUS=$2
    EXPNAME=$3
    JOB_NAME=${4:-`basename $EXPNAME`}
    RESUME=${5:-work_dirs/uniconv/exp/$EXPNAME/latest.pth}
    GPUS_PER_NODE=$(($GPUS<8?$GPUS:8))

    echo slurm_test; sleep 0.5s
    MMDET3D=$MMDET3D \
    SRUN_ARGS=$SRUN_ARGS \
    GPUS=$GPUS GPUS_PER_NODE=$GPUS_PER_NODE \
    sh ./tools/slurm_test.sh \
        $PARTITION \
        $JOB_NAME \
        configs/uniconv/exp/$EXPNAME.py \
        $RESUME \
        --out work_dirs/uniconv/exp/$EXPNAME/results/results.pkl \
        --eval bbox \
        2>&1 | tee work_dirs/uniconv/exp/$EXPNAME/log.test.$JOB_NAME.$T.txt
}

function slurm_analysis {
    PARTITION=$1
    EXPNAME=$2
    JOB_NAME=latency-analysis-${3:-`basename $EXPNAME`}

    echo slurm_analysis; sleep 0.5s
    mkdir -p work_dirs/uniconv/exp/$EXPNAME/
    MMDET3D=$MMDET3D \
    SRUN_ARGS=$SRUN_ARGS \
    GPUS=1 GPUS_PER_NODE=1 \
    sh ./tools/slurm_analysis.sh \
        $PARTITION \
        $JOB_NAME \
        configs/uniconv/exp/$EXPNAME.py \
        work_dirs/uniconv/exp/$EXPNAME/latest.pth \
    2>&1 | tee work_dirs/uniconv/exp/$EXPNAME/log.benchmark.$T.txt
}

PARTITION=$1
QUOTATYPE=$2

#################### train

# spring_slurm_train $PARTITION 32 $QUOTATYPE latency/uniconv_v0.1_r18_s256x704_v200x200x6
# spring_slurm_train $PARTITION 32 $QUOTATYPE latency/uniconv_v0.1_r50_s256x704_v200x200x6

# spring_slurm_train $PARTITION 32 $QUOTATYPE latency/uniconv_v0.1_r18_s256x704_v100x100x1
# spring_slurm_train $PARTITION 32 $QUOTATYPE latency/uniconv_v0.1_r50_s256x704_v100x100x1

# spring_slurm_train $PARTITION 32 $QUOTATYPE latency/uniconv_v0.1_r18_s256x704_v200x200x4
# spring_slurm_train $PARTITION 32 $QUOTATYPE latency/uniconv_v0.1_r18_s256x704_v200x200x3
# spring_slurm_train $PARTITION 32 $QUOTATYPE latency/uniconv_v0.1_r18_s256x704_v200x200x2
# spring_slurm_train $PARTITION 32 $QUOTATYPE latency/uniconv_v0.1_r18_s256x704_v200x200x1

# spring_slurm_train $PARTITION 32 $QUOTATYPE latency/uniconv_v0.1_r18_s256x704_v100x100x6
# spring_slurm_train $PARTITION 32 $QUOTATYPE latency/uniconv_v0.1_r18_s256x704_v125x125x6
# spring_slurm_train $PARTITION 32 $QUOTATYPE latency/uniconv_v0.1_r18_s256x704_v250x250x6
# spring_slurm_train $PARTITION 32 $QUOTATYPE latency/uniconv_v0.1_r18_s256x704_v400x400x6

# spring_slurm_train $PARTITION 32 $QUOTATYPE latency/uniconv_v0.1_r18_s256x704_v250x250x4

# spring_slurm_train $PARTITION 32 $QUOTATYPE latency/uniconv_v0.1_r18_s256x704_v400x400x2
# spring_slurm_train $PARTITION 32 $QUOTATYPE latency/uniconv_v0.1_r18_s256x704_v400x400x3
# spring_slurm_train $PARTITION 32 $QUOTATYPE latency/uniconv_v0.1_r18_s256x704_v400x400x4

# spring_slurm_train $PARTITION 32 $QUOTATYPE latency/uniconv_v0.1_r18_s384x1056_v250x250x4
# spring_slurm_train $PARTITION 32 $QUOTATYPE latency/uniconv_v0.1_r18_s512x1408_v250x250x4

# spring_slurm_train $PARTITION 32 $QUOTATYPE latency/uniconv_v0.1_r18_s256x704_v200x200x6_n3d2
# spring_slurm_train $PARTITION 32 $QUOTATYPE latency/uniconv_v0.1_r18_s256x704_v200x200x6_n3d4
# spring_slurm_train $PARTITION 32 $QUOTATYPE latency/uniconv_v0.1_r18_s256x704_v200x200x6_n3d8

# spring_slurm_train $PARTITION 32 $QUOTATYPE latency/uniconv_v0.1_r18_s256x704_v200x200x6_n3c128
# spring_slurm_train $PARTITION 32 $QUOTATYPE latency/uniconv_v0.1_r18_s256x704_v200x200x6_n3c192

# spring_slurm_train $PARTITION 32 $QUOTATYPE latency/uniconv_v0.1_r50_s256x704_v250x250x4

# spring_slurm_train $PARTITION 32 $QUOTATYPE latency/uniconv_v0.1_r18_s384x1056_v200x200x6
# spring_slurm_train $PARTITION 32 $QUOTATYPE latency/uniconv_v0.1_r18_s512x1408_v200x200x6

#################### analysis

# MASTER_PORT=29580 slurm_analysis $PARTITION latency/uniconv_v0.1_r18_s256x704_v200x200x6 &
# MASTER_PORT=29581 slurm_analysis $PARTITION latency/uniconv_v0.1_r34_s256x704_v200x200x6 &
# MASTER_PORT=29582 slurm_analysis $PARTITION latency/uniconv_v0.1_r50_s256x704_v200x200x6 &
# MASTER_PORT=29583 slurm_analysis $PARTITION latency/uniconv_v0.1_r101_s256x704_v200x200x6 &

# MASTER_PORT=29584 slurm_analysis $PARTITION latency/uniconv_v0.1_r18_s256x704_v200x200x4 &
# MASTER_PORT=29585 slurm_analysis $PARTITION latency/uniconv_v0.1_r18_s256x704_v200x200x3 &
# MASTER_PORT=29585 slurm_analysis $PARTITION latency/uniconv_v0.1_r18_s256x704_v200x200x2 &
# MASTER_PORT=29586 slurm_analysis $PARTITION latency/uniconv_v0.1_r18_s256x704_v200x200x1 &

# MASTER_PORT=29587 slurm_analysis $PARTITION latency/uniconv_v0.1_r18_s256x704_v100x100x6 &
# MASTER_PORT=29587 slurm_analysis $PARTITION latency/uniconv_v0.1_r18_s256x704_v125x125x6 &
# MASTER_PORT=29587 slurm_analysis $PARTITION latency/uniconv_v0.1_r18_s256x704_v250x250x6 &
# MASTER_PORT=29588 slurm_analysis $PARTITION latency/uniconv_v0.1_r18_s256x704_v400x400x6 &

# MASTER_PORT=29589 slurm_analysis $PARTITION latency/uniconv_v0.1_r18_s256x704_v250x250x4 &

# MASTER_PORT=29590 slurm_analysis $PARTITION latency/uniconv_v0.1_r18_s256x704_v400x400x2 &
# MASTER_PORT=29590 slurm_analysis $PARTITION latency/uniconv_v0.1_r18_s256x704_v400x400x3 &
# MASTER_PORT=29590 slurm_analysis $PARTITION latency/uniconv_v0.1_r18_s256x704_v400x400x4 &

# MASTER_PORT=29591 slurm_analysis $PARTITION latency/uniconv_v0.1_r18_s384x1056_v250x250x4 &
# MASTER_PORT=29592 slurm_analysis $PARTITION latency/uniconv_v0.1_r18_s512x1408_v250x250x4 &

# MASTER_PORT=29593 slurm_analysis $PARTITION latency/uniconv_v0.1_r18_s256x704_v200x200x6_n3d2 &
# MASTER_PORT=29594 slurm_analysis $PARTITION latency/uniconv_v0.1_r18_s256x704_v200x200x6_n3d4 &
# MASTER_PORT=29595 slurm_analysis $PARTITION latency/uniconv_v0.1_r18_s256x704_v200x200x6_n3d8 &

# MASTER_PORT=29595 slurm_analysis $PARTITION latency/uniconv_v0.1_r18_s256x704_v200x200x6_n3c128 &
# MASTER_PORT=29595 slurm_analysis $PARTITION latency/uniconv_v0.1_r18_s256x704_v200x200x6_n3c192 &

# MASTER_PORT=29596 slurm_analysis $PARTITION latency/uniconv_v0.1_r50_s256x704_v250x250x4 &

# MASTER_PORT=29597 slurm_analysis $PARTITION latency/uniconv_v0.1_r18_s256x704_v100x100x1 &
# MASTER_PORT=29598 slurm_analysis $PARTITION latency/uniconv_v0.1_r50_s256x704_v100x100x1 &
# MASTER_PORT=29599 slurm_analysis $PARTITION latency/uniconv_v0.1_r101_s256x704_v100x100x1 &

# MASTER_PORT=29600 slurm_analysis $PARTITION latency/uniconv_v0.1_r18_s384x1056_v200x200x6 &
# MASTER_PORT=29601 slurm_analysis $PARTITION latency/uniconv_v0.1_r18_s512x1408_v200x200x6 &

#################### test

# QUOTATYPE=spot slurm_test $PARTITION 16 latency/uniconv_v0.1_r18_s256x704_v200x200x6


wait
