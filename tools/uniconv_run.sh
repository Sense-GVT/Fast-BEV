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
    JOB_NAME=${5:-`basename $EXPNAME`}
    GPUS_PER_NODE=$(($GPUS<8?$GPUS:8))

    echo spring_slurm_train; sleep 0.5s
    # $QUOTATYPE-retrain
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
    JOB_NAME=${4:-`basename $EXPNAME`}
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
        --eval bbox
}


function batch_slurm_test {
    PARTITION=$1
    GPUS=$2
    EXPNAME=$3
    JOB_NAME=${4:-`basename $EXPNAME`}
    GPUS_PER_NODE=$(($GPUS<8?$GPUS:8))
    mkdir -p work_dirs/uniconv/exp/$EXPNAME/batch
    mkdir -p work_dirs/uniconv/exp/$EXPNAME/test/
    for RESUME in `ls work_dirs/uniconv/exp/$EXPNAME/batch/epoch*.pth`; do
        BASENAME=`basename $RESUME`
        SUB_JOB_NAME=test-$JOB_NAME-$BASENAME
        echo slurm_test; sleep 0.5s
        echo RESUME: $RESUME
        echo SUB_JOB_NAME: $SUB_JOB_NAME
        echo LOG: work_dirs/uniconv/exp/$EXPNAME/test/log.test.$BASENAME.$T
        MMDET3D=$MMDET3D \
        SRUN_ARGS=$SRUN_ARGS \
        GPUS=$GPUS GPUS_PER_NODE=$GPUS_PER_NODE QUOTATYPE="$QUOTATYPE" \
        sh ./tools/slurm_test.sh \
            $PARTITION \
            $SUB_JOB_NAME \
            configs/uniconv/exp/$EXPNAME.py \
            $RESUME \
            --out work_dirs/uniconv/exp/$EXPNAME/results/results.$BASENAME.pkl \
            --eval bbox \
            2>&1 | tee work_dirs/uniconv/exp/$EXPNAME/test/log.test.$BASENAME.$T \
        &
    done
}


function slurm_benchmark {
    PARTITION=$1
    EXPNAME=$2
    JOB_NAME=benchmark-${3:-`basename $EXPNAME`}

    echo slurm_analysis; sleep 0.5s
    MMDET3D=$MMDET3D \
    SRUN_ARGS=$SRUN_ARGS \
    GPUS=1 GPUS_PER_NODE=1 \
    ./tools/slurm_analysis.sh \
        $PARTITION \
        $JOB_NAME \
        configs/uniconv/exp/$EXPNAME.py \
        work_dirs/uniconv/exp/$EXPNAME/latest.pth
}


PARTITION=$1
QUOTATYPE=$2

#################### debug
# MASTER_PORT=29581 slurm_train $PARTITION 1 nuscenes/uniconv_v0.1_e48_s256x704_v200x200x6_seq3.1_ibaug_v2 debug-train
# MASTER_PORT=29582 slurm_test $PARTITION 1 nuscenes/uniconv_v0.1_e48_s256x704_v200x200x6_seq3.1_ibaug_v2 test work_dirs/uniconv/exp/nuscenes/uniconv_v0.1_e48_s256x704_v200x200x6_seq3.0_ibaug_v2/epoch_48.pth
# MASTER_PORT=29581 slurm_test $PARTITION 4 nuscenes/uniconv_v0.1_e12_s256x704_v200x200x6_seq3.1_ibaug_v2_interval4_f4s135 test
# MASTER_PORT=29581 slurm_train $PARTITION 1 nuscenes/debug debug-train
# MASTER_PORT=29581 slurm_train $PARTITION 1 nuscenes/uniconv_v0.1_e48_s256x704_v200x200x6_seq3.0_ibaug_v2 debug-train
# MASTER_PORT=29581 slurm_test $PARTITION 1 nuscenes/debug debug-test work_dirs/uniconv/exp/nuscenes/uniconv_v0.1_e48_s256x704_v200x200x6_seq2.0/epoch_48.pth
# MASTER_PORT=29581 slurm_test $PARTITION 1 nuscenes/uniconv_seg_v0.1_e48_s256x704_v200x200x6_seq2.0 debug-test work_dirs/uniconv/exp/nuscenes/uniconv_seg_v0.1_e48_s256x704_v200x200x6_seq2.0/epoch_12.pth
# MASTER_PORT=29581 slurm_train $PARTITION 1 nuscenes/uniconv_v0.1_e48_s256x704_v200x200x6_seq3.1_ibaug_v2_interval1_f4s135_ft4_lr1e-4
# MASTER_PORT=29581 slurm_train $PARTITION 1 nuscenes/uniconv_v0.1_e12_s256x704_v200x200x6_seq3.1_ibaug_v2_interval4_f5s-3-1+1+3


#################### fullset-cbgs-exps
# spring_slurm_train $PARTITION 32 $QUOTATYPE nuscenes/uniconv_v0.1_e12_s256x704_v200x200x6_seq3.1_ibaug_v2_interval4_f1
# spring_slurm_train $PARTITION 32 $QUOTATYPE nuscenes/uniconv_v0.1_e12_s256x704_v200x200x6_seq3.1_ibaug_v2_interval4_f1_cbgs
# spring_slurm_train $PARTITION 32 $QUOTATYPE nuscenes/uniconv_v0.1_e12_s256x704_v200x200x6_seq3.1_ibaug_v2_interval4_f1_anchor
# spring_slurm_train $PARTITION 32 $QUOTATYPE nuscenes/uniconv_v0.1_e12_s256x704_v200x200x6_seq3.1_ibaug_v2_interval4_f4s135
# spring_slurm_train $PARTITION 32 $QUOTATYPE nuscenes/uniconv_v0.1_e12_s256x704_v200x200x6_seq3.1_ibaug_v2_interval4_f4s135_cbgs
# spring_slurm_train $PARTITION 32 $QUOTATYPE nuscenes/uniconv_v0.1_e48_s256x704_v200x200x6_seq3.1_ibaug_v2_interval1_f2s4
# spring_slurm_train $PARTITION 32 $QUOTATYPE nuscenes/uniconv_v0.1_e48_s256x704_v200x200x6_seq3.1_ibaug_v2_interval1_f2s4_cbgs
# spring_slurm_train $PARTITION 32 $QUOTATYPE nuscenes/uniconv_v0.1_e48_s256x704_v200x200x6_seq3.1_ibaug_v2_interval1_f4s135
# spring_slurm_train $PARTITION 32 $QUOTATYPE nuscenes/uniconv_v0.1_e48_s256x704_v200x200x6_seq3.1_ibaug_v2_interval1_f4s135_cbgs
# spring_slurm_train $PARTITION 32 $QUOTATYPE nuscenes/uniconv_v0.1_e48_s256x704_v200x200x6_seq3.1_ibaug_v2_interval1_f4s135_ft4_lr1e-4
# spring_slurm_train $PARTITION 32 $QUOTATYPE nuscenes/uniconv_v0.1_e48_s256x704_v200x200x6_seq3.1_ibaug_v2_interval1_f4s135_ft4_lr1e-4_nodecay
# spring_slurm_train $PARTITION 32 $QUOTATYPE nuscenes/uniconv_v0.1_e48_s256x704_v200x200x6_seq3.1_ibaug_v2_interval1_f4s135_ft4_lr1e-5
# spring_slurm_train $PARTITION 32 $QUOTATYPE nuscenes/uniconv_v0.1_r18_e48_s256x704_v200x200x6_seq3.1_ibaug_v2_interval1_f4s135_cbgs


#################### fullset-seq3.0-exps
# spring_slurm_train $PARTITION 32 $QUOTATYPE nuscenes/uniconv_v0.1_e12_s256x704_v200x200x6_seq3.1_ibaug_v2_interval4_f1
# spring_slurm_train $PARTITION 32 $QUOTATYPE nuscenes/uniconv_v0.1_e12_s256x704_v200x200x6_seq3.1_ibaug_v2_interval4_f2s1
# spring_slurm_train $PARTITION 32 $QUOTATYPE nuscenes/uniconv_v0.1_e12_s256x704_v200x200x6_seq3.1_ibaug_v2_interval4_f2s2
# spring_slurm_train $PARTITION 32 $QUOTATYPE nuscenes/uniconv_v0.1_e12_s256x704_v200x200x6_seq3.1_ibaug_v2_interval4_f2s3
# spring_slurm_train $PARTITION 32 $QUOTATYPE nuscenes/uniconv_v0.1_e12_s256x704_v200x200x6_seq3.1_ibaug_v2_interval4_f2s4
# spring_slurm_train $PARTITION 32 $QUOTATYPE nuscenes/uniconv_v0.1_e12_s256x704_v200x200x6_seq3.1_ibaug_v2_interval4_f2s5
# spring_slurm_train $PARTITION 32 $QUOTATYPE nuscenes/uniconv_v0.1_e12_s256x704_v200x200x6_seq3.1_ibaug_v2_interval4_f2s6
# spring_slurm_train $PARTITION 32 $QUOTATYPE nuscenes/uniconv_v0.1_e12_s256x704_v200x200x6_seq3.1_ibaug_v2_interval4_f3s01
# spring_slurm_train $PARTITION 32 $QUOTATYPE nuscenes/uniconv_v0.1_e12_s256x704_v200x200x6_seq3.1_ibaug_v2_interval4_f3s13
# spring_slurm_train $PARTITION 32 $QUOTATYPE nuscenes/uniconv_v0.1_e12_s256x704_v200x200x6_seq3.1_ibaug_v2_interval4_f3s25
# spring_slurm_train $PARTITION 32 $QUOTATYPE nuscenes/uniconv_v0.1_e12_s256x704_v200x200x6_seq3.1_ibaug_v2_interval4_f3s37
# spring_slurm_train $PARTITION 32 $QUOTATYPE nuscenes/uniconv_v0.1_e12_s256x704_v200x200x6_seq3.1_ibaug_v2_interval4_f4s012
# spring_slurm_train $PARTITION 32 $QUOTATYPE nuscenes/uniconv_v0.1_e12_s256x704_v200x200x6_seq3.1_ibaug_v2_interval4_f4s135
# spring_slurm_train $PARTITION 32 $QUOTATYPE nuscenes/uniconv_v0.1_e12_s256x704_v200x200x6_seq3.1_ibaug_v2_interval4_f4s258
# spring_slurm_train $PARTITION 32 $QUOTATYPE nuscenes/uniconv_v0.1_e12_s256x704_v200x200x6_seq3.1_ibaug_v2_interval4_f5s0123
# spring_slurm_train $PARTITION 32 $QUOTATYPE nuscenes/uniconv_v0.1_e12_s256x704_v200x200x6_seq3.1_ibaug_v2_interval4_f5s1357
# spring_slurm_train $PARTITION 8 reserved nuscenes/uniconv_v0.1_e12_s256x704_v200x200x6_seq3.1_ibaug_v2_interval4_f5s-3-1+1+3
# spring_slurm_train $PARTITION 8 reserved nuscenes/uniconv_v0.1_e12_s256x704_v200x200x6_seq3.1_ibaug_v2_interval4_f5s-5-3-1+1+3+5

# spring_slurm_train $PARTITION 32 $QUOTATYPE nuscenes/uniconv_v0.1_e48_s256x704_v200x200x6_seq3.1_ibaug_v2_interval1_f2s4
# spring_slurm_train $PARTITION 32 $QUOTATYPE nuscenes/uniconv_v0.1_e48_s256x704_v200x200x6_seq3.1_ibaug_v2_interval1_f3s13
# spring_slurm_train $PARTITION 32 $QUOTATYPE nuscenes/uniconv_v0.1_e48_s256x704_v200x200x6_seq3.1_ibaug_v2_interval1_f4s135
# spring_slurm_train $PARTITION 32 $QUOTATYPE nuscenes/uniconv_v0.1_e48_s256x704_v200x200x6_seq3.1_ibaug_v2_interval1_f5s1357
# spring_slurm_train $PARTITION 32 $QUOTATYPE nuscenes/uniconv_v0.1_e48_s256x704_v200x200x6_seq3.1_ibaug_v2_interval1_f5s-3-1+1+3
# spring_slurm_train $PARTITION 32 $QUOTATYPE nuscenes/uniconv_v0.1_e48_s256x704_v200x200x6_seq3.1_ibaug_v2_interval1_f7s-5-3-1+1+3+5


#################### fullset-train-exps
# spring_slurm_train $PARTITION 16 $QUOTATYPE nuscenes/uniconv_v0.1_e12_s256x704_v200x200x6
# spring_slurm_train $PARTITION 16 $QUOTATYPE nuscenes/uniconv_v0.1_e12_s256x704_v200x200x12
# spring_slurm_train $PARTITION 16 $QUOTATYPE nuscenes/uniconv_v0.1_e12_s544x960_v200x200x12
# spring_slurm_train $PARTITION 16 $QUOTATYPE nuscenes/uniconv_v0.1_e12_s544x960_v400x400x12
# spring_slurm_train $PARTITION 16 $QUOTATYPE nuscenes/uniconv_v0.1_e12_s928x1600_v200x200x12
# spring_slurm_train $PARTITION 16 $QUOTATYPE nuscenes/uniconv_v0.1_e12_s928x1600_v400x400x12


#################### fullset-aug-exps
#### s704x1208_v125x125x4
# spring_slurm_train $PARTITION 32 $QUOTATYPE nuscenes/uniconv_v0.1_e48_s704x1208_v125x125x4_seq2.0_ibaug_v2
# spring_slurm_train $PARTITION 32 $QUOTATYPE nuscenes/uniconv_v0.1_e48_s704x1208_v125x125x4_ibaug_v2
# spring_slurm_train $PARTITION 32 $QUOTATYPE nuscenes/uniconv_v0.1_e48_s704x1208_v125x125x4
# spring_slurm_train $PARTITION 32 $QUOTATYPE nuscenes/uniconv_v0.1_e48_s704x1208_v125x125x4_seq2.0


#### s256x704_v200x200x12
# spring_slurm_train $PARTITION 16 $QUOTATYPE nuscenes/uniconv_v0.1_e12_s256x704_v200x200x12_imgaug
# spring_slurm_train $PARTITION 16 $QUOTATYPE nuscenes/uniconv_v0.1_e12_s256x704_v200x200x12_bevaug
# spring_slurm_train $PARTITION 16 $QUOTATYPE nuscenes/uniconv_v0.1_e12_s256x704_v200x200x12_ibaug
# spring_slurm_train $PARTITION 16 $QUOTATYPE nuscenes/uniconv_v0.1_e24_s256x704_v200x200x12
# spring_slurm_train $PARTITION 16 $QUOTATYPE nuscenes/uniconv_v0.1_e24_s256x704_v200x200x12_bevaug
# spring_slurm_train $PARTITION 16 $QUOTATYPE nuscenes/uniconv_v0.1_e24_s256x704_v200x200x12_imgaug
# spring_slurm_train $PARTITION 16 $QUOTATYPE nuscenes/uniconv_v0.1_e24_s256x704_v200x200x12_ibaug
# spring_slurm_train $PARTITION 32 $QUOTATYPE nuscenes/uniconv_v0.1_e48_s256x704_v200x200x12
# spring_slurm_train $PARTITION 32 $QUOTATYPE nuscenes/uniconv_v0.1_e48_s256x704_v200x200x12_bevaug
# spring_slurm_train $PARTITION 32 $QUOTATYPE nuscenes/uniconv_v0.1_e48_s256x704_v200x200x12_imgaug
# spring_slurm_train $PARTITION 32 $QUOTATYPE nuscenes/uniconv_v0.1_e48_s256x704_v200x200x12_ibaug
#### s256x704_v200x200x6
# spring_slurm_train $PARTITION 32 $QUOTATYPE nuscenes/uniconv_v0.1_e48_s256x704_v200x200x6
# spring_slurm_train $PARTITION 32 $QUOTATYPE nuscenes/uniconv_v0.1_e48_s256x704_v200x200x6_imgaug
# spring_slurm_train $PARTITION 32 $QUOTATYPE nuscenes/uniconv_v0.1_e48_s256x704_v200x200x6_bevaug
# spring_slurm_train $PARTITION 32 $QUOTATYPE nuscenes/uniconv_v0.1_e48_s256x704_v200x200x6_bevaug_v2
# spring_slurm_train $PARTITION 32 $QUOTATYPE nuscenes/uniconv_v0.1_e48_s256x704_v200x200x6_ibaug
# spring_slurm_train $PARTITION 32 $QUOTATYPE nuscenes/uniconv_v0.1_e48_s256x704_v200x200x6_ibaug_v2
# spring_slurm_train $PARTITION 32 $QUOTATYPE nuscenes/uniconv_v0.1_e48_s256x704_v200x200x6_seq1.0
# spring_slurm_train $PARTITION 32 $QUOTATYPE nuscenes/uniconv_v0.1_e48_s256x704_v200x200x6_seq2.0
# spring_slurm_train $PARTITION 32 $QUOTATYPE nuscenes/uniconv_v0.1_e48_s256x704_v200x200x6_seq2.0_ibaug_v2
# spring_slurm_train $PARTITION 32 $QUOTATYPE nuscenes/uniconv_v0.1_e48_s256x704_v200x200x6_seq3.0_ibaug_v2


#################### fullset-seg-exps
# spring_slurm_train $PARTITION 32 $QUOTATYPE nuscenes/uniconv_seg_v0.1_e12_s256x704_v200x200x6
# spring_slurm_train $PARTITION 16 $QUOTATYPE nuscenes/uniconv_seg_v0.1_e12_s256x704_v200x200x6_bs16
# spring_slurm_train $PARTITION 32 $QUOTATYPE nuscenes/uniconv_seg_v0.1_e48_s256x704_v200x200x6
# spring_slurm_train $PARTITION 32 $QUOTATYPE nuscenes/uniconv_seg_v0.1_e48_s256x704_v200x200x6_imgaug
# spring_slurm_train $PARTITION 32 $QUOTATYPE nuscenes/uniconv_seg_v0.1_e48_s256x704_v200x200x6_seq2.0
# spring_slurm_train $PARTITION 32 $QUOTATYPE nuscenes/uniconv_seg_v0.1_e48_s256x704_v200x200x6_seq2.0_imgaug


#################### fullset-test
# MASTER_PORT=29581 slurm_test $PARTITION 8 nuscenes/uniconv_v0.1_e12_s256x704_v200x200x12 test work_dirs/uniconv/exp/nuscenes/uniconv_v0.1_e12_s256x704_v200x200x12/epoch_1.pth
# MASTER_PORT=29581 slurm_test $PARTITION 8 nuscenes/uniconv_v0.1_e12_s256x704_v200x200x12_imgaug test work_dirs/uniconv/exp/nuscenes/uniconv_v0.1_e12_s256x704_v200x200x12_imgaug/epoch_1.pth
# MASTER_PORT=29581 slurm_test $PARTITION 8 nuscenes/uniconv_v0.1_e12_s256x704_v200x200x12_bevaug test work_dirs/uniconv/exp/nuscenes/uniconv_v0.1_e12_s256x704_v200x200x12_bevaug/epoch_9.pth
# MASTER_PORT=29581 slurm_test $PARTITION 8 nuscenes/uniconv_v0.1_e12_s256x704_v200x200x12_ibaug test work_dirs/uniconv/exp/nuscenes/uniconv_v0.1_e12_s256x704_v200x200x12_ibaug/epoch_1.pth
# MASTER_PORT=29581 slurm_test $PARTITION 8 nuscenes/uniconv_seq_v0.1_e12_s256x704_v200x200x12_repeat test work_dirs/uniconv/exp/nuscenes/uniconv_seq_v0.1_e12_s256x704_v200x200x12_repeat/epoch_4.pth
# MASTER_PORT=29581 slurm_test $PARTITION 32 nuscenes/uniconv_v0.1_e48_s256x704_v200x200x6 test work_dirs/uniconv/exp/nuscenes/uniconv_v0.1_e48_s256x704_v200x200x6/epoch_2.pth
# MASTER_PORT=29581 slurm_test $PARTITION 32 nuscenes/uniconv_v0.1_e48_s256x704_v200x200x6_seq1.0 test work_dirs/uniconv/exp/nuscenes/uniconv_v0.1_e48_s256x704_v200x200x6_seq1.0/epoch_2.pth
# MASTER_PORT=29581 slurm_test $PARTITION 32 nuscenes/uniconv_v0.1_e48_s256x704_v200x200x6_ibaug_v2 test work_dirs/uniconv/exp/nuscenes/uniconv_v0.1_e48_s256x704_v200x200x6_ibaug_v2/epoch_4.pth
# MASTER_PORT=29581 slurm_test $PARTITION 32 nuscenes/uniconv_v0.1_e48_s256x704_v200x200x6_seq2.0 test work_dirs/uniconv/exp/nuscenes/uniconv_v0.1_e48_s256x704_v200x200x6_seq2.0/epoch_48.pth
# MASTER_PORT=29581 slurm_test $PARTITION 32 nuscenes/uniconv_v0.1_e48_s256x704_v200x200x6_seq2.0_ibaug_v2 test work_dirs/uniconv/exp/nuscenes/uniconv_v0.1_e48_s256x704_v200x200x6_seq2.0_ibaug_v2/epoch_48.pth
# MASTER_PORT=29581 slurm_test $PARTITION 32 nuscenes/uniconv_seg_v0.1_e48_s256x704_v200x200x6 test work_dirs/uniconv/exp/nuscenes/uniconv_seg_v0.1_e48_s256x704_v200x200x6/epoch_48.pth
# MASTER_PORT=29581 slurm_test $PARTITION 32 nuscenes/uniconv_seg_v0.1_e48_s256x704_v200x200x6_imgaug test work_dirs/uniconv/exp/nuscenes/uniconv_seg_v0.1_e48_s256x704_v200x200x6_imgaug/epoch_48.pth
# MASTER_PORT=29581 slurm_test $PARTITION 32 nuscenes/uniconv_v0.1_e48_s256x704_v200x200x6_seq3.0_ibaug_v2 test work_dirs/uniconv/exp/nuscenes/uniconv_v0.1_e48_s256x704_v200x200x6_seq3.0_ibaug_v2/epoch_48.pth
# MASTER_PORT=29581 slurm_test $PARTITION 8 nuscenes/uniconv_v0.1_e12_s256x704_v200x200x6_seq3.1_ibaug_v2_interval4_f4s135 test
# MASTER_PORT=29581 slurm_test $PARTITION 1 nuscenes/uniconv_v0.1_e48_s256x704_v200x200x6_seq3.1_ibaug_v2_interval1_f5s1357 test
# MASTER_PORT=29581 slurm_test $PARTITION 1 nuscenes/uniconv_v0.1_e48_s256x704_v200x200x6_seq2.0_ibaug_v2 test work_dirs/uniconv/exp/paper/uniconv_v0.1_e48_s256x704_v200x200x6_seq2.0_ibaug_v2/epoch_1.pth


#################### fullset-batch-test
# MASTER_PORT=29581 batch_slurm_test $PARTITION 16 nuscenes/uniconv_v0.1_e12_s256x704_v200x200x12
# MASTER_PORT=29581 batch_slurm_test $PARTITION 16 nuscenes/uniconv_v0.1_e12_s256x704_v200x200x12_imgaug
# MASTER_PORT=29581 batch_slurm_test $PARTITION 16 nuscenes/uniconv_v0.1_e12_s256x704_v200x200x12_bevaug
# MASTER_PORT=29581 batch_slurm_test $PARTITION 16 nuscenes/uniconv_v0.1_e12_s256x704_v200x200x12_ibaug
# MASTER_PORT=29581 batch_slurm_test $PARTITION 16 nuscenes/uniconv_v0.1_e24_s256x704_v200x200x12
# MASTER_PORT=29581 batch_slurm_test $PARTITION 16 nuscenes/uniconv_v0.1_e24_s256x704_v200x200x12_imgaug
# MASTER_PORT=29581 batch_slurm_test $PARTITION 16 nuscenes/uniconv_v0.1_e24_s256x704_v200x200x12_bevaug
# MASTER_PORT=29581 batch_slurm_test $PARTITION 16 nuscenes/uniconv_v0.1_e24_s256x704_v200x200x12_ibaug


#################### analysis
# slurm_benchmark $PARTITION nuscenes/uniconv_v0.1_e48_s256x704_v200x200x6 # 256x704, 6.6 img / s
# slurm_benchmark $PARTITION nuscenes/uniconv_v0.1_e48_s256x704_v200x200x12 # 256x704, 4.1 img / s
# slurm_benchmark $PARTITION nuscenes/uniconv_v0.1_e12_s544x960_v200x200x6 # 544x960
# slurm_benchmark $PARTITION nuscenes/uniconv_v0.1_e12_s544x960_v200x200x12 # 544x960, 3.3 img / s
# slurm_benchmark $PARTITION nuscenes/uniconv_v0.1_e12_s544x960_v200x200x6 # 544x960, 4.5 img / s

wait
