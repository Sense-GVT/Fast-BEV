#!/usr/bin/env bash
T=`date +%m%d%H%M`

MMDET3D=${MMDET3D:-/mnt/cache/huangbin1/workspace/m2bev}
SRUN_ARGS=${SRUN_ARGS:-"-s"}
RETRY=${RETRY:-1}
echo MMDET3D: $MMDET3D
echo SRUN_ARGS: $SRUN_ARGS
echo RETRY: $RETRY


function spring_slurm_train {
    PARTITION=$1
    GPUS=$2
    JOB_TYPE=$3
    EXPNAME=$4
    JOB_NAME=cla-spring_slurm_train-${5:-`basename $EXPNAME`}
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
    JOB_NAME=cla-slurm_train-${4:-`basename $EXPNAME`}
    GPUS_PER_NODE=$(($GPUS<8?$GPUS:8))

    echo slurm_train; sleep 0.5s
    MMDET3D=$MMDET3D \
    SRUN_ARGS=$SRUN_ARGS \
    GPUS=$GPUS GPUS_PER_NODE=$GPUS_PER_NODE \
    ./tools/slurm_train.sh \
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
    JOB_NAME=cla-test-${4:-`basename $EXPNAME`}
    RESUME=${5:-work_dirs/uniconv/exp/$EXPNAME/latest.pth}
    SCORE_THR=${6:-0.4}
    VIS_RATE=${7:-1}
    GPUS_PER_NODE=$(($GPUS<8?$GPUS:8))
    echo slurm_test; sleep 0.5s

    MMDET3D=$MMDET3D \
    SRUN_ARGS=$SRUN_ARGS \
    GPUS=$GPUS GPUS_PER_NODE=$GPUS_PER_NODE \
    ./tools/slurm_test.sh \
        $PARTITION \
        $JOB_NAME \
        configs/uniconv/exp/$EXPNAME.py \
        $RESUME \
        --out work_dirs/uniconv/exp/$EXPNAME/results/results.pkl

    PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
    srun -p $PARTITION --job-name=$JOB_NAME -n1 \
        python -u internal_code/eval_vis_target.py configs/uniconv/exp/$EXPNAME.py \
            --out work_dirs/uniconv/exp/$EXPNAME/results/results.pkl \
            --eval \
            --eval-options jsonfile_prefix=work_dirs/uniconv/exp/$EXPNAME/results \
            --score_thr $SCORE_THR \
    2>&1 | tee work_dirs/uniconv/exp/$EXPNAME/results/log.evaluate.$SCORE_THR.$T.txt

    # PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
    # srun -p $PARTITION --job-name=$JOB_NAME -n1 \
    #     python -u internal_code/eval_vis_target.py configs/uniconv/exp/$EXPNAME.py \
    #         --out work_dirs/uniconv/exp/$EXPNAME/results/results.pkl \
    #         --vis \
    #         --sample_rate ${VIS_RATE} \
    # 2>&1 | tee work_dirs/uniconv/exp/$EXPNAME/results/log.vis.$T.txt
}

function batch_slurm_test {
    PARTITION=$1
    GPUS=$2
    EXPNAME=$3
    JOB_NAME=${4:-`basename $EXPNAME`}
    SCORE_THR=${SCORE_THR:-0.4}
    VIS_RATE=${VIS_RATE:-1}
    GPUS_PER_NODE=$(($GPUS<8?$GPUS:8))
    mkdir -p work_dirs/uniconv/exp/$EXPNAME/batch
    mkdir -p work_dirs/uniconv/exp/$EXPNAME/test/$BASENAME

    # for RESUME in `ls work_dirs/uniconv/exp/$EXPNAME/batch/epoch*.pth`; do
    #     BASENAME=`basename $RESUME`
    #     SUB_JOB_NAME=cla-test-$JOB_NAME-$BASENAME
    #     echo slurm_test; sleep 0.5s
    #     echo RESUME: $RESUME
    #     echo SUB_JOB_NAME: $SUB_JOB_NAME
    #     MMDET3D=$MMDET3D \
    #     SRUN_ARGS=$SRUN_ARGS \
    #     GPUS=$GPUS GPUS_PER_NODE=$GPUS_PER_NODE QUOTATYPE=$QUOTATYPE \
    #     sh ./tools/slurm_test.sh \
    #         $PARTITION \
    #         $SUB_JOB_NAME \
    #         configs/uniconv/exp/$EXPNAME.py \
    #         $RESUME \
    #         --out work_dirs/uniconv/exp/$EXPNAME/test/$BASENAME/results.$BASENAME.pkl \
    #     &
    # done
    # wait

    # for RESUME in `ls work_dirs/uniconv/exp/$EXPNAME/batch/epoch*.pth`; do
    #     echo slurm_test; sleep 0.5s
    #     BASENAME=`basename $RESUME`
    #     OUT=work_dirs/uniconv/exp/$EXPNAME/test/$BASENAME/results.$BASENAME.pkl
    #     PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
    #     srun -p $PARTITION --job-name=$JOB_NAME -n1 \
    #         python -u internal_code/eval_vis_target.py configs/uniconv/exp/$EXPNAME.py \
    #             --out $OUT \
    #             --eval \
    #             --eval-options jsonfile_prefix=work_dirs/uniconv/exp/$EXPNAME/test/$BASENAME \
    #             --score_thr $SCORE_THR \
    #         2>&1 | tee work_dirs/uniconv/exp/$EXPNAME/test/$BASENAME/log.evaluate.$BASENAME.$SCORE_THR.$T.txt \
    #     &
    # done
    # wait

    # for RESUME in `ls work_dirs/uniconv/exp/$EXPNAME/batch/epoch*.pth`; do
    #     echo slurm_test; sleep 0.5s
    #     BASENAME=`basename $RESUME`
    #     OUT=work_dirs/uniconv/exp/$EXPNAME/test/$BASENAME/results.$BASENAME.pkl

    #     PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
    #     srun -p $PARTITION --job-name=$JOB_NAME -n1 \
    #         python -u internal_code/eval_vis_target.py configs/uniconv/exp/$EXPNAME.py \
    #             --out $OUT \
    #             --vis \
    #             --sample_rate ${VIS_RATE} \
    #     2>&1 | tee work_dirs/uniconv/exp/$EXPNAME/results/log.vis.$T.txt \
    #     &
    # done
    # wait
}

function slurm_analysis {
    PARTITION=$1
    EXPNAME=$2
    JOB_NAME=cla-analysis-${3:-`basename $EXPNAME`}

    echo slurm_analysis; sleep 0.5s
    MMDET3D=$MMDET3D \
    SRUN_ARGS=$SRUN_ARGS \
    GPUS=1 GPUS_PER_NODE=1 \
    ./tools/slurm_analysis.sh \
        $PARTITION \
        $JOB_NAME \
        configs/uniconv/exp/$EXPNAME.py \
        work_dirs/uniconv/exp/$EXPNAME/latest.pth \
    2>&1 | tee work_dirs/uniconv/exp/$EXPNAME/log.benchmark.$T.txt
}

PARTITION=$1
QUOTATYPE=$2

#################### ipdb
# MASTER_PORT=29580 slurm_train $PARTITION 1 cla/uniconv_cla_p0.0.0
# MASTER_PORT=29581 slurm_train $PARTITION 1 cla/uniconv_cla_t0.3.0_s540x960_v425x250x6_seq3.1_e12_interval10.0
# MASTER_PORT=29582 slurm_train $PARTITION 1 cla/uniconv_cla_t0.3.0_s540x960_v425x250x6_seq3.1_e12_interval10.0_noseq
# MASTER_PORT=29583 QUOTATYPE=$QUOTATYPE slurm_test $PARTITION 1 cla/uniconv_cla_v0.2.2_z1.1_s540x960_v425x250x6_ibaug_e40_interval1.0_lastft5

# spring_slurm_train $PARTITION 64 $QUOTATYPE cla/uniconv_cla_v0.0.0
# spring_slurm_train $PARTITION 64 $QUOTATYPE cla/uniconv_cla_v0.0.0
# spring_slurm_train $PARTITION 64 $QUOTATYPE cla/uniconv_cla_v0.0.0
# spring_slurm_train $PARTITION 64 $QUOTATYPE cla/uniconv_cla_v0.0.0

#################### train
# spring_slurm_train $PARTITION 64 $QUOTATYPE cla/uniconv_cla_v0.2.2_s540x960_v425x250x6_e40_interval1.0
# spring_slurm_train $PARTITION 64 $QUOTATYPE cla/uniconv_cla_v0.2.2_z1_s540x960_v425x250x6_ibaug_e40_interval1.0
# spring_slurm_train $PARTITION 64 $QUOTATYPE cla/uniconv_cla_v0.2.2_z1.1_s540x960_v425x250x6_ibaug_e40_interval1.0_lastft5
# spring_slurm_train $PARTITION 64 $QUOTATYPE cla/uniconv_cla_v0.3.0_s544x960_v425x250x6_ibaug_e40_interval1.0
# spring_slurm_train $PARTITION 64 $QUOTATYPE cla/uniconv_cla_v0.3.0_z1_s544x960_v425x250x6_ibaug_e40_interval1.0_lastft5
# spring_slurm_train $PARTITION 64 $QUOTATYPE cla/uniconv_cla_v0.3.1_s544x960_v425x250x6_ibaug_e40_interval1.0
# spring_slurm_train $PARTITION 64 $QUOTATYPE cla/uniconv_cla_v0.3.1_z1_s544x960_v425x250x6_ibaug_e40_interval1.0_lastft5
# spring_slurm_train $PARTITION 64 $QUOTATYPE cla/uniconv_cla_v0.4.1_r34_s544x960_v340x200x4_ibaug_e40_interval1.0
# spring_slurm_train $PARTITION 64 $QUOTATYPE cla/uniconv_cla_v0.4.1_z1_r34_s544x960_v340x200x4_ibaug_e40_interval1.0_lastft5
# spring_slurm_train $PARTITION 64 $QUOTATYPE cla/uniconv_cla_v0.5.1_r18_s544x960_v200x100x3_ibaug_e40_interval1.0
# spring_slurm_train $PARTITION 64 $QUOTATYPE cla/uniconv_cla_v0.5.1_z1_r18_s544x960_v200x100x3_ibaug_e40_interval1.0_lastft5
# spring_slurm_train $PARTITION 64 $QUOTATYPE cla/uniconv_cla_v0.6.1_r18_s544x960_v200x100x3_ibaug_e20_interval3.0
# spring_slurm_train $PARTITION 64 $QUOTATYPE cla/uniconv_cla_v0.7.1_r50_s544x960_v340x80x4_ibaug_e40_interval1.0

#################### test
# QUOTATYPE=$QUOTATYPE slurm_test $PARTITION 32 cla/uniconv_cla_v0.2.2_s540x960_v425x250x6_e40_interval1.0 &
# QUOTATYPE=$QUOTATYPE slurm_test $PARTITION 32 cla/uniconv_cla_v0.2.2_z1_s540x960_v425x250x6_ibaug_e40_interval1.0 &
# QUOTATYPE=$QUOTATYPE slurm_test $PARTITION 32 cla/uniconv_cla_v0.2.2_z1.1_s540x960_v425x250x6_ibaug_e40_interval1.0_lastft5
# QUOTATYPE=$QUOTATYPE slurm_test $PARTITION 64 cla/uniconv_cla_v0.2.2_z1.2_s544x960_v425x250x6_ibaug_e40_interval1.0_lastft5
# QUOTATYPE=$QUOTATYPE slurm_test $PARTITION 64 cla/uniconv_cla_v0.3.0_s544x960_v425x250x6_ibaug_e40_interval1.0
# QUOTATYPE=$QUOTATYPE slurm_test $PARTITION 64 cla/uniconv_cla_v0.3.0_z1_s544x960_v425x250x6_ibaug_e40_interval1.0_lastft5
# QUOTATYPE=$QUOTATYPE slurm_test $PARTITION 64 cla/uniconv_cla_v0.3.1_z1_s544x960_v425x250x6_ibaug_e40_interval1.0_lastft5
# QUOTATYPE=$QUOTATYPE slurm_test $PARTITION 64 cla/uniconv_cla_v0.4.1_z1_r34_s544x960_v340x200x4_ibaug_e40_interval1.0_lastft5
# QUOTATYPE=$QUOTATYPE slurm_test $PARTITION 64 cla/uniconv_cla_v0.5.1_z1_r18_s544x960_v200x100x3_ibaug_e40_interval1.0_lastft5
# QUOTATYPE=$QUOTATYPE slurm_test $PARTITION 32 cla/uniconv_cla_v0.6.1_r18_s544x960_v200x100x3_ibaug_e20_interval3.0

#################### batch-test
# batch_slurm_test $PARTITION 16 cla/uniconv_cla_v0.2.2_s540x960_v425x250x6_e40_interval1.0
# batch_slurm_test $PARTITION 16 cla/uniconv_cla_v0.2.2_z1.1_s540x960_v425x250x6_ibaug_e40_interval1.0_lastft5 thre 0.0 &
# batch_slurm_test $PARTITION 16 cla/uniconv_cla_v0.2.2_z1.1_s540x960_v425x250x6_ibaug_e40_interval1.0_lastft5 thre 0.1 &
# batch_slurm_test $PARTITION 16 cla/uniconv_cla_v0.2.2_z1.1_s540x960_v425x250x6_ibaug_e40_interval1.0_lastft5 thre 0.2 &
# batch_slurm_test $PARTITION 16 cla/uniconv_cla_v0.2.2_z1.1_s540x960_v425x250x6_ibaug_e40_interval1.0_lastft5 thre 0.3 &
# batch_slurm_test $PARTITION 16 cla/uniconv_cla_v0.2.2_z1.1_s540x960_v425x250x6_ibaug_e40_interval1.0_lastft5 thre 0.4 &
# batch_slurm_test $PARTITION 16 cla/uniconv_cla_v0.2.2_z1.1_s540x960_v425x250x6_ibaug_e40_interval1.0_lastft5 thre 0.5 &
# batch_slurm_test $PARTITION 16 cla/uniconv_cla_v0.2.2_z1.1_s540x960_v425x250x6_ibaug_e40_interval1.0_lastft5 thre 0.6 &
# batch_slurm_test $PARTITION 16 cla/uniconv_cla_v0.2.2_z1.1_s540x960_v425x250x6_ibaug_e40_interval1.0_lastft5 thre 0.7 &
# batch_slurm_test $PARTITION 16 cla/uniconv_cla_v0.2.2_z1.1_s540x960_v425x250x6_ibaug_e40_interval1.0_lastft5 thre 0.8 &
# batch_slurm_test $PARTITION 16 cla/uniconv_cla_v0.2.2_z1.1_s540x960_v425x250x6_ibaug_e40_interval1.0_lastft5 thre 0.9 &
# batch_slurm_test $PARTITION 64 cla/uniconv_cla_v0.3.0_s544x960_v425x250x6_ibaug_e40_interval1.0

#################### analysis
# slurm_analysis $PARTITION cla/uniconv_cla_t0.2.2_s540x960_v425x250x6_e12_interval10.0


wait
