#!/usr/bin/env bash
T=`date +%m%d%H%M`

MMDET3D=${MMDET3D:-/mnt/lustre/chenzeren/m2bev}
SRUN_ARGS=${SRUN_ARGS:-""}
echo MMDET3D: $MMDET3D
echo SRUN_ARGS: $SRUN_ARGS

function slurm_test {
    PARTITION=$1
    GPUS=$2
    CFG=$3
    RESUME=$4
    OUT=$5
    SCORE_THR=${6:-0.4}
    JOB_NAME=cla-deploy-${`basename $CFG`}

    GPUS_PER_NODE=$(($GPUS<8?$GPUS:8))
    echo slurm_test; sleep 0.5s

    # infer
    MMDET3D=$MMDET3D \
    SRUN_ARGS=$SRUN_ARGS \
    GPUS=$GPUS GPUS_PER_NODE=$GPUS_PER_NODE \
    ./tools/slurm_test.sh $PARTITION $JOB_NAME $CFG $RESUME --out $OUT

    # eval
    PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
    srun -p $PARTITION --job-name=$JOB_NAME -n1 \
        python -u internal_code/eval_vis.py $CFG \
            --out $OUT \
            --eval \
            --score_thr $SCORE_THR

    # vis
    PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
    srun -p $PARTITION --job-name=$JOB_NAME -n1 \
        python -u internal_code/eval_vis.py $CFG \
            --out $OUT \
            --vis \
            --sample_rate 1
}

function slurm_deploy {
    NAME=$1
    SIZE_2D=$2
    SIZE_3D=$3
    CFG=$4
    RESUME=$5

    # PYTHONPATH="$(dirname $0)/..":$PYTHONPATH DEPLOY=True \
    #     srun -p AD_GVT -N1 --gres=gpu:1 python tools/deploy.py $CFG $RESUME \
    #         --type 2d --name ${NAME}_2D \
    #         --replace-bn \
    #         --size $SIZE_2D
    PYTHONPATH="$(dirname $0)/..":$PYTHONPATH DEPLOY=True \
        srun -p AD_GVT -N1 --gres=gpu:1 python tools/deploy.py $CFG $RESUME \
            --type 3d --name ${NAME}_3D \
            --replace-bn \
            --size $SIZE_3D
}

slurm_deploy FASTBEV_M0_R18_FAHEAD_V1 \
    6,3,256,704 \
    1,200,200,4,128 \
    configs/uniconv/exp/paper_v2/uniconv_v0.5_m0_r18_fahead_e24_s256x704_v200x200x4_c192_d2_f2.py \
    work_dirs/uniconv/exp/paper_v2/uniconv_v0.5_m0_r18_fahead_e24_s256x704_v200x200x4_c192_d2_f2/epoch_24.pth

# wait