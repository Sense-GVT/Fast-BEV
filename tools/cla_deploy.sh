#!/usr/bin/env bash
T=`date +%m%d%H%M`

MMDET3D=${MMDET3D:-/mnt/cache/huangbin1/workspace/m2bev}
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

    PYTHONPATH="$(dirname $0)/..":$PYTHONPATH DEPLOY=True \
        srun -p AD_GVT -N1 --gres=gpu:1 python tools/deploy.py $CFG $RESUME \
            --type 2d --name ${NAME}_2D \
            --replace-bn \
            --size $SIZE_2D
    PYTHONPATH="$(dirname $0)/..":$PYTHONPATH DEPLOY=True \
        srun -p AD_GVT -N1 --gres=gpu:1 python tools/deploy.py $CFG $RESUME \
            --type 3d --name ${NAME}_3D \
            --replace-bn \
            --size $SIZE_3D
}


# slurm_deploy KM_UniConvNV_STARDARD_R50_f1_V1 \
#     6,3,256,704 \
#     6,200,200,6,64 \
#     configs/uniconv/exp/paper/uniconv_v0.3_m1_r50_s256x704_v200x200x6_c256_d6_f1_v1.py \
#     ./pretrained_models/dummy.pth
# slurm_deploy KM_UniConvNV_STARDARD_R50_f4_V1 \
#     6,3,256,704 \
#     6,200,200,6,256 \
#     configs/uniconv/exp/paper/uniconv_v0.3_m2_r50_s256x704_v200x200x6_c256_d6_f4_v1.py \
#     ./pretrained_models/dummy.pth
# slurm_deploy KM_UniConvNV_STARDARD_R50_f4_V3 \
#     6,3,256,704 \
#     1,200,200,6,256 \
#     configs/uniconv/exp/paper/uniconv_v0.3_m0_r50_s256x704_v200x200x6_c256_d6.py \
#     ./pretrained_models/dummy.pth


# test
# slurm_test AD_GVT 1 \
#     configs/uniconv/deploy/uniconv_cla_v0.2.2_z1.2_s544x960_v425x250x6_ibaug_e40_interval1.0_lastft5.py \
#     work_dirs/uniconv/exp/cla/uniconv_cla_v0.2.2_z1.2_s544x960_v425x250x6_ibaug_e40_interval1.0_lastft5/epoch_40.pth \
#     work_dirs/uniconv/deploy/results.pkl

# deploy
# slurm_deploy \
#     configs/uniconv/deploy/uniconv_cla_v0.2.2_z1.2_s544x960_v425x250x6_ibaug_e40_interval1.0_lastft5.py \
#     work_dirs/uniconv/exp/cla/uniconv_cla_v0.2.2_z1.2_s544x960_v425x250x6_ibaug_e40_interval1.0_lastft5/epoch_40.pth

# slurm_deploy \
#     configs/uniconv/deploy/uniconv_cla_v0.4.1_z1_r34_s544x960_v340x200x4_ibaug_e40_interval1.0_lastft5.py \
#     work_dirs/uniconv/exp/cla/uniconv_cla_v0.4.1_z1_r34_s544x960_v340x200x4_ibaug_e40_interval1.0_lastft5/latest.pth

# slurm_deploy \
#     configs/uniconv/deploy/uniconv_cla_v0.4.1_z1_r34_s544x960_v200x200x4_ibaug_e40_interval1.0_lastft5.py \
#     work_dirs/uniconv/exp/cla/uniconv_cla_v0.4.1_z1_r34_s544x960_v340x200x4_ibaug_e40_interval1.0_lastft5/latest.pth

# slurm_deploy KM_UniConvNV_R18 \
#     6,3,544,960 \
#     6,200,100,3,64 \
#     configs/uniconv/deploy/uniconv_cla_v0.5.1_z1_r18_s544x960_v200x100x3_ibaug_e40_interval1.0_lastft5.py \
#     work_dirs/uniconv/exp/cla/uniconv_cla_v0.4.1_z1_r34_s544x960_v340x200x4_ibaug_e40_interval1.0_lastft5/latest.pth \

# slurm_deploy KM_UniConvNV_3D_Latency_f1 \
#     6,3,544,960 \
#     6,200,100,3,64 \
#     configs/uniconv/deploy/uniconv_cla_v0.0.1_z1_r18_s544x960_v200x100x3_ibaug_e40_interval1.0_lastft5.py \
#     work_dirs/uniconv/exp/cla/uniconv_cla_v0.4.1_z1_r34_s544x960_v340x200x4_ibaug_e40_interval1.0_lastft5/latest.pth
# slurm_deploy KM_UniConvNV_3D_Latency_f4 \
#     6,3,544,960 \
#     6,200,100,3,256 \
#     configs/uniconv/deploy/uniconv_cla_v0.0.1_z1_r18_s544x960_v200x100x3_ibaug_e40_interval1.0_lastft5.py \
#     work_dirs/uniconv/exp/cla/uniconv_cla_v0.4.1_z1_r34_s544x960_v340x200x4_ibaug_e40_interval1.0_lastft5/latest.pth
# slurm_deploy KM_UniConvNV_3D_Latency_fast_f4 \
#     6,3,544,960 \
#     6,200,100,3,256 \
#     configs/uniconv/deploy/uniconv_cla_v0.0.1_z1_r18_s544x960_v200x100x3_ibaug_e40_interval1.0_lastft5.py \
#     work_dirs/uniconv/exp/cla/uniconv_cla_v0.4.1_z1_r34_s544x960_v340x200x4_ibaug_e40_interval1.0_lastft5/latest.pth

# DEPLOY_DEBUG=True slurm_deploy KM_UniConvNV_3D_Latency_f1_permute \
#     6,3,544,960 \
#     6,200,100,3,64 \
#     configs/uniconv/deploy/uniconv_cla_v0.0.1_z1_r18_s544x960_v200x100x3_ibaug_e40_interval1.0_lastft5.py \
#     work_dirs/uniconv/exp/cla/uniconv_cla_v0.4.1_z1_r34_s544x960_v340x200x4_ibaug_e40_interval1.0_lastft5/latest.pth
# DEPLOY_DEBUG=True slurm_deploy KM_UniConvNV_3D_Latency_fast_f4_permute \
#     6,3,544,960 \
#     6,200,100,3,256 \
#     configs/uniconv/deploy/uniconv_cla_v0.0.1_z1_r18_s544x960_v200x100x3_ibaug_e40_interval1.0_lastft5.py \
#     work_dirs/uniconv/exp/cla/uniconv_cla_v0.4.1_z1_r34_s544x960_v340x200x4_ibaug_e40_interval1.0_lastft5/latest.pth

# DEPLOY_DEBUG=True slurm_deploy KM_UniConvNV_3D_Latency_fast_f1_permute_sum \
#     6,3,544,960 \
#     6,200,100,3,64 \
#     configs/uniconv/deploy/uniconv_cla_v0.0.1_z1_r18_s544x960_v200x100x3_ibaug_e40_interval1.0_lastft5.py \
#     work_dirs/uniconv/exp/cla/uniconv_cla_v0.4.1_z1_r34_s544x960_v340x200x4_ibaug_e40_interval1.0_lastft5/latest.pth
# DEPLOY_DEBUG=True slurm_deploy KM_UniConvNV_3D_Latency_fast_f4_permute_sum \
#     6,3,544,960 \
#     6,200,100,3,256 \
#     configs/uniconv/deploy/uniconv_cla_v0.0.1_z1_r18_s544x960_v200x100x3_ibaug_e40_interval1.0_lastft5.py \
#     work_dirs/uniconv/exp/cla/uniconv_cla_v0.4.1_z1_r34_s544x960_v340x200x4_ibaug_e40_interval1.0_lastft5/latest.pth
# DEPLOY_DEBUG=True slurm_deploy KM_UniConvNV_3D_Latency_fast_f4_permute_nosum \
#     6,3,544,960 \
#     1,200,100,3,256 \
#     configs/uniconv/deploy/uniconv_cla_v0.0.1_z1_r18_s544x960_v200x100x3_ibaug_e40_interval1.0_lastft5.py \
#     work_dirs/uniconv/exp/cla/uniconv_cla_v0.4.1_z1_r34_s544x960_v340x200x4_ibaug_e40_interval1.0_lastft5/latest.pth

# DEPLOY_DEBUG=True slurm_deploy K_MUniConvNV_3D_Latency_fast_f4_permute_r \
#     6,3,544,960 \
#     6,200,100,3,256 \
#     configs/uniconv/deploy/uniconv_cla_v0.0.1_z1_r18_s544x960_v200x100x3_ibaug_e40_interval1.0_lastft5.py \
#     work_dirs/uniconv/exp/cla/uniconv_cla_v0.4.1_z1_r34_s544x960_v340x200x4_ibaug_e40_interval1.0_lastft5/latest.pth
# DEPLOY_DEBUG=True slurm_deploy KM_UniConvNV_3D_Latency_fast_f4_permute_rp \
#     6,3,544,960 \
#     6,200,100,3,256 \
#     configs/uniconv/deploy/uniconv_cla_v0.0.1_z1_r18_s544x960_v200x100x3_ibaug_e40_interval1.0_lastft5.py \
#     work_dirs/uniconv/exp/cla/uniconv_cla_v0.4.1_z1_r34_s544x960_v340x200x4_ibaug_e40_interval1.0_lastft5/latest.pth

# DEPLOY_DEBUG=True slurm_deploy KM_UniConvNV_3D_Latency_fast_f4_lastsum_fix \
#     1,3,544,960 \
#     6,200,100,3,256 \
#     configs/uniconv/deploy/uniconv_cla_v0.0.1_z1_r18_s544x960_v200x100x3_ibaug_e40_interval1.0_lastft5.py \
#     work_dirs/uniconv/exp/cla/uniconv_cla_v0.4.1_z1_r34_s544x960_v340x200x4_ibaug_e40_interval1.0_lastft5/latest.pth

# DEPLOY_DEBUG=True slurm_deploy KM_UniConvNV_3D_Latency_fast_f1_lastsum_fix \
#     1,3,544,960 \
#     1,200,100,3,256 \
#     configs/uniconv/deploy/uniconv_cla_v0.0.1_z1_r18_s544x960_v200x100x3_ibaug_e40_interval1.0_lastft5.py \
#     work_dirs/uniconv/exp/cla/uniconv_cla_v0.4.1_z1_r34_s544x960_v340x200x4_ibaug_e40_interval1.0_lastft5/latest.pth

# slurm_deploy KM_UniConvNV_3D_Latency_f1 \
#     1,3,544,960 \
#     6,200,100,3,64 \
#     configs/uniconv/deploy/uniconv_cla_v0.0.1_z1_r18_s544x960_v200x100x3_ibaug_e40_interval1.0_lastft5.py \
#     work_dirs/uniconv/exp/cla/uniconv_cla_v0.4.1_z1_r34_s544x960_v340x200x4_ibaug_e40_interval1.0_lastft5/latest.pth

# slurm_deploy KM_UniConvNV_3D_Latency_f4 \
#     1,3,544,960 \
#     6,200,100,3,256 \
#     configs/uniconv/deploy/uniconv_cla_v0.0.1_z1_r18_s544x960_v200x100x3_ibaug_e40_interval1.0_lastft5.py \
#     work_dirs/uniconv/exp/cla/uniconv_cla_v0.4.1_z1_r34_s544x960_v340x200x4_ibaug_e40_interval1.0_lastft5/latest.pth

# slurm_deploy KM_UniConvNV_3D_Latency_fuse_f4 \
#     1,3,544,960 \
#     6,200,100,3,256 \
#     configs/uniconv/deploy/uniconv_cla_v0.0.1_z1_r18_s544x960_v200x100x3_ibaug_e40_interval1.0_lastft5.py \
#     work_dirs/uniconv/exp/cla/uniconv_cla_v0.4.1_z1_r34_s544x960_v340x200x4_ibaug_e40_interval1.0_lastft5/latest.pth

# slurm_deploy KM_UniConvNV_3D_Latency_fast_fuse_f4 \
#     1,3,544,960 \
#     6,200,100,3,256 \
#     configs/uniconv/deploy/uniconv_cla_v0.0.1_z1_r18_s544x960_v200x100x3_ibaug_e40_interval1.0_lastft5.py \
#     work_dirs/uniconv/exp/cla/uniconv_cla_v0.4.1_z1_r34_s544x960_v340x200x4_ibaug_e40_interval1.0_lastft5/latest.pth

# slurm_deploy KM_UniConvNV_3D_Latency_fast_fuse_f2 \
#     1,3,544,960 \
#     6,200,100,3,128 \
#     configs/uniconv/deploy/uniconv_cla_v0.0.1_z1_r18_s544x960_v200x100x3_ibaug_e40_interval1.0_lastft5.py \
#     work_dirs/uniconv/exp/cla/uniconv_cla_v0.4.1_z1_r34_s544x960_v340x200x4_ibaug_e40_interval1.0_lastft5/latest.pth


# DEPLOY_DEBUG=True slurm_deploy KM_UniConvNV_3D_Latency_neck_f1 \
#     1,3,544,960 \
#     1,200,200,4,64 \
#     configs/uniconv/deploy/uniconv_cla_v0.0.1_z1_r18_s544x960_v200x100x3_ibaug_e40_interval1.0_lastft5.py \
#     work_dirs/uniconv/exp/cla/uniconv_cla_v0.4.1_z1_r34_s544x960_v340x200x4_ibaug_e40_interval1.0_lastft5/latest.pth

# DEPLOY_DEBUG=True slurm_deploy KM_UniConvNV_3D_Latency_neck_f4 \
#     1,3,544,960 \
#     1,200,200,4,256 \
#     configs/uniconv/deploy/uniconv_cla_v0.0.1_z1_r18_s544x960_v200x100x3_ibaug_e40_interval1.0_lastft5.py \
#     work_dirs/uniconv/exp/cla/uniconv_cla_v0.4.1_z1_r34_s544x960_v340x200x4_ibaug_e40_interval1.0_lastft5/latest.pth

# DEPLOY_DEBUG=True slurm_deploy KM_UniConvNV_3D_Latency_neck_f4_permute_inchannel \
#     1,3,544,960 \
#     1,200,200,4,256 \
#     configs/uniconv/deploy/uniconv_cla_v0.0.1_z1_r18_s544x960_v200x100x3_ibaug_e40_interval1.0_lastft5.py \
#     work_dirs/uniconv/exp/cla/uniconv_cla_v0.4.1_z1_r34_s544x960_v340x200x4_ibaug_e40_interval1.0_lastft5/latest.pth

# DEPLOY_DEBUG=True slurm_deploy KM_UniConvNV_3D_Latency_neck_f4_permute_inchannel \
#     1,3,544,960 \
#     1,200,200,4,256 \
#     configs/uniconv/deploy/uniconv_cla_v0.0.1_z1_r18_s544x960_v200x100x3_ibaug_e40_interval1.0_lastft5.py \
#     work_dirs/uniconv/exp/cla/uniconv_cla_v0.4.1_z1_r34_s544x960_v340x200x4_ibaug_e40_interval1.0_lastft5/latest.pth

# DEPLOY_DEBUG=True slurm_deploy KM_UniConvNV_3D_Latency_neck_f4_permute_fuse \
#     1,3,544,960 \
#     1,200,200,4,256 \
#     configs/uniconv/deploy/uniconv_cla_v0.0.1_z1_r18_s544x960_v200x100x3_ibaug_e40_interval1.0_lastft5.py \
#     work_dirs/uniconv/exp/cla/uniconv_cla_v0.4.1_z1_r34_s544x960_v340x200x4_ibaug_e40_interval1.0_lastft5/latest.pth

# DEPLOY_DEBUG=True slurm_deploy KM_UniConvNV_3D_Latency_neck_fuse_f4 \
#     1,3,544,960 \
#     1,200,200,4,256 \
#     configs/uniconv/deploy/uniconv_cla_v0.0.1_z1_r18_s544x960_v200x100x3_ibaug_e40_interval1.0_lastft5.py \
#     work_dirs/uniconv/exp/cla/uniconv_cla_v0.4.1_z1_r34_s544x960_v340x200x4_ibaug_e40_interval1.0_lastft5/latest.pth

# slurm_deploy KM_UniConvNV_M0_baseline \
#     6,3,544,960 \
#     6,200,200,4,256 \
#     configs/uniconv/deploy/uniconv_v0.2_m0_r18_s256x704_v200x200x4_c192_d2.py \
#     work_dirs/uniconv/exp/cla/uniconv_cla_v0.4.1_z1_r34_s544x960_v340x200x4_ibaug_e40_interval1.0_lastft5/latest.pth
# slurm_deploy KM_UniConvNV_M0_f2_fast \
#     6,3,544,960 \
#     6,200,200,4,128 \
#     configs/uniconv/deploy/uniconv_v0.2_m0_r18_s256x704_v200x200x4_c192_d2.py \
#     work_dirs/uniconv/exp/cla/uniconv_cla_v0.4.1_z1_r34_s544x960_v340x200x4_ibaug_e40_interval1.0_lastft5/latest.pth

##### m-series
## m0: r18_s256x704_v200x200x4_c192_d2
# slurm_deploy KM_UniConvNV_M0_fast \
#     6,3,544,960 \
#     6,200,200,4,256 \
#     configs/uniconv/deploy/uniconv_v0.2_m0_r18_s256x704_v200x200x4_c192_d2.py \
#     work_dirs/uniconv/exp/cla/uniconv_cla_v0.4.1_z1_r34_s544x960_v340x200x4_ibaug_e40_interval1.0_lastft5/latest.pth
## m1: r34_s256x704_v200x200x4_c224_d4
# slurm_deploy KM_UniConvNV_M1_fast \
#     6,3,544,960 \
#     6,200,200,4,256 \
#     configs/uniconv/deploy/uniconv_v0.2_m1_r34_s256x704_v200x200x4_c224_d4.py \
#     work_dirs/uniconv/exp/cla/uniconv_cla_v0.4.1_z1_r34_s544x960_v340x200x4_ibaug_e40_interval1.0_lastft5/latest.pth
## m2: r34_s320x880_v250x250x6_c224_d4
# slurm_deploy KM_UniConvNV_M2_fast \
#     6,3,544,960 \
#     6,250,250,6,256 \
#     configs/uniconv/deploy/uniconv_v0.2_m2_r34_s320x880_v250x250x6_c224_d4.py \
#     work_dirs/uniconv/exp/cla/uniconv_cla_v0.4.1_z1_r34_s544x960_v340x200x4_ibaug_e40_interval1.0_lastft5/latest.pth
## m3: r50_s320x880_v250x250x6_c256_d6
# slurm_deploy KM_UniConvNV_M3_fast \
#     6,3,544,960 \
#     6,250,250,6,256 \
#     configs/uniconv/deploy/uniconv_v0.2_m3_r50_s320x880_v250x250x6_c256_d6.py \
#     work_dirs/uniconv/exp/cla/uniconv_cla_v0.4.1_z1_r34_s544x960_v340x200x4_ibaug_e40_interval1.0_lastft5/latest.pth
## m4: r50_s384x1056_v300x300x6_c256_d6
# slurm_deploy KM_UniConvNV_M4_fast \
#     6,3,544,960 \
#     6,300,300,6,256 \
#     configs/uniconv/deploy/uniconv_v0.2_m4_r50_s384x1056_v300x300x6_c256_d6.py \
#     work_dirs/uniconv/exp/cla/uniconv_cla_v0.4.1_z1_r34_s544x960_v340x200x4_ibaug_e40_interval1.0_lastft5/latest.pth

# m0: r18_s256x704_v200x200x4_c192_d2
# slurm_deploy KM_UniConvNV_M0_fast_b1 \
#     6,3,544,960 \
#     1,200,200,4,256 \
#     configs/uniconv/deploy/uniconv_v0.2_m0_r18_s256x704_v200x200x4_c192_d2.py \
#     work_dirs/uniconv/exp/cla/uniconv_cla_v0.4.1_z1_r34_s544x960_v340x200x4_ibaug_e40_interval1.0_lastft5/latest.pth
# # m1: r34_s256x704_v200x200x4_c224_d4
# slurm_deploy KM_UniConvNV_M1_fast_b1 \
#     6,3,544,960 \
#     1,200,200,4,256 \
#     configs/uniconv/deploy/uniconv_v0.2_m1_r34_s256x704_v200x200x4_c224_d4.py \
#     work_dirs/uniconv/exp/cla/uniconv_cla_v0.4.1_z1_r34_s544x960_v340x200x4_ibaug_e40_interval1.0_lastft5/latest.pth
# # m2: r34_s320x880_v250x250x6_c224_d4
# slurm_deploy KM_UniConvNV_M2_fast_b1 \
#     6,3,544,960 \
#     1,250,250,6,256 \
#     configs/uniconv/deploy/uniconv_v0.2_m2_r34_s320x880_v250x250x6_c224_d4.py \
#     work_dirs/uniconv/exp/cla/uniconv_cla_v0.4.1_z1_r34_s544x960_v340x200x4_ibaug_e40_interval1.0_lastft5/latest.pth
# m3: r50_s320x880_v250x250x6_c256_d6
# slurm_deploy KM_UniConvNV_M3_fast_b1 \
#     6,3,544,960 \
#     1,250,250,6,256 \
#     configs/uniconv/deploy/uniconv_v0.2_m3_r50_s320x880_v250x250x6_c256_d6.py \
#     work_dirs/uniconv/exp/cla/uniconv_cla_v0.4.1_z1_r34_s544x960_v340x200x4_ibaug_e40_interval1.0_lastft5/latest.pth
# # m4: r50_s384x1056_v300x300x6_c256_d6
# slurm_deploy KM_UniConvNV_M4_fast_b1 \
#     6,3,544,960 \
#     1,300,300,6,256 \
#     configs/uniconv/deploy/uniconv_v0.2_m4_r50_s384x1056_v300x300x6_c256_d6.py \
#     work_dirs/uniconv/exp/cla/uniconv_cla_v0.4.1_z1_r34_s544x960_v340x200x4_ibaug_e40_interval1.0_lastft5/latest.pth

# # m0: r18_s256x704_v200x200x4_c192_d2
# slurm_deploy KM_UniConvNV_M0_fast_f2 \
#     6,3,544,960 \
#     6,200,200,4,128 \
#     configs/uniconv/deploy/uniconv_v0.2_m0_r18_s256x704_v200x200x4_c192_d2.py \
#     work_dirs/uniconv/exp/cla/uniconv_cla_v0.4.1_z1_r34_s544x960_v340x200x4_ibaug_e40_interval1.0_lastft5/latest.pth
# # m1: r34_s256x704_v200x200x4_c224_d4
# slurm_deploy KM_UniConvNV_M1_fast_f2 \
#     6,3,544,960 \
#     6,200,200,4,128 \
#     configs/uniconv/deploy/uniconv_v0.2_m1_r34_s256x704_v200x200x4_c224_d4.py \
#     work_dirs/uniconv/exp/cla/uniconv_cla_v0.4.1_z1_r34_s544x960_v340x200x4_ibaug_e40_interval1.0_lastft5/latest.pth

# # m0: r18_s256x704_v200x200x4_c192_d2
# slurm_deploy KM_UniConvNV_M0_fast_f1 \
#     6,3,544,960 \
#     6,200,200,4,64 \
#     configs/uniconv/deploy/uniconv_v0.2_m0_r18_s256x704_v200x200x4_c192_d2.py \
#     work_dirs/uniconv/exp/cla/uniconv_cla_v0.4.1_z1_r34_s544x960_v340x200x4_ibaug_e40_interval1.0_lastft5/latest.pth
# # m1: r34_s256x704_v200x200x4_c224_d4
# slurm_deploy KM_UniConvNV_M1_fast_f1 \
#     6,3,544,960 \
#     6,200,200,4,64 \
#     configs/uniconv/deploy/uniconv_v0.2_m1_r34_s256x704_v200x200x4_c224_d4.py \
#     work_dirs/uniconv/exp/cla/uniconv_cla_v0.4.1_z1_r34_s544x960_v340x200x4_ibaug_e40_interval1.0_lastft5/latest.pth

##### update-m-series
# m0: uniconv_v0.2_m0_r34_s256x704_v200x200x4_c192_d2_f2
# slurm_deploy KM_UniConvNV_M0_PLUS_fast \
#     6,3,256,704 \
#     6,200,200,4,128 \
#     configs/uniconv/deploy/uniconv_v0.2_m0_r34_s256x704_v200x200x4_c192_d2_f2.py \
#     work_dirs/uniconv/exp/cla/uniconv_cla_v0.4.1_z1_r34_s544x960_v340x200x4_ibaug_e40_interval1.0_lastft5/latest.pth

## m1: uniconv_v0.2_m1_r34_s320x880_v200x200x4_c192_d2_f2
# slurm_deploy KM_UniConvNV_M1_PLUS_fast \
#     6,3,320,880 \
#     6,200,200,4,128 \
#     configs/uniconv/deploy/uniconv_v0.2_m1_r34_s320x880_v200x200x4_c192_d2_f2.py \
#     work_dirs/uniconv/exp/cla/uniconv_cla_v0.4.1_z1_r34_s544x960_v340x200x4_ibaug_e40_interval1.0_lastft5/latest.pth

## m2: uniconv_v0.2_m2_r50_s256x704_v200x200x6_c256_d6_f2
# slurm_deploy KM_UniConvNV_M2_PLUS_fast \
#     6,3,256,704 \
#     6,200,200,6,128 \
#     configs/uniconv/deploy/uniconv_v0.2_m2_r50_s256x704_v200x200x6_c256_d6_f2.py \
#     work_dirs/uniconv/exp/cla/uniconv_cla_v0.4.1_z1_r34_s544x960_v340x200x4_ibaug_e40_interval1.0_lastft5/latest.pth

## m3: uniconv_v0.2_m3_r50_s320x880_v250x250x6_c224_d4_f2
# slurm_deploy KM_UniConvNV_M3_PLUS_fast \
#     6,3,320,880 \
#     6,250,250,6,128 \
#     configs/uniconv/deploy/uniconv_v0.2_m3_r50_s320x880_v250x250x6_c224_d4_f2.py \
#     work_dirs/uniconv/exp/cla/uniconv_cla_v0.4.1_z1_r34_s544x960_v340x200x4_ibaug_e40_interval1.0_lastft5/latest.pth

## m4: uniconv_v0.2_m4_r50_s384x1056_v300x300x6_c256_d6_f2
# slurm_deploy KM_UniConvNV_M4_PLUS_fast \
#     6,3,384,1056 \
#     6,300,300,6,128 \
#     configs/uniconv/deploy/uniconv_v0.2_m4_r50_s384x1056_v300x300x6_c256_d6_f2.py \
#     work_dirs/uniconv/exp/cla/uniconv_cla_v0.4.1_z1_r34_s544x960_v340x200x4_ibaug_e40_interval1.0_lastft5/latest.pth


## m2: uniconv_v0.2_m2_r50_s256x704_v200x200x6_c256_d6_f2
# slurm_deploy KM_UniConvNV_M2_f4_V3 \
#     6,3,256,704 \
#     1,200,200,6,256 \
#     configs/uniconv/deploy/uniconv_v0.3_m2_r50_s256x704_v200x200x6_c256_d6_f4.py \
#     work_dirs/uniconv/exp/cla/uniconv_cla_v0.4.1_z1_r34_s544x960_v340x200x4_ibaug_e40_interval1.0_lastft5/latest.pth
# slurm_deploy KM_UniConvNV_M2_f4_fuse_V3 \
#     6,3,256,704 \
#     1,200,200,6,256 \
#     configs/uniconv/deploy/uniconv_v0.3_m2_r50_s256x704_v200x200x6_c256_d6_f4.py \
#     work_dirs/uniconv/exp/cla/uniconv_cla_v0.4.1_z1_r34_s544x960_v340x200x4_ibaug_e40_interval1.0_lastft5/latest.pth

# slurm_deploy KM_UniConvNV_M2_f4_fuse_V3 \
#     6,3,256,704 \
#     1,200,200,6,256 \
#     configs/uniconv/deploy/uniconv_v0.3_m2_r50_s256x704_v200x200x6_c256_d6_f4.py \
#     work_dirs/uniconv/exp/cla/uniconv_cla_v0.4.1_z1_r34_s544x960_v340x200x4_ibaug_e40_interval1.0_lastft5/latest.pth




# slurm_deploy KM_UniConvNV_NEW_M0 \
#     6,3,256,704 \
#     1,200,200,4,256 \
#     configs/uniconv/deploy/uniconv_v0.3_m0_r18_s256x704_v200x200x4_c192_d2_f4.py \
#     work_dirs/uniconv/exp/cla/uniconv_cla_v0.4.1_z1_r34_s544x960_v340x200x4_ibaug_e40_interval1.0_lastft5/latest.pth

# slurm_deploy KM_UniConvNV_NEW_M1 \
#     6,3,256,704 \
#     1,200,200,4,256 \
#     configs/uniconv/deploy/uniconv_v0.3_m1_r34_s256x704_v200x200x4_c224_d4_f4.py \
#     work_dirs/uniconv/exp/cla/uniconv_cla_v0.4.1_z1_r34_s544x960_v340x200x4_ibaug_e40_interval1.0_lastft5/latest.pth

# slurm_deploy KM_UniConvNV_NEW_M2 \
#     6,3,320,880 \
#     1,250,250,6,256 \
#     configs/uniconv/deploy/uniconv_v0.3_m2_r34_s320x880_v250x250x6_c224_d4_f4.py \
#     work_dirs/uniconv/exp/cla/uniconv_cla_v0.4.1_z1_r34_s544x960_v340x200x4_ibaug_e40_interval1.0_lastft5/latest.pth

# slurm_deploy KM_UniConvNV_NEW_M3 \
#     6,3,320,880 \
#     1,250,250,6,256 \
#     configs/uniconv/deploy/uniconv_v0.3_m3_r50_s320x880_v250x250x6_c256_d6_f4.py \
#     work_dirs/uniconv/exp/cla/uniconv_cla_v0.4.1_z1_r34_s544x960_v340x200x4_ibaug_e40_interval1.0_lastft5/latest.pth

# slurm_deploy KM_UniConvNV_NEW_M4 \
#     6,3,384,1056 \
#     1,300,300,6,256 \
#     configs/uniconv/deploy/uniconv_v0.3_m4_r50_s384x1056_v300x300x6_c256_d6_f4.py \
#     work_dirs/uniconv/exp/cla/uniconv_cla_v0.4.1_z1_r34_s544x960_v340x200x4_ibaug_e40_interval1.0_lastft5/latest.pth

# slurm_deploy KM_UniConvNV_R18_TINY \
#     6,3,544,960 \
#     1,200,100,3,64 \
#     configs/uniconv/deploy/uniconv_cla_v0.6.1_r18_s544x960_v200x100x3_ibaug_e20_interval3.0.py \
#     work_dirs/uniconv/exp/cla/uniconv_cla_v0.6.1_r18_s544x960_v200x100x3_ibaug_e20_interval3.0/latest.pth


wait






















