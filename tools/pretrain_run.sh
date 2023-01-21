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


PARTITION=$1
QUOTATYPE=$2


#################### pretrain-resnet-exps
# spring_slurm_train $PARTITION 32 $QUOTATYPE coco/cascade_mask_rcnn_r18_fpn_mstrain_3x_coco
# spring_slurm_train $PARTITION 32 $QUOTATYPE coco/cascade_mask_rcnn_r34_fpn_mstrain_3x_coco
# spring_slurm_train $PARTITION 32 $QUOTATYPE coco/cascade_mask_rcnn_r101_dcn_fpn_mstrain_3x_coco

# spring_slurm_train $PARTITION 8 $QUOTATYPE nuimages/cascade_mask_rcnn_r18_fpn_coco-mstrain_3x_20e_nuim
# spring_slurm_train $PARTITION 16 $QUOTATYPE nuimages/cascade_mask_rcnn_r34_fpn_coco-mstrain_3x_20e_nuim
# spring_slurm_train $PARTITION 8 $QUOTATYPE nuimages/cascade_mask_rcnn_r50_fpn_coco-mstrain_3x_20e_nuim
# spring_slurm_train $PARTITION 8 $QUOTATYPE nuimages/cascade_mask_rcnn_r101_fpn_coco-mstrain_3x_20e_nuim
# spring_slurm_train $PARTITION 16 $QUOTATYPE nuimages/cascade_mask_rcnn_r101_dcn_fpn_coco-mstrain_3x_20e_nuim

#################### pretrain-regnet-exps
# spring_slurm_train $PARTITION 32 $QUOTATYPE coco/cascade_mask_rcnn_regnetx-3.2GF_fpn_mstrain_3x_coco
# spring_slurm_train $PARTITION 32 $QUOTATYPE coco/cascade_mask_rcnn_regnetx-12GF_dcn_fpn_mstrain_3x_coco
# spring_slurm_train $PARTITION 32 $QUOTATYPE coco/cascade_mask_rcnn_regnetx-12GF_fpn_mstrain_3x_coco
# spring_slurm_train $PARTITION 32 $QUOTATYPE coco/cascade_mask_rcnn_regnetx-8GF_fpn_mstrain_3x_coco
# spring_slurm_train $PARTITION 32 $QUOTATYPE coco/cascade_mask_rcnn_regnetx-6.4GF_fpn_mstrain_3x_coco

# spring_slurm_train $PARTITION 16 $QUOTATYPE nuimages/cascade_mask_rcnn_regnetx-400MF_fpn_coco-mstrain_3x_20e_nuim
# spring_slurm_train $PARTITION 16 $QUOTATYPE nuimages/cascade_mask_rcnn_regnetx-800MF_fpn_coco-mstrain_3x_20e_nuim
# spring_slurm_train $PARTITION 16 $QUOTATYPE nuimages/cascade_mask_rcnn_regnetx-1.6GF_fpn_coco-mstrain_3x_20e_nuim
# spring_slurm_train $PARTITION 16 $QUOTATYPE nuimages/cascade_mask_rcnn_regnetx-3.2GF_fpn_coco-mstrain_3x_20e_nuim
# spring_slurm_train $PARTITION 16 $QUOTATYPE nuimages/cascade_mask_rcnn_regnetx-4GF_fpn_coco-mstrain_3x_20e_nuim

# spring_slurm_train $PARTITION 16 $QUOTATYPE nuimages/cascade_mask_rcnn_regnetx-6.4GF_fpn_coco-mstrain_3x_20e_nuim
# spring_slurm_train $PARTITION 16 $QUOTATYPE nuimages/cascade_mask_rcnn_regnetx-8GF_fpn_coco-mstrain_3x_20e_nuim
# spring_slurm_train $PARTITION 16 $QUOTATYPE nuimages/cascade_mask_rcnn_regnetx-12GF_fpn_coco-mstrain_3x_20e_nuim
# spring_slurm_train $PARTITION 16 $QUOTATYPE nuimages/cascade_mask_rcnn_regnetx-12GF_dcn_fpn_coco-mstrain_3x_20e_nuim

wait
