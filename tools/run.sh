#!/usr/bin/env bash

function spring_slurm_train {
    PARTITION=$1
    GPUS=$2
    JOB_TYPE=$3
    EXPNAME=$4
    JOB_NAME=${5:-`basename $EXPNAME`}
    GPUS_PER_NODE=$(($GPUS<8?$GPUS:8))

    echo spring_slurm_train
    GPUS=$GPUS GPUS_PER_NODE=$GPUS_PER_NODE \
    ./tools/spring_slurm_train.sh \
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

    echo slurm_train
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
    JOB_NAME=${4:-`basename $EXPNAME`}
    RESUME=${5:-work_dirs/uniconv/exp/$EXPNAME/latest.pth}
    GPUS_PER_NODE=$(($GPUS<8?$GPUS:8))

    echo slurm_test
    GPUS=$GPUS GPUS_PER_NODE=$GPUS_PER_NODE \
    ./tools/slurm_test.sh \
        $PARTITION \
        $JOB_NAME \
        configs/uniconv/exp/$EXPNAME.py \
        $RESUME \
        --out work_dirs/uniconv/exp/$EXPNAME/results/results.pkl \
        --eval bbox
}

function slurm_analysis {
    PARTITION=$1
    EXPNAME=$2
    JOB_NAME=${3:-`basename $EXPNAME`}

    echo slurm_analysis
    GPUS=1 GPUS_PER_NODE=1 \
    ./tools/slurm_analysis.sh \
        $PARTITION \
        $JOB_NAME \
        configs/uniconv/exp/$EXPNAME.py \
        work_dirs/uniconv/exp/$EXPNAME/latest.pth
}


#################### debug
# MASTER_PORT=29581 slurm_train AD_RD_A100_40G 1 nuscenes/debug debug


#################### fullset-train-aug-exps
spring_slurm_train AD_GVT_A100_40G 1 auto nuscenes/uniconv_v0.1_e20_s256x704_v200x200x12_noaug
# spring_slurm_train AD_GVT_A100_40G 16 auto nuscenes/uniconv_v0.1_e20_s256x704_v200x200x12

# uniconv_v0.1_e20_s256x704_v200x200x12_noaug
# uniconv_v0.1_e20_s256x704_v200x200x12
# uniconv_v0.1_e20_s544x960_v200x200x12
# uniconv_v0.1_e20_s544x960_v400x400x12
# uniconv_v0.1_e20_s928x1600_v200x200x12
# uniconv_v0.1_e20_s928x1600_v400x400x12


#################### train-aug-exps
# spring_slurm_train AD_RD_A100_40G 16 auto nuscenes/m2bev_baseline baseline
# spring_slurm_train AD_RD_A100_40G 16 auto nuscenes/m2bev_baseline+distort distort
# spring_slurm_train AD_RD_A100_40G 16 auto nuscenes/m2bev_baseline+rescale+distort rescale+distort
# spring_slurm_train AD_RD_A100_40G 16 auto nuscenes/m2bev_baseline+rescale0.8+distort rescale0.8+distort
# spring_slurm_train AD_RD_A100_40G 16 auto nuscenes/m2bev_baseline+rescale0.9-1.1+distort rescale0.9-1.1+distort
# spring_slurm_train AD_RD_A100_40G 16 auto nuscenes/m2bev_baseline+rescale1.2+distort rescale1.2+distort
# spring_slurm_train AD_RD_A100_40G 16 auto nuscenes/m2bev_baseline+rescale1.0+distort rescale1.0+distort
# spring_slurm_train AD_RD_A100_40G 8 auto nuscenes/m2bev_baseline+imgaug imgaug
# spring_slurm_train AD_RD_A100_40G 16 auto nuscenes/m2bev_baseline+imgaug_v1.0 imgaug
# spring_slurm_train AD_RD_A100_40G 16 auto nuscenes/m2bev_baseline+imgaug_v1.1 imgaug
# spring_slurm_train AD_RD_A100_40G 16 auto nuscenes/m2bev_baseline+imgaug_v1.2 imgaug


#################### test-aug-exps
# slurm_test AD_RD_A100_40G 8 nuscenes/debug test ./work_dirs/m2bev/exp/nuscenes/m2bev_det_v0.2_lossdir0.8_200x200bev_6conv_wodcn_tiny_pure/epoch_12.pth
# slurm_test AD_RD_A100_40G 8 nuscenes/m2bev_baseline+rescale0.8+distort test ./work_dirs/m2bev/exp/nuscenes/m2bev_baseline+rescale0.8+distort/epoch_20.pth


#################### train
# spring_slurm_train AD_RD_A100_40G 8 auto nuscenes/m2bev_det_v0.2_lossdir0.8_200x200bev_6+0_4conv_seg_box2d train
# spring_slurm_train AD_RD_A100_40G 8 auto nuscenes/m2bev_det_v0.2_lossdir0.8_200x200bev_6conv train


#################### train-latency-exps
# spring_slurm_train AD_RD_A100_40G 8 auto nuscenes/m2bev_det_v0.2_lossdir0.8_200x200bev_6conv_wodcn_s928x1600_v400x400x12 s928x1600_v400x400x12
# spring_slurm_train AD_RD_A100_40G 8 auto nuscenes/m2bev_det_v0.2_lossdir0.8_200x200bev_6conv_wodcn_s928x1600_v200x200x12 s928x1600_v200x200x12 # done
# spring_slurm_train AD_RD_A100_40G 8 auto nuscenes/m2bev_det_v0.2_lossdir0.8_200x200bev_6conv_wodcn_s544x960_v200x200x12 s544x960_v200x200x12
# spring_slurm_train AD_GVT_A100_40G 8 auto nuscenes/m2bev_det_v0.2_lossdir0.8_200x200bev_6conv_wodcn_s544x960_v400x400x12 s544x960_v400x400x12
# spring_slurm_train AD_GVT_A100_40G 8 auto nuscenes/m2bev_det_v0.2_lossdir0.8_200x200bev_6conv_wodcn_s396x704_v200x200x12 s396x704_v200x200x12


#################### train-cla-exps
# spring_slurm_train AD_RD_A100_40G 8 auto cla/m2bev_det_v0.2_lossdir0.8_200x200bev_6conv_wodcn_tiny train


#################### test
# slurm_test AD_RD_A100_40G 8 nuscenes/m2bev_det_v0.2_lossdir0.8_200x200bev_6+0_4conv_seg_box2d_r101_20ep test
# slurm_test AD_RD_A100_40G 1 cla/m2bev_det_v0.2_lossdir0.8_200x200bev_6conv_wodcn_tiny test


#################### analysis
# slurm_analysis AD_RD_A100_40G nuscenes/m2bev_det_v0.2_lossdir0.8_200x200bev_6conv analysis
# slurm_analysis AD_RD_A100_40G nuscenes/m2bev_det_v0.2_lossdir0.8_200x200bev_6conv_wodcn_tiny analysis


#################### debug
# MASTER_PORT=29581 slurm_train AD_RD_A100_40G 1 cla/debug debug
# MASTER_PORT=29581 slurm_train AD_RD_A100_40G 1 nuscenes/debug debug

