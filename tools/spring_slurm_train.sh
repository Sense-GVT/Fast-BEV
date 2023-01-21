#!/usr/bin/env bash

set -x

PARTITION=$1
JOB_TYPE=$2
JOB_NAME=$3
CONFIG=$4
WORK_DIR=$5
PY_ARGS=${@:6}
GPUS=${GPUS:-8}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
CPUS_PER_TASK=${CPUS_PER_TASK:-5}
SRUN_ARGS=${SRUN_ARGS:-"-s"}
RETRY=${RETRY:-1}

for try in `seq $RETRY`; do
    PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
    spring.submit arun -p ${PARTITION} --gpu -n$GPUS --ntasks-per-node=${GPUS_PER_NODE} --cpus-per-task=${CPUS_PER_TASK} \
        --quotatype=${JOB_TYPE} \
        --job-name=${JOB_NAME}-$try-$RETRY \
        ${SRUN_ARGS} \
    "python -u tools/train.py ${CONFIG} --work-dir=${WORK_DIR} --launcher='slurm' ${PY_ARGS}"
done
