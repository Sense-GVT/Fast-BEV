#!/bin/bash


function srun_speed_test {
    VT_RES_DIR=../
    device=$1
    feature_shape=$2
    project_shape=$3
    n_voxels_shape=$4
    voxel_size_shape=$5
    
    srun -p toolchain --job-name=detr3d_test -n1 --gres=gpu:1 --ntasks=1 --ntasks-per-node=1 --cpus-per-task=16 ./build/bin/ViewTransformer.run $VT_RES_DIR/input $VT_RES_DIR/ground_truth $device $feature_shape $project_shape $n_voxels_shape $voxel_size_shape
    
}


function speed_test {
    VT_RES_DIR=../
    device=$1
    feature_shape=$2
    project_shape=$3
    n_voxels_shape=$4
    voxel_size_shape=$5
    

    if [[ $device == "CPU" ]]; then
        ./build/bin/ViewTransformer.run $VT_RES_DIR/input $VT_RES_DIR/ground_truth $device $feature_shape $project_shape $n_voxels_shape $voxel_size_shape 2>&1 | tee ./logs/${device}_feature_${feature_shape}_n_voxels_${n_voxels_shape}_voxel_size_${voxel_size_shape}.txt
    
    else
        srun_speed_test $device $feature_shape $project_shape $n_voxels_shape $voxel_size_shape 2>&1 | tee ./logs/${device}_feature_${feature_shape}_n_voxels_${n_voxels_shape}_voxel_size_${voxel_size_shape}.txt
    fi
}

# BevDepth benchmark 

# speed_test GPU 6_16_44_64 d16_44 128_128_6 078_078_1

# speed_test GPU 6_32_88_64 d32_88 128_128_6 078_078_1

# speed_test GPU 6_40_110_64 d40_110 128_128_6 078_078_1

# speed_test CPU 6_16_44_64 d16_44 128_128_6 078_078_1

# speed_test CPU 6_32_88_64 d32_88 128_128_6 078_078_1

# speed_test CPU 6_40_110_64 d40_110 128_128_6 078_078_1



# NV Matrix benchmark

# speed_test CPU 6_16_44_80 d16_44 128_128_6 078_078_1
# speed_test CPU 6_16_44_160 d16_44 128_128_6 078_078_1


# speed_test CPU 6_32_88_80 d32_88 128_128_6 078_078_1
# speed_test CPU 6_32_88_160 d32_88 128_128_6 078_078_1


# speed_test CPU 6_32_88_80 d32_88 256_256_6 039_039_1
# speed_test CPU 6_32_88_160 d32_88 256_256_6 039_039_1



speed_test GPU 6_16_44_80 d16_44 128_128_6 078_078_1
speed_test GPU 6_16_44_160 d16_44 128_128_6 078_078_1


speed_test GPU 6_32_88_80 d32_88 128_128_6 078_078_1
speed_test GPU 6_32_88_160 d32_88 128_128_6 078_078_1


speed_test GPU 6_32_88_80 d32_88 256_256_6 039_039_1
speed_test GPU 6_32_88_160 d32_88 256_256_6 039_039_1