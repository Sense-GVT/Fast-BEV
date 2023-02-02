#include "cuda_accelerated_functions_project_v3.h"
#include <cuda.h>
#include "stdio.h"
#include <iostream>

__global__ void build_LUT_kernel(int32_t n_x_voxels, int32_t n_y_voxels, int32_t n_z_voxels,
                                    float* voxel_size, float* origin, float* projection,
                                    int32_t* LUT, int32_t* valid,
                                    int32_t n_images, int32_t height, int32_t width) {
    int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int32_t zi = idx % n_z_voxels;
    idx /= n_z_voxels;
    int32_t yi = idx % n_y_voxels;
    idx /= n_y_voxels;
    int32_t xi = idx % n_x_voxels;
    idx /= n_x_voxels;
    int32_t img = idx;
    if (img < n_images && LUT[(xi * n_y_voxels + yi) * n_z_voxels + zi] == -1 ) {

        float size_x = voxel_size[0];
        float size_y = voxel_size[1];
        float size_z = voxel_size[2];
        float ar[3];
        float pt[3];
        pt[0] = (xi - n_x_voxels / 2.0f) * size_x + origin[0];
        pt[1] = (yi - n_y_voxels / 2.0f) * size_y + origin[1];
        pt[2] = (zi - n_z_voxels / 2.0f) * size_z + origin[2];

        for (int i = 0; i < 3; ++i) {
            ar[i] = 0;
            for (int j = 0; j < 3; ++j) {
                ar[i] += projection[(img * 3 + i) * 4 + j] * pt[j];
            }
            ar[i] += projection[((img * 3) + i) * 4 + 3];
        }
        int32_t x = round(ar[0] / ar[2]);
        int32_t y = round(ar[1] / ar[2]);
        float z = ar[2];

        bool fit_in = (x >= 0) && (y >= 0) && (x < width) && (y < height) && (z > 0);
        int32_t target;
        if (fit_in) {
            target = (img * height + y) * width + x;
            
            int offset = (xi * n_y_voxels + yi) * n_z_voxels + zi;  // [xi,yi,zi]
            LUT[offset] = target;
            valid[offset] = fit_in;
        }
        else {
            target = -1;
        }

    }
}


void build_LUT_cuda(vector<int32_t> n_voxels, float* voxel_size_dev, float* origin_dev, float* projection,
                    int32_t* LUT, int32_t* valid,
                    int32_t n_images, int32_t height, int32_t width) {
    int32_t n_x_voxels = n_voxels[0];
    int32_t n_y_voxels = n_voxels[1];
    int32_t n_z_voxels = n_voxels[2];
    size_t total_nrof_voxels = n_images * n_voxels[0] * n_voxels[1] * n_voxels[2];
    #define BLOCK_SIZE 1024
    dim3 thread_per_block(BLOCK_SIZE);
    dim3 block_per_grid((total_nrof_voxels + thread_per_block.x - 1) / thread_per_block.x);

    printf("build here\n");
    build_LUT_kernel<<< block_per_grid, thread_per_block >>>(n_x_voxels, n_y_voxels, n_z_voxels, 
                        voxel_size_dev, origin_dev, projection,
                        LUT, valid,
                        n_images, height, width);
}


__global__ void backproject_LUT_kernel(float* features, int32_t* LUT, float* volume,
                                        size_t total_nrof_voxels, int32_t n_channels) {
    int32_t offset = blockIdx.x * blockDim.x + threadIdx.x;
    size_t nrof_float4_copies_per_iter = n_channels / 4; // We assume n_channels % 4 == 0

    if (offset < total_nrof_voxels) {
        int32_t target = LUT[offset];
        if (target >= 0) {
                float4* src = (float4*)(features + target * n_channels);
                float4* dst = (float4*)(volume + offset * n_channels);
                for (size_t i = 0; i < nrof_float4_copies_per_iter; ++i) {
                    dst[i] = src[i];
                }
            }
    }
}

void backproject_LUT_CUDA(float* features_dev, int32_t* LUT_dev, float* volume_dev,
                        int32_t n_images, int32_t height, int32_t width, int32_t n_channels,
                        vector<int32_t> n_voxels) {
    int32_t n_x_voxels = n_voxels[0];
    int32_t n_y_voxels = n_voxels[1];
    int32_t n_z_voxels = n_voxels[2];
    size_t total_nrof_voxels = n_voxels[0] * n_voxels[1] * n_voxels[2];
    #define BLOCK_SIZE 1024
    dim3 thread_per_block(BLOCK_SIZE);
    dim3 block_per_grid((total_nrof_voxels + thread_per_block.x - 1) / thread_per_block.x);
    backproject_LUT_kernel<<< block_per_grid, thread_per_block >>>(features_dev, LUT_dev, volume_dev,
        total_nrof_voxels, n_channels
    );
}