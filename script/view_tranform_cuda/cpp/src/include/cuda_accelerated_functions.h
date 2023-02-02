#pragma once
#include <cuda.h>
#include <vector>
#include <iostream>

using std::vector;

#ifndef CUDA_CHECK
#define CUDA_CHECK(callstr)                                                                    \
    {                                                                                          \
        cudaError_t error_code = callstr;                                                      \
        if (error_code != cudaSuccess) {                                                       \
            std::cerr << "CUDA error " << error_code << " at " << __FILE__ << ":" << __LINE__; \
            assert(0);                                                                         \
        }                                                                                      \
    }

#endif


void build_LUT_cuda(vector<int32_t> n_voxels, float* voxel_size_dev, float* origin_dev, float* projection_dev,
                    int32_t* LUT, int32_t* valid,
                    int32_t n_images, int32_t height, int32_t width);


void backproject_LUT_CUDA(float* features_dev, int32_t* LUT_dev, float* volume_dev,
                        int32_t n_images, int32_t height, int32_t width, int32_t n_channels,
                        vector<int32_t> n_voxels);






