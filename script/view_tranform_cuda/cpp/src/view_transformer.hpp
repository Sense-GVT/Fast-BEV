#pragma once
#include <vector>
#include <string>
#include <array>
#include "tensor.hpp"

using std::vector;
using std::array;
using std::string;


void demo(string input_dir, string groundtruth_dir, string device, string feature_shape, string project_shape, string n_voxels_shape, string voxel_size_shape);

void build_LUT_GPU(vector<int32_t> n_voxes, Tensor voxel_size, Tensor origin,
                    Tensor projection, int n_images, int height, int width, int32_t n_channels,
                    std::shared_ptr<int32_t>& LUT, std::shared_ptr<int32_t>& valid, std::shared_ptr<float>& volume);

void backproject_LUT_GPU(Tensor features, std::shared_ptr<int32_t> LUT, std::shared_ptr<float> volume,
                        vector<int32_t> n_voxels);


void build_LUT_CPU(vector<int32_t> n_voxes, Tensor voxel_size, Tensor origin,
                    Tensor projection, int n_images, int height, int width, int32_t n_channels,
                    std::shared_ptr<int32_t>& LUT, std::shared_ptr<int32_t>& valid, std::shared_ptr<float>& volume);

void backproject_LUT_CPU(Tensor features, std::shared_ptr<int32_t> LUT, std::shared_ptr<float> volume,
                        vector<int32_t> n_voxels);













