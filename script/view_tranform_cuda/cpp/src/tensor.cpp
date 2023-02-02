#include "tensor.hpp"
#include <memory>
#include <cstring>
#include <cuda_runtime.h>
// #include "cuda_accelerated_functions.h"
#include "cuda_accelerated_functions_project_v3.h"

Tensor Tensor::to(DeviceType device) {
    Tensor result;
    size_t nrof_elements = this->get_shape().nrof_elements();
    size_t total_size_in_bytes = nrof_elements * Sizeof(this->dtype_);
    std::shared_ptr<void> data;
    if (device == DeviceType::kHOST) {
        data.reset(new uint8_t[total_size_in_bytes], [](uint8_t* p) {delete []p;});
    }
    else {
        void* ptr;
        CUDA_CHECK(cudaMalloc(&ptr, total_size_in_bytes));
        data.reset(ptr, [](void* p) {
            CUDA_CHECK(cudaFree(p));
        });
    }
    if (this->device_ == DeviceType::kHOST) {
        if (device == DeviceType::kHOST) {
            memcpy(data.get(), this->get_data(), total_size_in_bytes);
        }
        else {
            cudaMemcpy(data.get(), this->get_data(), total_size_in_bytes, cudaMemcpyHostToDevice);
        }
    }
    else {
        if (device == DeviceType::kHOST) {
            cudaMemcpy(data.get(), this->get_data(), total_size_in_bytes, cudaMemcpyDeviceToHost);
        }
        else {
            cudaMemcpy(data.get(), this->get_data(), total_size_in_bytes, cudaMemcpyDeviceToDevice);
        }
    }
    return Tensor(data, this->shape_, this->dtype_, this->device_);
}