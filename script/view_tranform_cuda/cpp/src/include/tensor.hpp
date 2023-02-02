#pragma once
#include <vector>
#include <array>
#include <memory>
#include <assert.h>
#include <sstream>
#include <numeric>
#include <string>


using std::array;
using std::vector;
using std::string;


enum class DataType: int32_t {
    kFP32       = 0,
    kFP16       = 1,
    kINT32      = 2,
}; // enum class DataType

inline size_t Sizeof(DataType t) {
    switch (t) {
        case DataType::kFP32:
        case DataType::kINT32:
            return 4;
        case DataType::kFP16:
            return 2;
    }
    return 0;
}

inline string ToString(DataType t) {
        switch (t) {
        case DataType::kFP32:
            return "FP32";
        case DataType::kINT32:
            return "INT32";
        case DataType::kFP16:
            return "FP16";
    }
    return "UNKNOWN";
}

enum class DeviceType: int32_t {
    kHOST = 0,
    kGPU = 1
}; // emum class DeviceType

inline string ToString(DeviceType t) {
    switch (t) {
        case DeviceType::kHOST:
            return "HOST";
        case DeviceType::kGPU:
            return "GPU";
    }
    return "UNKNOWN";
}


struct TensorShape {
    static constexpr int32_t MAX_DIMS{8};
    int32_t d[MAX_DIMS];
    size_t rank;
    int32_t channel_axis;       // useless in this demo
    size_t get_rank() const { return rank;}
    size_t nrof_elements() const {
        if (rank == 0) {
            return 0;
        }
        size_t result = 1;
        for (size_t i = 0; i < rank; ++i) {
            result *= d[i];
        }
        return result;
    }
    string to_string() const {
        std::stringstream ss;
        ss << "(";
        for (size_t i = 0; i < rank; ++i) {
            if (i > 0) {
                ss << ",";
            }
            ss << d[i];
        }
        ss << ")";
        return ss.str();
    }
}; // struct TensorShape


class Tensor {
public:
    Tensor()=default;
    Tensor(const Tensor& rhs)=default;
    Tensor(std::shared_ptr<void> data, TensorShape shape, DataType dtype, DeviceType device) :
        data_(data), shape_(shape), dtype_(dtype), device_(device) {}
    ~Tensor()=default;
    Tensor to(DeviceType device);
    void* get_data() {
        assert(data_);
        return data_.get();
    }
    const TensorShape& get_shape() const {return shape_;}
    DataType get_dtype() const {return dtype_;}
    DeviceType get_device() const {return device_;}
private:
    std::shared_ptr<void> data_;
    TensorShape shape_;
    DataType dtype_;
    DeviceType device_;
};












