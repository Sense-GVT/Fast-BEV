#include <iostream>
#include <vector>
#include <cmath>
#include <cstdio>
#include <chrono>
#include <cstring>
#include <assert.h>
#include <iostream>
#include <fstream>
#include <array>
#include <cuda_runtime.h>
#include "tensor.hpp"
#include "nlohmann/json.hpp"
// #include "cuda_accelerated_functions.h"
#include "cuda_accelerated_functions_project_v3.h"
#include "view_transformer.hpp"



using std::vector;
using std::string;
using std::array;
using namespace std;
using nlohmann::json;


static Tensor build_tensor(string meta_json_path, string data_path, DeviceType device_type) {
    
    std::ifstream infile(meta_json_path);
    assert(infile.is_open());
    json meta_json;
    infile >> meta_json;
    string dtype_str = meta_json.at("dtype").get<string>();
    DataType dtype;
    if (dtype_str == "float32") {
        dtype = DataType::kFP32;
    }
    else if (dtype_str == "float16") {
        dtype = DataType::kFP16;
    } else if (dtype_str == "int32") {
        dtype = DataType::kINT32;
    }
    else {
        assert(false);
    }
    std::cout <<"Building tensor from " << meta_json_path << " and " << data_path << " to " << ToString(device_type) << std::endl;
    vector<int32_t> dims;
    
    for (json item : meta_json.at("dims")) {
        dims.push_back(item.get<int32_t>());
    }
//     TensorShape ts = {
//         .rank = dims.size(),
//         .channel_axis = dims.size() >= 3 ? -1 : 0,
//     };
    
    TensorShape ts;
    ts.rank = dims.size();
    ts.channel_axis = dims.size() >= 3 ? -1 : 0;
    
    for (size_t i = 0; i < dims.size(); ++i) {
        ts.d[i] = dims[i];
    }
    vector<char> data_host(ts.nrof_elements() * Sizeof(dtype));
    infile.close();
    infile.open(data_path, std::ios::binary | std::ios::in);
    assert(infile.is_open());
    infile.seekg(0, std::ios::end);
    int64_t length = infile.tellg();
    if (length != data_host.size()) {
        std::cout << "in " << data_path << " got " << length << " bytes of binary data, but meta file tells a shape of " << ts.to_string();
        std::cout << " == " << data_host.size() << " bytes" << std::endl;
        assert(length == data_host.size());
    }
    infile.seekg(0, std::ios::beg);
    infile.read(data_host.data(), length);

    std::cout << "Got shape of " << ts.to_string() << ", type of " << ToString(dtype) << std::endl;
    if (device_type == DeviceType::kGPU) {
        void* ptr;
        CUDA_CHECK(cudaMalloc(&ptr, length));
        std::shared_ptr<void> data_device(ptr, [](void* p) {
            CUDA_CHECK(cudaFree(p));
        });
        cudaMemcpy(data_device.get(), data_host.data(), length, cudaMemcpyHostToDevice);
        return Tensor(data_device, ts, dtype, DeviceType::kGPU);
    }
    else {
        std::shared_ptr<void> data(new char[length], [](void* p) {
            delete []((char*)p);
        });
        memcpy(data.get(), data_host.data(), length);
        return Tensor(data, ts, dtype, DeviceType::kHOST);
    }
}

template<typename T>
static vector<T> build_cpu_vector(Tensor features_tensor) {
    size_t nrof_elements = features_tensor.get_shape().nrof_elements();
    size_t length = nrof_elements * sizeof(T);
    vector<T> result(nrof_elements);
    if (features_tensor.get_device() == DeviceType::kHOST) {
        memcpy(result.data(), features_tensor.get_data(), length);
    }
    else {
        cudaMemcpy(result.data(), features_tensor.get_data(), length, cudaMemcpyDeviceToHost);
    }
    return result;
}

template<typename T>
static void compare_two_vector(T* A, T* B, size_t nrof_elements, string name) {
    double sum_of_squared_error = 0;
    double sum_of_squared_A = 0;
    double sum_of_squared_B = 0;
    double A_dot_B = 0;
    for (size_t i = 0; i < nrof_elements; ++i) {
        T a = A[i];
        T b = B[i];
        sum_of_squared_error += (a - b) * (a - b);
        sum_of_squared_A += a * a;
        sum_of_squared_B += b * b;
        A_dot_B += a * b;
    }
    double RMSE = sqrt((sum_of_squared_error / nrof_elements));
    double cos_sim = A_dot_B / (sqrt(sum_of_squared_A) * sqrt(sum_of_squared_B));
    std::cout << name << ": RMSE=" << RMSE << ", cos similarity=" << cos_sim << std::endl;
    // std::cout <<"sum of A squared=" << sum_of_squared_A << ", sum of B squared=" << sum_of_squared_B << ", total " << nrof_elements << " elements" << std::endl;
}


void demo(string input_dir, string groundtruth_dir, string device, string feature_shape, string project_shape, string n_voxels_shape, string voxel_size_shape) {
   
    //     参数
    //     string device = "CPU";
    //     string feature_shape = "6_16_44_64";
    //     string project_shape = "d16_44";
    //     string n_voxels_shape = "128_128_6";
    //     string voxel_size_shape = "078_078_1";
    
   
    
    
    DeviceType device_type = DeviceType::kGPU;
    if (device == "CPU"){
        device_type = DeviceType::kHOST;
    }
    

    
    // 加载参数
    std::chrono::high_resolution_clock::time_point t1, t2;
    Tensor features_tensor = build_tensor(input_dir + "/features_meta_"+feature_shape+".json", input_dir + "/features_data_"+feature_shape+".bin", device_type);
    Tensor projection_tensor = build_tensor(input_dir + "/projection_meta.json", input_dir + "/projection_data_"+project_shape+".bin", device_type);

    
    Tensor n_voxels_tensor = build_tensor(input_dir + "/n_voxels_meta.json", input_dir + "/n_voxels_data_"+n_voxels_shape+".bin", DeviceType::kHOST);
    Tensor voxel_size_tensor = build_tensor(input_dir + "/voxel_size_meta.json", input_dir + "/voxel_size_data_"+voxel_size_shape+".bin", device_type);
    Tensor origin_tensor = build_tensor(input_dir + "/origin_meta.json", input_dir + "/origin_data.bin", device_type);
    
    vector<int32_t> n_voxels = build_cpu_vector<int32_t>(n_voxels_tensor);
    std::cout << "using voxels: (" << n_voxels[0] << "," << n_voxels[1] << "," << n_voxels[2] << ")" << std::endl;
    vector<float> voxel_size = build_cpu_vector<float>(voxel_size_tensor);
    std::cout << "using voxel size: (" << voxel_size[0] << "," << voxel_size[1] << "," << voxel_size[2] << ")" << std::endl;
    vector<float> origin = build_cpu_vector<float>(origin_tensor);
    std::cout << "using origin: (" << origin[0] << "," << origin[1] << "," << origin[2] << ")" << std::endl;
    
    int32_t n_images = features_tensor.get_shape().d[0];
    int32_t height = features_tensor.get_shape().d[1];
    int32_t width = features_tensor.get_shape().d[2];
    int32_t n_channels = features_tensor.get_shape().d[3];
    size_t nrof_voxels = n_voxels[0] * n_voxels[1] * n_voxels[2];
    
    
    
    int lut_times = device_type == DeviceType::kGPU?1:2;
    
    std::shared_ptr<int32_t> LUT(new int32_t[nrof_voxels * lut_times], [](int32_t* p) {
            delete []((int32_t*)p);
    });
    std::shared_ptr<int32_t> valid(new int32_t[nrof_voxels], [](int32_t* p) {
            delete []((int32_t*)p);
    });
    std::shared_ptr<float> volume(new float[nrof_voxels*n_channels], [](float* p) {
            delete []((float*)p);
    });
    
    // 执行测试
    int test_count = 50;
    float test_time = 0;
    for (int i =0; i < test_count; i++){
        t1 = std::chrono::high_resolution_clock::now();
        
        if (device_type == DeviceType::kGPU){
            build_LUT_GPU(n_voxels, voxel_size_tensor, origin_tensor, projection_tensor,
                        n_images, height, width, n_channels,
                        LUT, valid, volume);
        }else{
            build_LUT_CPU(n_voxels, voxel_size_tensor, origin_tensor, projection_tensor,
                        n_images, height, width, n_channels,
                        LUT, valid, volume);
        }
           
        t2 = std::chrono::high_resolution_clock::now();
        std::cout << "build LUT @ GPU cost : " << std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1).count() * 1000 << " ms" << std::endl;
    
        
        
        if (device_type == DeviceType::kGPU){
            t1 = std::chrono::high_resolution_clock::now();
            
            backproject_LUT_GPU(features_tensor, LUT, volume, n_voxels);
            
            t2 = std::chrono::high_resolution_clock::now();
            std::cout << "backproject using LUT @ GPU cost : " << std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1).count() * 1000 << " ms" << std::endl;
            test_time +=  std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1).count() * 1000;
        }else{
            t1 = std::chrono::high_resolution_clock::now();
            
            backproject_LUT_CPU(features_tensor, LUT, volume, n_voxels);
            
            t2 = std::chrono::high_resolution_clock::now();
            std::cout << "backproject using LUT @ GPU cost : " << std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1).count() * 1000 << " ms" << std::endl;
            test_time +=  std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1).count() * 1000;
        }
        
        
        if (device_type == DeviceType::kGPU){
            vector<float> volume_cpu(nrof_voxels * n_channels);
            vector<int32_t> valid_cpu(nrof_voxels);
            cudaMemcpy(volume_cpu.data(), volume.get(), nrof_voxels * n_channels * sizeof(float), cudaMemcpyDeviceToHost);
            cudaMemcpy(valid_cpu.data(), valid.get(), nrof_voxels * sizeof(int32_t), cudaMemcpyDeviceToHost);


            float valid_num = 0.0;
            for (size_t i = 0; i < nrof_voxels; ++i) {
                if (valid_cpu[i] > 0){
                    valid_num += 1.0;
                }
            }
            std::cout<<"\n";
            std::cout << "backproject project rate: "<<valid_num << ", " << valid_num/nrof_voxels << std::endl;
        }else{
        
            float valid_num = 0.0;
            for (size_t i = 0; i < nrof_voxels; ++i) {
                if (valid.get()[i] > 0){
                    valid_num += 1.0;
                }
            }
            std::cout<<"\n";
            std::cout << "backproject project rate: "<<valid_num << ", " << valid_num/nrof_voxels << std::endl;
        }
        
        
    }
    
    std::cout << "AVG backproject using LUT @ GPU cost times " <<test_count<<" cost: "<<test_time/test_count<< " ms" << std::endl;
    
    
    

    // check result accuracy 
    

    
    
//     Tensor volume_groundtruth_tensor = build_tensor(groundtruth_dir + "/volume_sum_meta.json", groundtruth_dir + "/volume_sum_data.bin", DeviceType::kHOST);
//     Tensor valid_groundtruth_tensor = build_tensor(groundtruth_dir + "/valid_meta.json", groundtruth_dir + "/valid_data.bin", DeviceType::kHOST);
//     compare_two_vector<float>((float*)volume_groundtruth_tensor.get_data(), volume_cpu.data(), nrof_voxels * n_channels, "volume");
//     compare_two_vector<int32_t>((int32_t*)valid_groundtruth_tensor.get_data(), valid_cpu.data(), nrof_voxels, "valid");
}



void build_LUT_CPU(vector<int32_t> n_voxels, Tensor voxel_size, Tensor origin,
                    Tensor projection, int32_t n_images, int32_t height, int32_t width, int32_t n_channels,
                    std::shared_ptr<int32_t>& LUT, std::shared_ptr<int32_t>& valid, std::shared_ptr<float>& volume) {
    
    
    // projection:  6 x 3 x 4               (N, 3, 4)
    int n_x_voxels = n_voxels[0];
    int n_y_voxels = n_voxels[1];
    int n_z_voxels = n_voxels[2];
    
    float* voxel_sizep = (float*)voxel_size.get_data();
    float size_x = voxel_sizep[0];
    float size_y = voxel_sizep[1];
    float size_z = voxel_sizep[2];

    float* originp = (float*)origin.get_data();
    float origin_x = originp[0];
    float origin_y = originp[1];
    float origin_z = originp[2];
    
    int nrof_voxels = n_x_voxels * n_y_voxels * n_z_voxels;
    
    int32_t* LUTp = LUT.get();
    int32_t* validp = valid.get();
    
    std::vector<float> ar(3);
    std::vector<float> pt(3);
    size_t offset = 0;
    float count = 0.0;
    
    for (int zi = 0; zi < n_z_voxels; ++zi) {          // 4
        for (int yi = 0; yi < n_y_voxels; ++yi) {      // 100
            for (int xi = 0; xi < n_x_voxels; ++xi) {  // 200
                auto current_lut = &LUTp[offset * 2];
                *current_lut = -1;
                *(current_lut + 1) = 0;
                for (int img = 0; img < n_images; img++) {  // 6
                    // const auto &projection_img = projection[img];
                    pt[0] = (xi - n_x_voxels / 2.0f) * size_x + origin_x;
                    pt[1] = (yi - n_y_voxels / 2.0f) * size_y + origin_y;
                    pt[2] = (zi - n_z_voxels / 2.0f) * size_z + origin_z;

                    for (int i = 0; i < 3; ++i) {
                        // ar[i] = projection_img[i][3];
                        ar[i] = ((float*)projection.get_data())[((img * 3) + i) * 4 + 3];
                        for (int j = 0; j < 3; ++j) {
                            // ar[i] += projection_img[i][j] * pt[j];
                             ar[i] += ((float*)projection.get_data())[(img * 3 + i) * 4 + j] * pt[j];
                        }
                    }

                    int x = round(ar[0] / ar[2]);
                    int y = round(ar[1] / ar[2]);
                    float z = ar[2];

                    if ((x >= 0) && (y >= 0) && (x < width) && (y < height) &&
                        (z > 0)) {
                        *current_lut = img;
                        // *(current_lut + 1) = y * w_stride + x;
                        *(current_lut + 1) = y * width + x;
                        count+=1;
                        
                        validp[offset] = 1;
                        break;
                    }
                }
                ++offset;
            }
        }
    }
}

void backproject_LUT_CPU(Tensor features, std::shared_ptr<int32_t> LUT, std::shared_ptr<float> volume,
                        vector<int32_t> n_voxels) {
    
    
    int32_t n_images = features.get_shape().d[0];
    int32_t height = features.get_shape().d[1];
    int32_t width = features.get_shape().d[2];
    int32_t n_channels = features.get_shape().d[3];
    
    int n_x_voxels = n_voxels[0];
    int n_y_voxels = n_voxels[1];
    int n_z_voxels = n_voxels[2];
    
    float* featuresp = (float*)features.get_data();
    float* volumep = volume.get();
    
    size_t volume_count = n_x_voxels * n_y_voxels * n_z_voxels * 2;
    size_t copy_size_per_iter = n_channels * sizeof(float);
    
       
    int32_t* LUTp = LUT.get();
    
    // #pragma omp parallel for  // slower in this circumstance, should NOT ENABLE
    for (size_t offset = 0; offset < volume_count; offset=offset+2) {
        int img = LUTp[offset];
        int target = LUTp[offset+1];
        if (img >= 0) {
            float* src = featuresp + img * height * width * n_channels + target * n_channels;
            float* dst = volumep + offset/2 * n_channels;
            memcpy(dst, src, copy_size_per_iter);
        }
    }

    
}



void build_LUT_GPU(vector<int32_t> n_voxels, Tensor voxel_size, Tensor origin,
                    Tensor projection, int32_t n_images, int32_t height, int32_t width, int32_t n_channels,
                    std::shared_ptr<int32_t>& LUT, std::shared_ptr<int32_t>& valid, std::shared_ptr<float>& volume) {
    size_t nrof_elements = n_voxels[0] * n_voxels[1] * n_voxels[2];
    
    
    int32_t* lut_p;
    CUDA_CHECK(cudaMalloc(&lut_p, nrof_elements * sizeof(int32_t)));
    LUT.reset(lut_p, [](int32_t* p) {
        CUDA_CHECK(cudaFree(p));
    });
    vector<int32_t> tmp1(nrof_elements, -1);
    cudaMemcpy(LUT.get(), tmp1.data(), nrof_elements * sizeof(int32_t), cudaMemcpyHostToDevice);
    
    
    int32_t* valid_p;
    CUDA_CHECK(cudaMalloc(&valid_p, nrof_elements * sizeof(int32_t)));
    valid.reset(valid_p, [](int32_t* p) {
        CUDA_CHECK(cudaFree(p));
    });
    float* volume_p;
    CUDA_CHECK(cudaMalloc(&volume_p, nrof_elements * n_channels * sizeof(float)));
    volume.reset(volume_p, [](float* p) {
        CUDA_CHECK(cudaFree(p));
    });
    // Zero initialize volume
    vector<float> tmp(nrof_elements * n_channels, 0.0f);
    cudaMemcpy(volume.get(), tmp.data(), nrof_elements * n_channels * sizeof(float), cudaMemcpyHostToDevice);
    build_LUT_cuda(n_voxels, (float*)voxel_size.get_data(), (float*)origin.get_data(), (float*)projection.get_data(),
                    LUT.get(), valid.get(),
                    n_images, height, width
                    );
}


void backproject_LUT_GPU(Tensor features, std::shared_ptr<int32_t> LUT, std::shared_ptr<float> volume,
                        vector<int32_t> n_voxels) {
    int32_t n_images = features.get_shape().d[0];
    int32_t height = features.get_shape().d[1];
    int32_t width = features.get_shape().d[2];
    int32_t n_channels = features.get_shape().d[3];
    backproject_LUT_CUDA((float*)features.get_data(), LUT.get(), volume.get(),
        n_images, height, width, n_channels,
        n_voxels
    );
}