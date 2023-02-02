cmake_minimum_required(VERSION 3.13)
project(ViewTransformer)

enable_language(CUDA)
find_package(CUDAToolkit REQUIRED)
message(STATUS "CUDA version (${CUDAToolkit_VERSION})")
message(STATUS "CUDA include dir (${CUDAToolkit_INCLUDE_DIRS})")
message(STATUS "CUDA link dir (${CUDAToolkit_LIBRARY_DIR})")
set(CMAKE_CUDA_ARCHITECTURES 70) # set this according to YOUR GPU ARCH

set(CMAKE_CXX_STANDARD 14)

set(PROJECT_INCLUDE_DIR
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CUDAToolkit_INCLUDE_DIRS}
)
include_directories(${PROJECT_INCLUDE_DIR})
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin)
set(LIBRARY_OUTPUT_PATH ${CMAKE_BINARY_DIR}/lib)



file(GLOB SRC_LIST *.cpp)
file(GLOB CUDA_SRC "*.cu")

ADD_EXECUTABLE(ViewTransformer.run ${SRC_LIST} ${CUDA_SRC})

target_link_directories(ViewTransformer.run PUBLIC ${CUDAToolkit_LIBRARY_DIR})

target_link_libraries(ViewTransformer.run PUBLIC cudart ${CMAKE_DL_LIBS})

ADD_LIBRARY(VT SHARED view_transformer.cpp)
