cmake_minimum_required(VERSION 3.18)
project(CUDA_Notes LANGUAGES CXX CUDA)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)

# Find CUDA is implicit with LANGUAGES CUDA in newer CMake
# but we might need specific libraries like cusparse
find_package(CUDAToolkit REQUIRED)

enable_testing()

# 启用 CUDA 语言支持
enable_language(CUDA)

# 告诉 CMake 导出编译命令文件给 clangd 使用
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)


set(CUDA_SOURCES
    colorhist.cu
    conv.cu
    enqueue.cu
    graph.cu
    merge.cu
    mm.cu
    query_device.gemini.cu
    reduce.cu
    scan.cu
    spmv.cu
    stencil.cu
    vecadd.cu
    bamps_gpu_thrust.cu
)

foreach(src ${CUDA_SOURCES})
    get_filename_component(name ${src} NAME_WE)
    add_executable(${name} ${src})
    
    # Common linking
    target_link_libraries(${name} PRIVATE CUDA::cudart)
    
    # Specific linking
    if(name STREQUAL "spmv")
        target_link_libraries(${name} PRIVATE CUDA::cusparse)
    endif()
    
    # Add test
    add_test(NAME test_${name} COMMAND ${name})
endforeach()
