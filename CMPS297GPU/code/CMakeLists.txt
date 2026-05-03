cmake_minimum_required(VERSION 3.18)
project(CUDA_Notes LANGUAGES CXX CUDA)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)

# Find CUDA is implicit with LANGUAGES CUDA in newer CMake
# but we might need specific libraries like cusparse
find_package(CUDAToolkit REQUIRED)

enable_testing()

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
