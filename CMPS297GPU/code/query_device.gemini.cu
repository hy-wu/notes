#include <stdio.h>
#include <cuda_runtime.h>

// 根据计算能力 (Compute Capability) 推算每个 SM 的 CUDA 核心数 (流处理器)
int getCoresPerSM(int major, int minor) {
    switch (major) {
        case 2: return (minor == 1) ? 48 : 32;       // Fermi 架构
        case 3: return 192;                          // Kepler 架构
        case 5: return 128;                          // Maxwell 架构
        case 6: return (minor == 1 || minor == 2) ? 128 : 64; // Pascal 架构
        case 7: return 64;                           // Volta (7.0) 和 Turing (7.5) 架构
        case 8: return (minor == 0) ? 64 : 128;      // Ampere 架构 (A100是64, RTX30系是128)
        case 9: return 128;                          // Hopper 架构 (以及 Ada Lovelace 8.9)
        default: return -1;                          // 未知或未来的架构
    }
}

int main() {
    int deviceCount = 0;
    cudaError_t error = cudaGetDeviceCount(&deviceCount);
    
    if (error != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(error));
        return -1;
    }

    if (deviceCount == 0) {
        printf("没有找到支持 CUDA 的设备。\n");
        return 1;
    }

    for (int i = 0; i < deviceCount; ++i) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);

        printf("========== 第 %d 个设备: %s ==========\n", i, prop.name);
        printf("计算能力 (Compute Capability): %d.%d\n", prop.major, prop.minor);
        printf("多处理器数量 (SMs): %d\n", prop.multiProcessorCount);

        int coresPerSM = getCoresPerSM(prop.major, prop.minor);
        if (coresPerSM != -1) {
            printf("每个 SM 的流处理器数量 (CUDA Cores per SM): %d\n", coresPerSM);
            printf("总流处理器数量 (Total CUDA Cores): %d\n", coresPerSM * prop.multiProcessorCount);
        } else {
            printf("每个 SM 的流处理器数量: 未知 (无法识别的架构)\n");
        }

        printf("是否支持 Tensor Core: %s\n", (prop.major >= 7) ? "是 (计算能力 >= 7.0)" : "否");
        printf("是否支持硬件 BF16: %s\n", (prop.major >= 8) ? "是 (计算能力 >= 8.0)" : "否");
        printf("单双精度性能比 (FP32/FP64 Perf Ratio): 不可直接从 cudaDeviceProp 读取\n");

        printf("每个 Block 的最大线程数 (Max Threads Per Block): %d\n", prop.maxThreadsPerBlock);
        printf("每个 SM 的最大线程数 (Max Threads Per SM): %d\n", prop.maxThreadsPerMultiProcessor);
        printf("线程束大小 (Warp Size): %d\n", prop.warpSize);
        printf("各 Block 维度的最大值 (Max Thread Dimensions): (%d, %d, %d)\n", 
               prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
        printf("各 Grid 维度的最大值 (Max Grid Dimensions): (%d, %d, %d)\n", 
               prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
        printf("每个 Block 可用的最大共享内存 (Shared Mem Per Block): %zu Bytes\n", prop.sharedMemPerBlock);
        printf("总全局内存 (Total Global Memory): %.2f GB\n", (float)prop.totalGlobalMem / (1024 * 1024 * 1024));
        printf("\n");
    }

    return 0;
}
