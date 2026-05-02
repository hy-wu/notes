#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define OUT_TILE_DIM 32
#define MASK_RADIUS 2
#define MASK_DIM (2 * MASK_RADIUS + 1)
#define IN_TILE_DIM (OUT_TILE_DIM + 2 * MASK_RADIUS)

__constant__ float mask_c[MASK_DIM][MASK_DIM];

__global__ void convolution_2d(float* input, float* output, unsigned int width, unsigned int height) {
    int outRow = blockIdx.y * blockDim.y + threadIdx.y;
    int outCol = blockIdx.x * blockDim.x + threadIdx.x;

    if (outRow < height && outCol < width) {
        float sum = 0.0f;
        for (int maskRow = 0; maskRow < MASK_DIM; maskRow++) {
            for (int maskCol = 0; maskCol < MASK_DIM; maskCol++) {
                int inRow = outRow + maskRow - MASK_RADIUS;
                int inCol = outCol + maskCol - MASK_RADIUS;
                if (inRow >= 0 && inRow < height && inCol >= 0 && inCol < width) {
                    sum += input[inRow * width + inCol] * mask_c[maskRow][maskCol];
                }
            }
        }
        output[outRow * width + outCol] = sum;
    }
}

__global__ void convolution_2d_tiled(float* input, float* output, unsigned int width, unsigned int height) {
    // 定义共享内存，需要包含卷积所需的边界（Halo）大小
    __shared__ float N_ds[IN_TILE_DIM][IN_TILE_DIM];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    int outRow = blockIdx.y * OUT_TILE_DIM + ty;
    int outCol = blockIdx.x * OUT_TILE_DIM + tx;

    // 1. 协作将输入数据加载到共享内存
    int linearIdx = ty * blockDim.x + tx;
    int numThreads = blockDim.x * blockDim.y;
    
    // 由于需要加载的元素 (IN_TILE_DIM * IN_TILE_DIM) 比线程数 (OUT_TILE_DIM * OUT_TILE_DIM) 多，采用循环加载
    for (int i = linearIdx; i < IN_TILE_DIM * IN_TILE_DIM; i += numThreads) {
        int sharedRow = i / IN_TILE_DIM;
        int sharedCol = i % IN_TILE_DIM;
        
        int inRow = blockIdx.y * OUT_TILE_DIM + sharedRow - MASK_RADIUS;
        int inCol = blockIdx.x * OUT_TILE_DIM + sharedCol - MASK_RADIUS;
        
        if (inRow >= 0 && inRow < height && inCol >= 0 && inCol < width) {
            N_ds[sharedRow][sharedCol] = input[inRow * width + inCol];
        } else {
            N_ds[sharedRow][sharedCol] = 0.0f; // 边界外填充0
        }
    }

    __syncthreads(); // 确保所有数据加载完成

    // 2. 使用共享内存数据进行卷积计算
    if (outRow < height && outCol < width) {
        float sum = 0.0f;
        for (int maskRow = 0; maskRow < MASK_DIM; maskRow++) {
            for (int maskCol = 0; maskCol < MASK_DIM; maskCol++) {
                sum += N_ds[ty + maskRow][tx + maskCol] * mask_c[maskRow][maskCol];
            }
        }
        output[outRow * width + outCol] = sum;
    }
}

void convolution(float mask[][MASK_DIM], float* input, float* output, unsigned int width, unsigned int height) {
    float *input_d, *output_d;
    cudaMalloc(&input_d, width * height * sizeof(float));
    cudaMalloc(&output_d, width * height * sizeof(float));
    cudaDeviceSynchronize();
    cudaMemcpy(input_d, input, width * height * sizeof(float), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    cudaMemcpyToSymbol(mask_c, mask, MASK_DIM * MASK_DIM * sizeof(float));
    cudaDeviceSynchronize();
    dim3 blockDim(OUT_TILE_DIM, OUT_TILE_DIM);
    dim3 gridDim((width + OUT_TILE_DIM - 1) / OUT_TILE_DIM, (height + OUT_TILE_DIM - 1) / OUT_TILE_DIM);
    convolution_2d<<<gridDim, blockDim>>>(input_d, output_d, width, height);
    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    float milliseconds = 0;
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Time taken by GPU kernel: %f ms\n", milliseconds);
    cudaMemcpy(output, output_d, width * height * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(input_d);
    cudaFree(output_d);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

void convolution_tiled(float mask[][MASK_DIM], float* input, float* output, unsigned int width, unsigned int height) {
    float *input_d, *output_d;
    cudaMalloc(&input_d, width * height * sizeof(float));
    cudaMalloc(&output_d, width * height * sizeof(float));
    cudaDeviceSynchronize();
    cudaMemcpy(input_d, input, width * height * sizeof(float), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    cudaMemcpyToSymbol(mask_c, mask, MASK_DIM * MASK_DIM * sizeof(float));
    cudaDeviceSynchronize();
    dim3 blockDim(OUT_TILE_DIM, OUT_TILE_DIM);
    dim3 gridDim((width + OUT_TILE_DIM - 1) / OUT_TILE_DIM, (height + OUT_TILE_DIM - 1) / OUT_TILE_DIM);
    convolution_2d_tiled<<<gridDim, blockDim>>>(input_d, output_d, width, height);
    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    float milliseconds = 0;
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Time taken by Tiled GPU kernel: %f ms\n", milliseconds);
    cudaMemcpy(output, output_d, width * height * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(input_d);
    cudaFree(output_d);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

void convolution_cpu(float mask[][MASK_DIM], float* input, float* output, unsigned int width, unsigned int height) {
    for (unsigned int outRow = 0; outRow < height; outRow++) {
        for (unsigned int outCol = 0; outCol < width; outCol++) {
            float sum = 0.0f;
            for (int maskRow = 0; maskRow < MASK_DIM; maskRow++) {
                for (int maskCol = 0; maskCol < MASK_DIM; maskCol++) {
                    int inRow = outRow + maskRow - MASK_RADIUS;
                    int inCol = outCol + maskCol - MASK_RADIUS;
                    if (inRow >= 0 && inRow < height && inCol >= 0 && inCol < width) {
                        sum += input[inRow * width + inCol] * mask[maskRow][maskCol];
                    }
                }
            }
            output[outRow * width + outCol] = sum;
        }
    }
}


int main() {
    unsigned int width = 4096;
    unsigned int height = 4096;
    float* input = (float*)malloc(width * height * sizeof(float));
    float* output = (float*)malloc(width * height * sizeof(float));
    float* output_tiled = (float*)malloc(width * height * sizeof(float));
    float* output_cpu = (float*)malloc(width * height * sizeof(float));
    float mask[MASK_DIM][MASK_DIM] = {0};
    // Initialize mask with some values
    for (int i = 0; i < MASK_DIM; i++) {
        for (int j = 0; j < MASK_DIM; j++) {
            mask[i][j] = rand() / (float)RAND_MAX;
        }
    }

    // Initialize input with some values
    for (unsigned int i = 0; i < width * height; i++) {
        input[i] = rand() / (float)RAND_MAX;
    }
    time_t start = clock();

    convolution(mask, input, output, width, height);
    
    time_t end = clock();
    double time = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("Time taken by GPU: %f seconds\n", time);

    time_t start_tiled = clock();
    convolution_tiled(mask, input, output_tiled, width, height);
    time_t end_tiled = clock();
    double time_tiled = ((double)(end_tiled - start_tiled)) / CLOCKS_PER_SEC;
    printf("Time taken by Tiled GPU: %f seconds\n", time_tiled);

    time_t start_cpu = clock();
    convolution_cpu(mask, input, output_cpu, width, height);
    time_t end_cpu = clock();
    double time_cpu = ((double)(end_cpu - start_cpu)) / CLOCKS_PER_SEC;
    printf("Time taken by CPU: %f seconds\n", time_cpu);

    // Compare the results
    for (unsigned int i = 0; i < width * height; i++) {
        if (abs(output[i] - output_cpu[i]) > 1e-4) {
            printf("Mismatch at index %u: GPU = %f, CPU = %f\n", i, output[i], output_cpu[i]);
            break;
        }
    }
    
    // Compare the Tiled results
    for (unsigned int i = 0; i < width * height; i++) {
        if (abs(output_tiled[i] - output_cpu[i]) > 1e-4) {
            printf("Mismatch at index %u: Tiled GPU = %f, CPU = %f\n", i, output_tiled[i], output_cpu[i]);
            break;
        }
    }

    // Process output as needed

    free(input);
    free(output);
    free(output_tiled);
    free(output_cpu);
}
