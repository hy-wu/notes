#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define BLOCK_DIM 1024
#define COARSE_FACTOR 8

__global__ void scanKoggeStoneCoarsed_kernel(float *input, float *output, float *partialSums, unsigned int N) {
    unsigned int bSegment = blockIdx.x * blockDim.x * COARSE_FACTOR;
    __shared__ float buffer_s[BLOCK_DIM * COARSE_FACTOR];
    for (unsigned int c = 0; c < COARSE_FACTOR; c++) {
        unsigned int i = bSegment + threadIdx.x + c * blockDim.x;
        buffer_s[threadIdx.x + c * blockDim.x] = (i < N) ? input[i] : 0.0f;
    }
    __syncthreads();
    unsigned int tSegment = threadIdx.x * COARSE_FACTOR;
    for (unsigned int c = 1; c < COARSE_FACTOR; c++) {
        buffer_s[tSegment + c] += buffer_s[tSegment + c - 1];
    }

    __shared__ float buffer1_s[BLOCK_DIM];
    __shared__ float buffer2_s[BLOCK_DIM];
    float *inBuffer_s = buffer1_s;
    float *outBuffer_s = buffer2_s;

    inBuffer_s[threadIdx.x] = buffer_s[tSegment + COARSE_FACTOR - 1];  // TODO: boundary check?
    __syncthreads();
    for (unsigned int stride = 1; stride <= blockDim.x / 2; stride *= 2) {
        if (threadIdx.x >= stride) {
            outBuffer_s[threadIdx.x] = inBuffer_s[threadIdx.x] + inBuffer_s[threadIdx.x - stride];
        } else {
            outBuffer_s[threadIdx.x] = inBuffer_s[threadIdx.x];
        }
        __syncthreads();
        float *temp = inBuffer_s;
        inBuffer_s = outBuffer_s;
        outBuffer_s = temp;
    }
    if (threadIdx.x > 0) {
        for (unsigned int c = 0; c < COARSE_FACTOR; c++) {
            buffer_s[tSegment + c] += inBuffer_s[threadIdx.x - 1];
        }
    }
    if (threadIdx.x == blockDim.x - 1) {
        partialSums[blockIdx.x] = inBuffer_s[threadIdx.x];
    }
    __syncthreads();
    for (unsigned int c = 0; c < COARSE_FACTOR; c++) {
        unsigned int i = bSegment + threadIdx.x + c * blockDim.x;
        if (i < N) {
            output[i] = buffer_s[tSegment + c];
        }
    }
}

__global__ void addKoggeStoneCoarsed_kernel(float *output, float *partialSums, unsigned int N) {
    unsigned int bSegment = blockIdx.x * blockDim.x * COARSE_FACTOR;
    if (blockIdx.x > 0) {
        float addValue = partialSums[blockIdx.x - 1];
        for (unsigned int c = 0; c < COARSE_FACTOR; c++) {
            unsigned int i = bSegment + threadIdx.x + c * blockDim.x;
            if (i < N) {
                output[i] += addValue;
            }
        }
    }
}

__global__ void scanKoggeStone_kernel(float *input, float *output, float *partialSums, unsigned int N) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    __shared__ float buffer1_s[BLOCK_DIM];
    __shared__ float buffer2_s[BLOCK_DIM];
    float *inBuffer_s = buffer1_s;
    float *outBuffer_s = buffer2_s;

    inBuffer_s[threadIdx.x] = (i < N) ? input[i] : 0.0f;
    __syncthreads();
    for (unsigned int stride = 1; stride <= blockDim.x / 2; stride *= 2) {
        if (threadIdx.x >= stride) {
            outBuffer_s[threadIdx.x] = inBuffer_s[threadIdx.x] + inBuffer_s[threadIdx.x - stride];
        } else {
            outBuffer_s[threadIdx.x] = inBuffer_s[threadIdx.x];
        }
        __syncthreads();
        float *temp = inBuffer_s;
        inBuffer_s = outBuffer_s;
        outBuffer_s = temp;
    }
    if (threadIdx.x == blockDim.x - 1) {
        partialSums[blockIdx.x] = inBuffer_s[threadIdx.x];
    }
    if (i < N) {
        output[i] = inBuffer_s[threadIdx.x];
    }

}

__global__ void addKoggeStone_kernel(float *output, float *partialSums, unsigned int N) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (blockIdx.x > 0 && i < N) {
        output[i] += partialSums[blockIdx.x - 1];
    }
}


__global__ void scanBrentKung_kernel(float *input, float *output, float *partialSums, unsigned int N) {
    unsigned int segment = blockIdx.x * blockDim.x * 2;
    __shared__ float buffer_s[BLOCK_DIM*2];
    buffer_s[threadIdx.x] = (segment + threadIdx.x < N) ? input[segment + threadIdx.x] : 0.0f;
    buffer_s[threadIdx.x + blockDim.x] = (segment + threadIdx.x + blockDim.x < N) ? input[segment + threadIdx.x + blockDim.x] : 0.0f;
    __syncthreads();
    
    // Reduction
    for (unsigned int stride = 1; stride <= blockDim.x; stride *= 2) {
        unsigned int i = (threadIdx.x + 1) * stride * 2 - 1;
        if (i < 2 * blockDim.x) {
            buffer_s[i] += buffer_s[i - stride];
        }
        __syncthreads();
    }

    // Post-reduction
    for (unsigned int stride = blockDim.x / 2; stride >= 1; stride /= 2) {
        unsigned int i = (threadIdx.x + 1) * stride * 2 - 1;
        if (i + stride < 2 * blockDim.x) {
            buffer_s[i + stride] += buffer_s[i];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        partialSums[blockIdx.x] = buffer_s[2*blockDim.x-1];
    }
    if (segment + threadIdx.x < N) {
        output[segment + threadIdx.x] = buffer_s[threadIdx.x];
    }
    if (segment + threadIdx.x + blockDim.x < N) {
        output[segment + threadIdx.x + blockDim.x] = buffer_s[threadIdx.x + blockDim.x];
    }
}

__global__ void addBrentKung_kernel(float *output, float *partialSums, unsigned int N) {
    unsigned int segment = blockIdx.x * blockDim.x * 2;
    if (blockIdx.x > 0) {
        float addValue = partialSums[blockIdx.x - 1];
        unsigned int i = segment + threadIdx.x;
        if (i < N) {
            output[i] += addValue;
        }
        i += blockDim.x;
        if (i < N) {
            output[i] += addValue;
        }
    }
}

void scanBrentKung_gpu_d(float *input_d, float *output_d, unsigned int N) {
    time_t t_start, t_end;
    t_start = clock();

    const unsigned int numThreadsPerBlock = BLOCK_DIM;
    const unsigned int numElementsPerBlock = numThreadsPerBlock * 2;
    const unsigned int numBlocks = (N + numElementsPerBlock - 1) / numElementsPerBlock;

    float *partialSums_d;
    cudaMalloc(&partialSums_d, numBlocks * sizeof(float));
    cudaDeviceSynchronize();
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    scanBrentKung_kernel<<<numBlocks, numThreadsPerBlock>>>(input_d, output_d, partialSums_d, N);
    cudaDeviceSynchronize();

    if (numBlocks > 1) {
        scanBrentKung_gpu_d(partialSums_d, partialSums_d, numBlocks);
        addBrentKung_kernel<<<numBlocks, numThreadsPerBlock>>>(output_d, partialSums_d, N);
        cudaDeviceSynchronize();
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("GPU kernels time: %f ms\n", milliseconds);

    cudaFree(partialSums_d);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    t_end = clock();
    double t_time = (double)(t_end - t_start) / CLOCKS_PER_SEC;
    printf("Total GPU scan time: %f seconds\n", t_time);
}


void scanKoggeStone_gpu_d(float *input_d, float *output_d, unsigned int N) {
    time_t t_start, t_end;
    t_start = clock();

    const unsigned int numThreadsPerBlock = BLOCK_DIM;
    const unsigned int numElementsPerBlock = numThreadsPerBlock;
    const unsigned int numBlocks = (N + numElementsPerBlock - 1) / numElementsPerBlock;

    float *partialSums_d;
    cudaMalloc(&partialSums_d, numBlocks * sizeof(float));
    cudaDeviceSynchronize();
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    scanKoggeStone_kernel<<<numBlocks, numThreadsPerBlock>>>(input_d, output_d, partialSums_d, N);
    cudaDeviceSynchronize();

    if (numBlocks > 1) {
        scanKoggeStone_gpu_d(partialSums_d, partialSums_d, numBlocks);
        addKoggeStone_kernel<<<numBlocks, numThreadsPerBlock>>>(output_d, partialSums_d, N);
        cudaDeviceSynchronize();
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("GPU kernels time: %f ms\n", milliseconds);

    cudaFree(partialSums_d);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    t_end = clock();
    double t_time = (double)(t_end - t_start) / CLOCKS_PER_SEC;
    printf("Total GPU scan time: %f seconds\n", t_time);
}

void scanKoggeStoneCoarsed_gpu_d(float *input_d, float *output_d, unsigned int N) {
    time_t t_start, t_end;
    t_start = clock();

    const unsigned int numThreadsPerBlock = BLOCK_DIM;
    const unsigned int numElementsPerBlock = numThreadsPerBlock * COARSE_FACTOR;
    const unsigned int numBlocks = (N + numElementsPerBlock - 1) / numElementsPerBlock;

    float *partialSums_d;
    cudaMalloc(&partialSums_d, numBlocks * sizeof(float));
    cudaDeviceSynchronize();
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    scanKoggeStoneCoarsed_kernel<<<numBlocks, numThreadsPerBlock>>>(input_d, output_d, partialSums_d, N);
    cudaDeviceSynchronize();

    if (numBlocks > 1) {
        scanKoggeStone_gpu_d(partialSums_d, partialSums_d, numBlocks);
        addKoggeStoneCoarsed_kernel<<<numBlocks, numThreadsPerBlock>>>(output_d, partialSums_d, N);
        cudaDeviceSynchronize();
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("GPU kernels time: %f ms\n", milliseconds);

    cudaFree(partialSums_d);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    t_end = clock();
    double t_time = (double)(t_end - t_start) / CLOCKS_PER_SEC;
    printf("Total GPU scan time: %f seconds\n", t_time);
}

void scan_cpu(float *input, float *output, unsigned int N) {
    time_t t_start, t_end;
    t_start = clock();

    float partialSum = 0.0f;
    for (unsigned int i = 0; i < N; i++) {
        partialSum += input[i];
        output[i] = partialSum;
    }

    t_end = clock();
    double t_time = (double)(t_end - t_start) / CLOCKS_PER_SEC;
    printf("CPU scan time: %f seconds\n", t_time);
}

bool checkResult(const float* cpuResult, const float* gpuResult, int N) {
    const float EPSILON = 1e-4f; // 针对单精度 float 推荐的容差范围
    for (int i = 0; i < N; i++) {
        float expected = cpuResult[i];
        float actual = gpuResult[i];
        
        // 计算绝对误差
        float diff = std::abs(expected - actual);
        
        // 防止除以0，计算相对误差
        float relativeError = (std::abs(expected) > 1e-5f) ? (diff / std::abs(expected)) : diff;

        if (relativeError > EPSILON) {
            // std::cerr << "Error at index " << i 
            //           << ": expected " << expected 
            //           << " but got " << actual 
            //           << " (Relative Error: " << relativeError << ")" << std::endl;
            printf("Error at index %d: expected %f but got %f (Relative Error: %f)\n", i, expected, actual, relativeError);
            return false;
        }
    }
    printf("Results are correct within the tolerance of %e.\n", EPSILON);
    return true;
}


int main() {
    const unsigned int N = 1 << 24; // 1 million elements
    float *input_h = (float *)malloc(N * sizeof(float));
    float *output_h = (float *)malloc(N * sizeof(float));
    float *output_cpu = (float *)malloc(N * sizeof(float));

    for (unsigned int i = 0; i < N; i++) {
        input_h[i] = random() / (float)RAND_MAX; // Random float between 0 and 1
    }
    scan_cpu(input_h, output_cpu, N);

    float *input_d, *output_d;
    cudaMalloc(&input_d, N * sizeof(float));
    cudaMalloc(&output_d, N * sizeof(float));

    cudaMemcpy(input_d, input_h, N * sizeof(float), cudaMemcpyHostToDevice);
    printf("Running Kogge-Stone scan on GPU...\n");
    scanKoggeStone_gpu_d(input_d, output_d, N);
    cudaMemcpy(output_h, output_d, N * sizeof(float), cudaMemcpyDeviceToHost);
    checkResult(output_cpu, output_h, N);

    printf("Running Brent-Kung scan on GPU...\n");
    scanBrentKung_gpu_d(input_d, output_d, N);
    cudaMemcpy(output_h, output_d, N * sizeof(float), cudaMemcpyDeviceToHost);
    checkResult(output_cpu, output_h, N);

    printf("Running Coarsed Kogge-Stone scan on GPU...\n");
    scanKoggeStoneCoarsed_gpu_d(input_d, output_d, N);
    cudaMemcpy(output_h, output_d, N * sizeof(float), cudaMemcpyDeviceToHost);
    checkResult(output_cpu, output_h, N);

    free(input_h);
    free(output_h);
    cudaFree(input_d);
    cudaFree(output_d);
    return 0;
}
