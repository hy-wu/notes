#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define BLOCK_DIM 1024
#define COARSE_FACTOR 8
#define WARP_SIZE 32

#define RUN_ALL_SIZES false

__global__ void reduce_kernel(float *input, float *partialSums,
                              unsigned int N) {
  unsigned int segment = blockIdx.x * blockDim.x * 2;
  unsigned int i = segment + threadIdx.x * 2;
  for (unsigned int stride = 1; stride <= BLOCK_DIM; stride *= 2) {
    if (i < N && threadIdx.x % stride == 0) {
      input[i] += input[i + stride];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    partialSums[blockIdx.x] = input[segment];
  }
}

__global__ void reduce_kernel_coalesced(float *input, float *partialSums,
                                        unsigned int N) {
  unsigned int segment = blockIdx.x * blockDim.x * 2;
  unsigned int i = segment + threadIdx.x;

  __shared__ float input_s[BLOCK_DIM];
  input_s[threadIdx.x] =
      input[i] + (i + blockDim.x < N ? input[i + blockDim.x] : 0.0f);
  __syncthreads();
  for (unsigned int stride = BLOCK_DIM / 2; stride > 0; stride /= 2) {
    if (threadIdx.x < stride) {
      if (i + stride < N) {
        input_s[threadIdx.x] += input_s[threadIdx.x + stride];
      }
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    partialSums[blockIdx.x] = input_s[0];
  }
}

__global__ void reduce_kernel_warped(float *input, float *partialSums,
                                     unsigned int N) {
  unsigned int segment = blockIdx.x * blockDim.x * 2;
  unsigned int i = segment + threadIdx.x;

  __shared__ float input_s[BLOCK_DIM];
  input_s[threadIdx.x] =
      input[i] + (i + blockDim.x < N ? input[i + blockDim.x] : 0.0f);
  __syncthreads();
  for (unsigned int stride = BLOCK_DIM / 2; stride > WARP_SIZE; stride /= 2) {
    if (threadIdx.x < stride) {
      if (i + stride < N) {
        input_s[threadIdx.x] += input_s[threadIdx.x + stride];
      }
    }
    __syncthreads();
  }

  // Reduction tree using shuffle
  float sum = 0.0f;
  if (threadIdx.x < WARP_SIZE) {
    sum = input_s[threadIdx.x] + input_s[threadIdx.x + WARP_SIZE];
    for (unsigned int stride = WARP_SIZE / 2; stride > 0; stride /= 2) {
      sum += __shfl_down_sync(0xffffffff, sum, stride);
    }
  }

  if (threadIdx.x == 0) {
    partialSums[blockIdx.x] = sum;
  }
}

__global__ void reduce_kernel_coarsened(float *input, float *partialSums,
                                        unsigned int N) {
  unsigned int segment = blockIdx.x * blockDim.x * COARSE_FACTOR * 2;
  unsigned int i = segment + threadIdx.x;
  float sum = 0.0f;
  for (unsigned int j = 0; j < COARSE_FACTOR * 2 && i + j * blockDim.x < N;
       j++) {
    sum += input[i + j * blockDim.x];
  }
  __shared__ float input_s[BLOCK_DIM];
  input_s[threadIdx.x] = sum;
  __syncthreads();
  for (unsigned int stride = BLOCK_DIM / 2; stride > 0; stride /= 2) {
    if (threadIdx.x < stride) {
      input_s[threadIdx.x] += input_s[threadIdx.x + stride];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    partialSums[blockIdx.x] = input_s[0];
  }
}

float reduce_gpu(float *input, unsigned int N, int type) {
  time_t t_start, t_end;
  float t_total;
  t_start = clock();

  float *input_d, *partialSums_d;
  unsigned int numBlocks;
  if (type == 2) { // COARSENED
    numBlocks = (N + BLOCK_DIM * COARSE_FACTOR * 2 - 1) /
                (BLOCK_DIM * COARSE_FACTOR * 2);
  } else {
    numBlocks = (N + BLOCK_DIM * 2 - 1) / (BLOCK_DIM * 2);
  }

  float *partialSums = (float *)malloc(numBlocks * sizeof(float));
  cudaMalloc(&input_d, N * sizeof(float));
  cudaMalloc(&partialSums_d, numBlocks * sizeof(float));

  cudaEvent_t start, stop;

  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaMemcpy(input_d, input, N * sizeof(float), cudaMemcpyHostToDevice);

  cudaEventRecord(start); // 将计时器移到Memcpy之后，纯粹衡量Kernel性能
  if (type == 0) {
    reduce_kernel<<<numBlocks, BLOCK_DIM>>>(input_d, partialSums_d, N);
  } else if (type == 1) {
    reduce_kernel_coalesced<<<numBlocks, BLOCK_DIM>>>(input_d, partialSums_d,
                                                      N);
  } else if (type == 2) {
    reduce_kernel_coarsened<<<numBlocks, BLOCK_DIM>>>(input_d, partialSums_d,
                                                      N);
  } else if (type == 3) {
    reduce_kernel_warped<<<numBlocks, BLOCK_DIM>>>(input_d, partialSums_d, N);
  } else {
    fprintf(stderr, "Invalid reduction type: %d\n", type);
    return -1.0f;
  }
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  cudaMemcpy(partialSums, partialSums_d, numBlocks * sizeof(float),
             cudaMemcpyDeviceToHost);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);

  cudaFree(input_d);
  cudaFree(partialSums_d);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  double sum = 0.0;
  for (unsigned int i = 0; i < numBlocks; i++) {
    sum += partialSums[i];
  }
  free(partialSums);

  t_end = clock();
  t_total = double(t_end - t_start) / CLOCKS_PER_SEC * 1000.0;
  const char *type_names[] = {"Interleaved", "Coalesced", "Coarsened",
                              "WarpByCoale"};
  printf("GPU %-11s Time: %8.3f ms | Kernel Time: %8.3f ms | Result: %f\n",
         type_names[type], t_total, milliseconds, (float)sum);

  return (float)sum;
}

float reduce_cpu(float *input, unsigned int N) {
  double sum = 0.0;
  for (unsigned int i = 0; i < N; i++) {
    sum += (double)input[i];
  }
  return (float)sum;
}

int main() {
  // 定义要遍历的不同大小的 N
  unsigned int N_values[] = {1 << 20, 1 << 21, 1 << 22, 1 << 23,
                             1 << 24, 1 << 25, 1 << 26, 1 << 27,
                             1 << 28, 1 << 29, 1 << 30};
  int num_sizes = sizeof(N_values) / sizeof(unsigned int);
    if (!RUN_ALL_SIZES) {
      num_sizes = 4; // 只测试前4个大小，快速验证
    }
  for (int i = 0; i < num_sizes; i++) {
    unsigned int N = N_values[i];
    printf("\n================================================\n");
    printf("Reducing %u elements (%.2f MB)\n", N,
           N * sizeof(float) / (1024.0 * 1024.0));

    float *input = (float *)malloc(N * sizeof(float));
    for (unsigned int j = 0; j < N; j++) {
      input[j] = rand() / (float)RAND_MAX / sqrtf(N);
    }

    time_t start = clock();
    float cpuResult = reduce_cpu(input, N);
    time_t end = clock();
    double cpu_time = double(end - start) / CLOCKS_PER_SEC * 1000.0;
    printf("CPU             Time: %8.3f ms | Result: %f\n", cpu_time,
           cpuResult);

    reduce_gpu(input, N, 0); // 0 = 原始交错访问
    reduce_gpu(input, N, 1); // 1 = 连续内存合并访问
    reduce_gpu(input, N, 2); // 2 = 共享内存+线程粗化
    reduce_gpu(input, N, 3); // 3 = Warp-optimized
    free(input);
  }
  return 0;
}
