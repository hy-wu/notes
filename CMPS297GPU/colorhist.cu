
#include <cstdio>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define NUM_BINS 256
#define BLOCK_SIZE 1024
#define COARSE_FACTOR 1

__global__ void histogram_kernel(unsigned char *image, unsigned int *bins,
                                 unsigned int width, unsigned int height) {
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < width * height) {
    unsigned char b = image[i];
    atomicAdd(&bins[b], 1);
  }
}

__global__ void histPrivatized_kernel(unsigned char *image, unsigned int *bins,
                                      unsigned int width, unsigned int height) {
  __shared__ unsigned int bins_s[NUM_BINS];
  int n_init = (NUM_BINS + BLOCK_SIZE - 1) / BLOCK_SIZE;
  for (unsigned int b = 0; b < n_init; ++b) {
    unsigned int j = threadIdx.x * n_init + b;
    if (j < NUM_BINS) {
      bins_s[j] = 0;
    }
  }

  __syncthreads();

  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < width * height) {
    unsigned char b = image[i];
    atomicAdd(&bins_s[b], 1);
  }

  __syncthreads();

  for (unsigned int b = 0; b < n_init; ++b) {
    unsigned int j = threadIdx.x * n_init + b;
    if (j < NUM_BINS) {
      atomicAdd(&bins[j], bins_s[j]);
    }
  }
}

void histPrivatized_gpu(unsigned char *image, unsigned int *bins,
                        unsigned int width, unsigned int height) {

  unsigned char *image_d;
  unsigned int *bins_d;
  cudaMalloc(&image_d, width * height * sizeof(unsigned char));
  cudaMalloc(&bins_d, NUM_BINS * sizeof(unsigned int));

  cudaDeviceSynchronize();

  cudaMemcpy(image_d, image, width * height * sizeof(unsigned char),
             cudaMemcpyHostToDevice);
  cudaMemset(bins_d, 0, NUM_BINS * sizeof(unsigned int));

  cudaDeviceSynchronize();

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);

  unsigned int numThreadsPerBlock = BLOCK_SIZE;

  unsigned int numBlocks =
      (width * height + numThreadsPerBlock - 1) / numThreadsPerBlock;

  histPrivatized_kernel<<<numBlocks, numThreadsPerBlock>>>(image_d, bins_d,
                                                           width, height);

  cudaDeviceSynchronize();
  cudaEventRecord(stop);
  float milliseconds = 0;
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&milliseconds, start, stop);
  printf("Time taken by Privatized GPU kernel: %f ms\n", milliseconds);
  cudaMemcpy(bins, bins_d, NUM_BINS * sizeof(unsigned int),
             cudaMemcpyDeviceToHost);

  cudaFree(image_d);
  cudaFree(bins_d);
}

__global__ void histCoarsed_kernel(unsigned char *image, unsigned int *bins,
                                   unsigned int width, unsigned int height) {
  __shared__ unsigned int bins_s[NUM_BINS];
  int n_init = (NUM_BINS + BLOCK_SIZE - 1) / BLOCK_SIZE;
  for (unsigned int b = 0; b < n_init; ++b) {
    unsigned int j = threadIdx.x * n_init + b;
    if (j < NUM_BINS) {
      bins_s[j] = 0;
    }
  }

  __syncthreads();

  for (unsigned int c = 0; c < COARSE_FACTOR; ++c) {
    unsigned int i =
        blockIdx.x * blockDim.x * COARSE_FACTOR + c * BLOCK_SIZE + threadIdx.x;
    if (i < width * height) {
      unsigned char b = image[i];
      atomicAdd(&bins_s[b], 1);
    }
  }

  __syncthreads();

  for (unsigned int b = 0; b < n_init; ++b) {
    unsigned int j = threadIdx.x * n_init + b;
    if (j < NUM_BINS) {
      atomicAdd(&bins[j], bins_s[j]);
    }
  }
}

void histCoarsed_gpu(unsigned char *image, unsigned int *bins,
                     unsigned int width, unsigned int height) {

  unsigned char *image_d;
  unsigned int *bins_d;
  cudaMalloc(&image_d, width * height * sizeof(unsigned char));
  cudaMalloc(&bins_d, NUM_BINS * sizeof(unsigned int));

  cudaDeviceSynchronize();

  cudaMemcpy(image_d, image, width * height * sizeof(unsigned char),
             cudaMemcpyHostToDevice);
  cudaMemset(bins_d, 0, NUM_BINS * sizeof(unsigned int));

  cudaDeviceSynchronize();

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);

  unsigned int numThreadsPerBlock = BLOCK_SIZE;

  unsigned int numBlocks =
      (width * height + (numThreadsPerBlock * COARSE_FACTOR) - 1) /
      (numThreadsPerBlock * COARSE_FACTOR);

  histCoarsed_kernel<<<numBlocks, numThreadsPerBlock>>>(image_d, bins_d,
                                                        width, height);

  cudaDeviceSynchronize();
  cudaEventRecord(stop);
  float milliseconds = 0;
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&milliseconds, start, stop);
  printf("Time taken by Coarsed GPU kernel: %f ms\n", milliseconds);
  cudaMemcpy(bins, bins_d, NUM_BINS * sizeof(unsigned int),
             cudaMemcpyDeviceToHost);

  cudaFree(image_d);
  cudaFree(bins_d);
}

void histogram_gpu(unsigned char *image, unsigned int *bins, unsigned int width,
                   unsigned int height) {

  unsigned char *image_d;
  unsigned int *bins_d;
  cudaMalloc(&image_d, width * height * sizeof(unsigned char));
  cudaMalloc(&bins_d, NUM_BINS * sizeof(unsigned int));

  cudaDeviceSynchronize();

  cudaMemcpy(image_d, image, width * height * sizeof(unsigned char),
             cudaMemcpyHostToDevice);
  cudaMemset(bins_d, 0, NUM_BINS * sizeof(unsigned int));

  cudaDeviceSynchronize();

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);

  unsigned int numThreadsPerBlock = BLOCK_SIZE;

  unsigned int numBlocks =
      (width * height + numThreadsPerBlock - 1) / numThreadsPerBlock;

  histogram_kernel<<<numBlocks, numThreadsPerBlock>>>(image_d, bins_d, width,
                                                      height);

  cudaDeviceSynchronize();
  cudaEventRecord(stop);
  float milliseconds = 0;
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&milliseconds, start, stop);
  printf("Time taken by GPU kernel: %f ms\n", milliseconds);
  cudaMemcpy(bins, bins_d, NUM_BINS * sizeof(unsigned int),
             cudaMemcpyDeviceToHost);

  cudaFree(image_d);
  cudaFree(bins_d);
}

int main() {
  unsigned int width = 8192;
  unsigned int height = 8192;

  unsigned char *image = (unsigned char *)malloc(width * height * sizeof(char));
  unsigned int bins_cpu[NUM_BINS] = {0};
  unsigned int bins_gpu[NUM_BINS] = {0};
  unsigned int bins_priv[NUM_BINS] = {0};
  unsigned int bins_coar[NUM_BINS] = {0};

  for (int i = 0; i < width * height; ++i) {
    image[i] = (int)(rand() / (float)RAND_MAX * NUM_BINS);
  }

  time_t start_cpu, stop_cpu;
  start_cpu = clock();
  for (int i = 0; i < width * height; ++i) {
    bins_cpu[image[i]]++;
  }
  stop_cpu = clock();
  printf("CPU: %f s\n", ((double)(stop_cpu - start_cpu)) / CLOCKS_PER_SEC);
  start_cpu = clock();
  histogram_gpu(image, bins_gpu, width, height);
  cudaDeviceSynchronize();
  stop_cpu = clock();
  printf("GPU: %f s\n", ((double)(stop_cpu - start_cpu)) / CLOCKS_PER_SEC);
  start_cpu = clock();
  histPrivatized_gpu(image, bins_priv, width, height);
  stop_cpu = clock();
  printf("GPU privatized: %f s\n",
         ((double)(stop_cpu - start_cpu)) / CLOCKS_PER_SEC);
  start_cpu = clock();
  histCoarsed_gpu(image, bins_coar, width, height);
  stop_cpu = clock();
  printf("GPU coarsed: %f s\n",
         ((double)(stop_cpu - start_cpu)) / CLOCKS_PER_SEC);
  for (unsigned int i = 0; i < NUM_BINS; ++i) {
    if (bins_cpu[i] != bins_gpu[i] || bins_cpu[i] != bins_priv[i] ||
        bins_cpu[i] != bins_coar[i]) {
      printf("Mismatch at %u: CPU %u, GPU %u, privatized %u, coarsed %u\n", i,
             bins_cpu[i], bins_gpu[i], bins_priv[i], bins_coar[i]);
      break;
    }
  }
}
