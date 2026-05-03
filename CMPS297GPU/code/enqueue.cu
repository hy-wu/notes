
#include <algorithm>
#include <chrono>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define BLOCK_DIM 1024

#define WARP_SIZE 32

#define N_REPEATS 1

__host__ __device__ bool cond(unsigned int value) {
  // Replace with your actual condition
  return (value % 2) == 0;
}

__global__ void enqueue_kernel(unsigned int *input, unsigned int *queue,
                               unsigned int N, unsigned int *queueSize) {
  unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) {
    unsigned int value = input[idx];
    if (cond(value)) {
      unsigned int pos = atomicAdd(queueSize, 1);
      queue[pos] = value;
    }
  }
}

__global__ void enq_vote_kernel(unsigned int *input, unsigned int *queue,
                                unsigned int N, unsigned int *queueSize) {
  unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) {
    unsigned int value = input[idx];
    unsigned int is_active = cond(value);
    unsigned int a = __ballot_sync(0xffffffff, is_active);
    if (is_active) {
      // assign a leader thread
      unsigned int leader = __ffs(a) - 1; // find the first active thread
      unsigned int numActive = __popc(a); // count active threads
      unsigned int j;                     // the starting index
      if (threadIdx.x % WARP_SIZE == leader) {
        j = atomicAdd(queueSize, numActive);
      }
      j = __shfl_sync(a, j, leader);
      // find the position of each active thread in the queue
      unsigned int previousThreads = (1u << (threadIdx.x % WARP_SIZE)) - 1;
      unsigned int previousActiveThreads = a & previousThreads;
      unsigned int offset = __popc(previousActiveThreads);
      // store the value
      queue[j + offset] = value;
    }
  }
}

unsigned int enqueue_gpu(unsigned int *input, unsigned int *queue,
                         unsigned int N, const bool voting) {
  unsigned int *input_d, *queue_d, *queueSize_d;
  cudaMalloc(&input_d, N * sizeof(unsigned int));
  cudaMalloc(&queue_d, N * sizeof(unsigned int));
  cudaMalloc(&queueSize_d, sizeof(unsigned int));
  cudaDeviceSynchronize();
  cudaMemcpy(input_d, input, N * sizeof(unsigned int), cudaMemcpyHostToDevice);
  cudaMemset(queue_d, 0, N * sizeof(unsigned int));
  cudaMemset(queueSize_d, 0, sizeof(unsigned int));

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);
  unsigned int numBlocks = (N + BLOCK_DIM - 1) / BLOCK_DIM;
  if (voting) {
    enq_vote_kernel<<<numBlocks, BLOCK_DIM>>>(input_d, queue_d, N, queueSize_d);
  } else {
    enqueue_kernel<<<numBlocks, BLOCK_DIM>>>(input_d, queue_d, N, queueSize_d);
  }
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  printf("Enqueue kernel (%s) Time: %8.3f ms\n", voting ? "voting" : "simple",
         milliseconds);
  unsigned int queueSize;
  cudaMemcpy(&queueSize, queueSize_d, sizeof(unsigned int),
             cudaMemcpyDeviceToHost);
  cudaMemcpy(queue, queue_d, queueSize * sizeof(unsigned int),
             cudaMemcpyDeviceToHost);
  cudaFree(input_d);
  cudaFree(queue_d);
  cudaFree(queueSize_d);
  return queueSize;
}

unsigned int enqueue_cpu(unsigned int *input, unsigned int *queue,
                         unsigned int N) {
  unsigned int queueSize = 0;
  for (unsigned int i = 0; i < N; i++) {
    if (cond(input[i])) {
      queue[queueSize++] = input[i];
    }
  }
  return queueSize;
}

unsigned int enqueue_cpu_std(unsigned int *input, unsigned int *queue,
                             unsigned int N) {
  auto end_it = std::copy_if(input, input + N, queue,
                             [](unsigned int v) { return cond(v); });
  return std::distance(queue, end_it);
}

int main() {
  // 预热 CUDA 上下文，消除首次调用的高昂开销
  cudaFree(0);

  unsigned int N = 1 << 24; // Example size
  unsigned int *input = (unsigned int *)malloc(N * sizeof(unsigned int));
  unsigned int *queue = (unsigned int *)malloc(N * sizeof(unsigned int));
  unsigned int *queue0 = (unsigned int *)malloc(N * sizeof(unsigned int));
  unsigned int *queue1 = (unsigned int *)malloc(N * sizeof(unsigned int));
  unsigned int *queue2 = (unsigned int *)malloc(N * sizeof(unsigned int));
  for (unsigned int i = 0; i < N; i++) {
    input[i] = rand();
  }

  auto t_start = std::chrono::high_resolution_clock::now();
  unsigned int queueSize = enqueue_cpu(input, queue, N);
  auto t_end = std::chrono::high_resolution_clock::now();
  double t_total = std::chrono::duration<double, std::milli>(t_end - t_start).count();
  printf("CPU serial Enqueue Time: %8.3f ms | Queue Size: %u\n", t_total,
         queueSize);
  std::sort(queue, queue + queueSize); // Sort CPU result for easier comparison

  for (unsigned int iter = 0; iter < N_REPEATS; iter++) {
    t_start = std::chrono::high_resolution_clock::now();
    unsigned int queueSizeStd = enqueue_cpu_std(input, queue0, N);
    t_end = std::chrono::high_resolution_clock::now();
    double t_total_std = std::chrono::duration<double, std::milli>(t_end - t_start).count();
    printf("CPU std::copy_if Time:   %8.3f ms | Queue Size: %u\n", t_total_std,
           queueSizeStd);
    std::sort(
        queue0,
        queue0 +
            queueSizeStd); // Sort CPU std::copy_if result for easier comparison

    if (queueSize != queueSizeStd) {
      fprintf(stderr,
              "Error: Queue sizes do not match between CPU implementations! "
              "CPU: %u, CPU std::copy_if: %u\n",
              queueSize, queueSizeStd);
    } else {
      std::sort(queue0, queue0 + queueSizeStd); // Sort CPU std::copy_if result
                                                // for easier comparison
      bool match = true;
      for (unsigned int i = 0; i < queueSize; i++) {
        if (queue[i] != queue0[i]) {
          fprintf(
              stderr,
              "Error: Queue contents do not match between CPU implementations "
              "at index %u! CPU: %u, CPU std::copy_if: %u\n",
              i, queue[i], queue0[i]);
          match = false;
          break;
        }
      }
      if (match) {
        printf("CPU queues match!\n");
      }
    }
  }

  for (int i = 0; i < N_REPEATS; i++) {
    t_start = std::chrono::high_resolution_clock::now();
    unsigned int queueSizeSimple = enqueue_gpu(input, queue1, N, false);
    t_end = std::chrono::high_resolution_clock::now();
    t_total = std::chrono::duration<double, std::milli>(t_end - t_start).count();
    printf("Simple Enqueue Time: %8.3f ms\n", t_total);
    t_start = std::chrono::high_resolution_clock::now();
    unsigned int queueSizeVoting = enqueue_gpu(input, queue2, N, true);
    t_end = std::chrono::high_resolution_clock::now();
    t_total = std::chrono::duration<double, std::milli>(t_end - t_start).count();
    printf("Voting Enqueue Time: %8.3f ms\n", t_total);

    if (queueSizeSimple != queueSize || queueSizeVoting != queueSize) {
      fprintf(stderr,
              "Error: Queue sizes do not match! CPU: %u, GPU Simple: %u, GPU "
              "Voting: %u\n",
              queueSize, queueSizeSimple, queueSizeVoting);
    } else {
      std::sort(queue1, queue1 + queueSize);
      std::sort(queue2, queue2 + queueSize);
      bool match = true;
      for (unsigned int i = 0; i < queueSize; i++) {
        if (queue[i] != queue1[i] || queue[i] != queue2[i]) {
          fprintf(stderr,
                  "Error: Queue contents do not match at index %u! CPU: %u, "
                  "GPU Simple: %u, GPU Voting: %u\n",
                  i, queue[i], queue1[i], queue2[i]);
          match = false;
          break;
        }
      }
      if (match) {
        printf("Queues match!\n");
      }
    }
  }
  free(input);
  free(queue);
  free(queue1);
  free(queue2);
  return 0;
}
