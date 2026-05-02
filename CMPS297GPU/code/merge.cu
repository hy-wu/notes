
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>

#define ELEMENTS_PER_THREAD 6
#define THREADS_PER_BLOCK 256
#define ELEMENTS_PER_BLOCK (ELEMENTS_PER_THREAD * THREADS_PER_BLOCK)


__host__ __device__ int coRank(float *A, float *B, unsigned int m, unsigned int n, unsigned int k) {
  unsigned int iLow = (k > n) ? (k - n) : 0;
  unsigned int iHigh = min(k, m);
  while (true) {
    unsigned int i = (iLow + iHigh) / 2;
    unsigned int j = k - i;
    if (i > 0 && j < n && A[i - 1] > B[j]) {
      iHigh = i - 1;
    } else if (j > 0 && i < m && B[j - 1] > A[i]) {
      iLow = i + 1;
    } else {
      return i;
    }
  }
}

__global__ void merge_kernel(float *A, float *B, float *C, unsigned int m,
                                 unsigned int n) {
  unsigned int k = blockIdx.x * ELEMENTS_PER_BLOCK + threadIdx.x * ELEMENTS_PER_THREAD;
  for (unsigned int i = 0; i < ELEMENTS_PER_THREAD; ++i) {
    if (k + i < m + n) {
      unsigned int co_rank = coRank(A, B, m, n, k + i);
      unsigned int j = k + i - co_rank;
      if (co_rank < m && (j >= n || A[co_rank] <= B[j])) {
        C[k + i] = A[co_rank];
      } else {
        C[k + i] = B[j];
      }
    }
  }
}

__global__ void tiled_merge_kernel(float *A, float *B, float *C, unsigned int m,
                                 unsigned int n) {
  unsigned int kBlock = blockIdx.x * ELEMENTS_PER_BLOCK;
  unsigned int kNextBlock = min(kBlock + ELEMENTS_PER_BLOCK, m + n);
  __shared__ unsigned int iBlock, iNextBlock;
  if (threadIdx.x == 0) {
    iBlock = coRank(A, B, m, n, kBlock);
    iNextBlock = coRank(A, B, m, n, kNextBlock);
  }
  __syncthreads();

  unsigned int jBlock = kBlock - iBlock;
  unsigned int iBlockSize = iNextBlock - iBlock;
  unsigned int jBlockSize = (kNextBlock - kBlock) - iBlockSize;

  __shared__ float A_s[ELEMENTS_PER_BLOCK];
  __shared__ float B_s[ELEMENTS_PER_BLOCK];

  for (unsigned int i = threadIdx.x; i < iBlockSize; i += blockDim.x) {
    A_s[i] = A[iBlock + i];
  }
  for (unsigned int j = threadIdx.x; j < jBlockSize; j += blockDim.x) {
    B_s[j] = B[jBlock + j];
  }
  __syncthreads();

  unsigned int kThread = kBlock + threadIdx.x * ELEMENTS_PER_THREAD;
  for (unsigned int k = 0; k < ELEMENTS_PER_THREAD; ++k) {
    if (kThread + k < m + n) {
      unsigned int co_rank = coRank(A_s, B_s, iBlockSize, jBlockSize, threadIdx.x * ELEMENTS_PER_THREAD + k);
      unsigned int j = threadIdx.x * ELEMENTS_PER_THREAD + k - co_rank;
      if (co_rank < iBlockSize && (j >= jBlockSize || A_s[co_rank] <= B_s[j])) {
        C[kThread + k] = A_s[co_rank];
      } else {
        C[kThread + k] = B_s[j];
      }
    }
  }
}

void merge_gpu(float *A, float *B, float *C, unsigned int m,
               unsigned int n) {
  float *A_d, *B_d, *C_d;
  cudaMalloc(&A_d, m * sizeof(float));
  cudaMalloc(&B_d, n * sizeof(float));
  cudaMalloc(&C_d, (m + n) * sizeof(float));

  cudaMemcpy(A_d, A, m * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(B_d, B, n * sizeof(float), cudaMemcpyHostToDevice);

  unsigned int num_blocks = (m + n + ELEMENTS_PER_BLOCK - 1) / ELEMENTS_PER_BLOCK;
  tiled_merge_kernel<<<num_blocks, THREADS_PER_BLOCK>>>(A_d, B_d, C_d, m, n);

  cudaMemcpy(C, C_d, (m + n) * sizeof(float), cudaMemcpyDeviceToHost);

  cudaFree(A_d);
  cudaFree(B_d);
  cudaFree(C_d);
}


int main() {
  unsigned int m = 1 << 24;
  unsigned int n = 1 << 24;

  float *A = (float *)malloc(m * sizeof(float));
  float *B = (float *)malloc(n * sizeof(float));
  float *C = (float *)malloc((m + n) * sizeof(float));
  float *C_cpu = (float *)malloc((m + n) * sizeof(float));

  for (unsigned int i = 0; i < m; ++i) {
    A[i] = rand() / (float)RAND_MAX;
  }
  for (unsigned int i = 0; i < n; ++i) {
    B[i] = rand() / (float)RAND_MAX;
  }
  // sort A and B
  qsort(A, m, sizeof(float), [](const void *a, const void *b) {
    float fa = *(const float *)a;
    float fb = *(const float *)b;
    return (fa > fb) - (fa < fb);
  });
  qsort(B, n, sizeof(float), [](const void *a, const void *b) {
    float fa = *(const float *)a;
    float fb = *(const float *)b;
    return (fa > fb) - (fa < fb);
  });

  printf("Starting merge on GPU...\n");
  clock_t start = clock();
  merge_gpu(A, B, C, m, n);
  cudaDeviceSynchronize();
  clock_t end = clock();
  double elapsed_time = (double)(end - start) / CLOCKS_PER_SEC;
  printf("Merge completed in %.2f seconds on GPU.\n", elapsed_time);

  // CPU merge
  printf("Starting merge on CPU...\n");
  start = clock();
  // for (unsigned int i = 0; i < m + n; ++i) {
  //   unsigned int co_rank = coRank(A, B, m, n, i);
  //   unsigned int j = i - co_rank;
  //   if (co_rank < m && (j >= n || A[co_rank] <= B[j])) {
  //     C_cpu[i] = A[co_rank];
  //   } else {
  //     C_cpu[i] = B[j];
  //   }
  // }
  unsigned int i = 0, j = 0, k = 0;
  while (i < m && j < n) {
    if (A[i] <= B[j]) {
      C_cpu[k++] = A[i++];
    } else {
      C_cpu[k++] = B[j++];
    }
  }
  while (i < m) {
    C_cpu[k++] = A[i++];
  }
  while (j < n) {
    C_cpu[k++] = B[j++];
  }
  end = clock();
  elapsed_time = (double)(end - start) / CLOCKS_PER_SEC;
  printf("Merge completed in %.2f seconds on CPU.\n", elapsed_time);


  // Verify the result
  for (unsigned int i = 0; i < m + n; ++i) {
    if (C[i] != C_cpu[i]) {
      printf("Error at index %u: expected %f, got %f\n", i, C_cpu[i], C[i]);
      break;
    }
  }

  free(A);
  free(B);
  free(C);
  free(C_cpu);

  return 0;
}
