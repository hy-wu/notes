#include <algorithm>
#include <bits/stdc++.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

__global__ void vecAdd(const float *__restrict__ A, const float *__restrict__ B,
                       float *__restrict__ C, int N) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < N) {
    C[i] = A[i] + B[i];
  }
}

// 使用 float4 进行向量化访存 (Vectorized Memory Access)
__global__ void vecAddFloat4(const float4 *__restrict__ A, const float4 *__restrict__ B,
                             float4 *__restrict__ C, int N4) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < N4) {
    float4 a = A[i];
    float4 b = B[i];
    C[i] = make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
  }
}

void vecAddGPU(const float *A, const float *B, float *C, int N,
               unsigned int numStreams = 1) {
  float *A_d, *B_d, *C_d;
  size_t size = N * sizeof(float);

  // Allocate memory on the device
  cudaMalloc(&A_d, size);
  cudaMalloc(&B_d, size);
  cudaMalloc(&C_d, size);

  // Setup streams
  if (numStreams > 1) {
    printf("Using %u CUDA streams ", numStreams);
    cudaStream_t streams[numStreams];
    for (unsigned int s = 0; s < numStreams; s++) {
      cudaStreamCreate(&streams[s]);
    }
    unsigned int segmentSize = (N + numStreams - 1) / numStreams;
    for (unsigned int s = 0; s < numStreams; s++) {
      unsigned int startSeg = s * segmentSize;
      unsigned int endSeg = std::min(startSeg + segmentSize, (unsigned int)N);
      unsigned int nSeg = endSeg - startSeg;
      unsigned int sizeSeg = nSeg * sizeof(float);

      // Copy data from host to device asynchronously
      cudaMemcpyAsync(&A_d[startSeg], &A[startSeg], sizeSeg,
                      cudaMemcpyHostToDevice, streams[s]);
      cudaMemcpyAsync(&B_d[startSeg], &B[startSeg], sizeSeg,
                      cudaMemcpyHostToDevice, streams[s]);

      cudaMemsetAsync(&C_d[startSeg], 0, sizeSeg, streams[s]);

      int blockSize = 512;
      int numBlocks = (nSeg + blockSize - 1) / blockSize;

      vecAdd<<<numBlocks, blockSize, 0, streams[s]>>>(
          A_d + startSeg, B_d + startSeg, C_d + startSeg, nSeg);
      cudaMemcpyAsync(&C[startSeg], &C_d[startSeg], sizeSeg,
                      cudaMemcpyDeviceToHost, streams[s]);
    }
  } else {
    printf("Using default stream ");

    // Copy data from host to device
    cudaMemcpy(A_d, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B, size, cudaMemcpyHostToDevice);

    // 消除 GPU 显存延迟分配 (Lazy Allocation) 带来的 Page Fault 开销
    cudaMemset(C_d, 0, size);

    int blockSize = 512;
    int numBlocks = (N + blockSize - 1) / blockSize;

    vecAdd<<<numBlocks, blockSize>>>(A_d, B_d, C_d, N);
    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    vecAdd<<<numBlocks, blockSize>>>(A_d, B_d, C_d, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
      printf("  -> Kernel time: %f ms\n", milliseconds);

    // Copy result back to host
    cudaMemcpy(C, C_d, size, cudaMemcpyDeviceToHost);
  }

  // Free device memory
  cudaFree(A_d);
  cudaFree(B_d);
  cudaFree(C_d);
}

void vecAddGPU_Float4(const float *A, const float *B, float *C, int N) {
  float4 *A_d, *B_d, *C_d;
  size_t size = N * sizeof(float);
  // 因为每个 float4 包含 4 个 float，线程总数变为原来的 1/4
  int N4 = N / 4; 

  cudaMalloc(&A_d, size);
  cudaMalloc(&B_d, size);
  cudaMalloc(&C_d, size);

  // 这里为了纯粹对比 Kernel 性能，我们依旧使用 Memcpy
  cudaMemcpy(A_d, (const float4*)A, size, cudaMemcpyHostToDevice);
  cudaMemcpy(B_d, (const float4*)B, size, cudaMemcpyHostToDevice);
  cudaMemset(C_d, 0, size);

  int blockSize = 256;
  int numBlocks = (N4 + blockSize - 1) / blockSize;

  // Warm-up
  vecAddFloat4<<<numBlocks, blockSize>>>(A_d, B_d, C_d, N4);
  cudaDeviceSynchronize();

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);
  vecAddFloat4<<<numBlocks, blockSize>>>(A_d, B_d, C_d, N4);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  printf("  -> Kernel time (float4): %f ms\n", milliseconds);

  cudaMemcpy((float4*)C, C_d, size, cudaMemcpyDeviceToHost);

  cudaFree(A_d);
  cudaFree(B_d);
  cudaFree(C_d);
}

void vecAddCPU(const float *__restrict__ A, const float *__restrict__ B,
               float *__restrict__ C, int N) {
  for (int i = 0; i < N; i++) {
    C[i] = A[i] + B[i];
  }
}

void vecAddStd(const float *A, const float *B, float *C, int N) {
  std::transform(A, A + N, B, C, std::plus<float>());
}

int main() {
  int N = 1 << 24;
  size_t size = N * sizeof(float);

  float *A = (float *)malloc(size);
  float *B = (float *)malloc(size);
  float *C0 = (float *)malloc(size);
  float *C1 = (float *)malloc(size);
  float *C2 = (float *)malloc(size);
  float *C0p = (float *)malloc(size);
  float *C1p = (float *)malloc(size);
  float *C2p = (float *)malloc(size);
  float *C0s = (float *)malloc(size);
  float *C0ps = (float *)malloc(size);

  for (int i = 0; i < N; i++) {
    A[i] = rand() / (float)RAND_MAX * 100.0f;
    B[i] = rand() / (float)RAND_MAX * 100.0f;
  }
  printf("Vector addition of %d elements\n", N);

  // 预先触发 CPU 的缺页中断 (Page Fault)，强制分配物理内存
  memset(C0, 0, size);
  memset(C1, 0, size);
  memset(C2, 0, size);
  memset(C0p, 0, size);
  memset(C1p, 0, size);
  memset(C2p, 0, size);
  memset(C0s, 0, size);
  memset(C0ps, 0, size);
  float *C_float4 = (float *)malloc(size);


  auto start = std::chrono::high_resolution_clock::now();
  vecAddGPU(A, B, C0, N);
  auto end = std::chrono::high_resolution_clock::now();
  double t = std::chrono::duration<double, std::milli>(end - start).count();
  printf("Total time taken by GPU: %f ms\n", t);

  unsigned int numStreams = 4;
  start = std::chrono::high_resolution_clock::now();
  vecAddGPU(A, B, C0s, N, numStreams);
  end = std::chrono::high_resolution_clock::now();
  t = std::chrono::duration<double, std::milli>(end - start).count();
  printf("Total time taken by GPU with %u streams: %f ms\n", numStreams, t);

  float *A_pinned, *B_pinned, *C_pinned;

  cudaMallocHost(&A_pinned, size);
  cudaMallocHost(&B_pinned, size);
  cudaMallocHost(&C_pinned, size);

  for (int i = 0; i < N; i++) {
    A_pinned[i] = A[i];
    B_pinned[i] = B[i];
  }

  start = std::chrono::high_resolution_clock::now();
  vecAddGPU_Float4(A_pinned, B_pinned, C_float4, N);
  end = std::chrono::high_resolution_clock::now();
  t = std::chrono::duration<double, std::milli>(end - start).count();
  printf("If using pinned memory + float4 kernel: %f ms\n", t);

  start = std::chrono::high_resolution_clock::now();
  vecAddGPU(A_pinned, B_pinned, C_pinned, N);
  end = std::chrono::high_resolution_clock::now();
  t = std::chrono::duration<double, std::milli>(end - start).count();
  printf("If using pinned memory: %f ms\n", t);
  for (int i = 0; i < N; i++) {
    C0p[i] = C_pinned[i];
  }
  cudaMemset(C_pinned, 0, size);

  start = std::chrono::high_resolution_clock::now();
  vecAddGPU(A_pinned, B_pinned, C_pinned, N, numStreams);
  end = std::chrono::high_resolution_clock::now();
  t = std::chrono::duration<double, std::milli>(end - start).count();
  printf("Total time taken by GPU with %u streams: %f ms\n", numStreams, t);
  for (int i = 0; i < N; i++) {
    C0ps[i] = C_pinned[i];
  }
  cudaMemset(C_pinned, 0, size);


  start = std::chrono::high_resolution_clock::now();
  vecAddCPU(A, B, C1, N);
  end = std::chrono::high_resolution_clock::now();
  t = std::chrono::duration<double, std::milli>(end - start).count();
  printf("Total time taken by CPU: %f ms\n", t);

  start = std::chrono::high_resolution_clock::now();
  vecAddCPU(A_pinned, B_pinned, C_pinned, N);
  end = std::chrono::high_resolution_clock::now();
  t = std::chrono::duration<double, std::milli>(end - start).count();
  printf("If using pinned memory: %f ms\n", t);
  for (int i = 0; i < N; i++) {
    C1p[i] = C_pinned[i];
  }
  cudaMemset(C_pinned, 0, size);

  start = std::chrono::high_resolution_clock::now();
  vecAddStd(A, B, C2, N);
  end = std::chrono::high_resolution_clock::now();
  t = std::chrono::duration<double, std::milli>(end - start).count();
  printf("Total time taken by std::transform: %f ms\n", t);

  start = std::chrono::high_resolution_clock::now();
  vecAddStd(A_pinned, B_pinned, C_pinned, N);
  end = std::chrono::high_resolution_clock::now();
  t = std::chrono::duration<double, std::milli>(end - start).count();
  printf("If using pinned memory: %f ms\n", t);
  for (int i = 0; i < N; i++) {
    C2p[i] = C_pinned[i];
  }
  cudaMemset(C_pinned, 0, size);

  for (int i = 0; i < N; i++) {
    if (abs(C0[i] - (A[i] + B[i])) > 1e-8 ||
        abs(C0s[i] - (A[i] + B[i])) > 1e-8 ||
        abs(C0ps[i] - (A[i] + B[i])) > 1e-8 ||
        abs(C1[i] - (A[i] + B[i])) > 1e-8 ||
        abs(C2[i] - (A[i] + B[i])) > 1e-8 ||
        abs(C0p[i] - (A[i] + B[i])) > 1e-8 ||
        abs(C1p[i] - (A[i] + B[i])) > 1e-8 ||
        abs(C2p[i] - (A[i] + B[i])) > 1e-8 ||
        abs(C_float4[i] - (A[i] + B[i])) > 1e-8) {
      printf("Error at index %d: C0=%f, C0s=%f, C0ps=%f, C1=%f, C2=%f, C0p=%f, C1p=%f, C2p=%f, "
             "expected=%f\n",
             i, C0[i], C0s[i], C0ps[i], C1[i], C2[i], C0p[i], C1p[i], C2p[i], A[i] + B[i]);
      break;
    }
  }

  free(A);
  free(B);
  free(C0);
  free(C1);
  free(C2);
  free(C0p);
  free(C1p);
  free(C2p);
  free(C0s);
  free(C0ps);
  free(C_float4);

  cudaFreeHost(A_pinned);
  cudaFreeHost(B_pinned);
  cudaFreeHost(C_pinned);

  return 0;
}
