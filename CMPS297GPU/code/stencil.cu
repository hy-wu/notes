#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <time.h>

#define BLOCK_DIM 8
#define IN_TILE_DIM BLOCK_DIM
#define OUT_TILE_DIM (BLOCK_DIM-2)

#define C0 1.0f
#define C1 0.125f


__global__ void stencil_3d_tile(float *in, float *out, unsigned int N) {
    int i = blockIdx.z * OUT_TILE_DIM + threadIdx.z;
    int j = blockIdx.y * OUT_TILE_DIM + threadIdx.y;
    int k = blockIdx.x * OUT_TILE_DIM + threadIdx.x;

    __shared__ float tile[IN_TILE_DIM][IN_TILE_DIM][IN_TILE_DIM];
    if (i >= 0 && j >= 0 && k >= 0 && i < N && j < N && k < N) {
        tile[threadIdx.z][threadIdx.y][threadIdx.x] = in[i * N * N + j * N + k];
    }
    __syncthreads();

    if (i >= 1 && j >= 1 && k >= 1 && i < N-1 && j < N-1 && k < N-1) {
        if (threadIdx.x >= 1 && threadIdx.x < blockDim.x-1 && threadIdx.y >= 1 && threadIdx.y < blockDim.y-1 && threadIdx.z >= 1 && threadIdx.z < blockDim.z-1) {
            out[i * N * N + j * N + k] = C0 * tile[threadIdx.z][threadIdx.y][threadIdx.x] +
                C1 * (tile[threadIdx.z-1][threadIdx.y][threadIdx.x] +
                tile[threadIdx.z+1][threadIdx.y][threadIdx.x] +
                tile[threadIdx.z][threadIdx.y-1][threadIdx.x] +
                tile[threadIdx.z][threadIdx.y+1][threadIdx.x] +
                tile[threadIdx.z][threadIdx.y][threadIdx.x-1] +
                tile[threadIdx.z][threadIdx.y][threadIdx.x+1]);
        }
    }
}

__global__ void stencil_3d(float *in, float *out, unsigned int N) {
    unsigned int i = blockIdx.z * blockDim.z + threadIdx.z;
    unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= 1 && j >= 1 && k >= 1 && i < N-1 && j < N-1 && k < N-1) {
        out[i * N * N + j * N + k] = C0 * in[i * N * N + j * N + k] +
            C1 * (in[(i-1) * N * N + j * N + k] +
            in[(i+1) * N * N + j * N + k] +
            in[i * N * N + (j-1) * N + k] +
            in[i * N * N + (j+1) * N + k] +
            in[i * N * N + j * N + (k-1)] +
            in[i * N * N + j * N + (k+1)]);
    }
}

__global__ void stencil_3d_coarsen_z(float *in, float *out, unsigned int N) {
    // 线程粗化：在X、Y轴上并行，在Z轴上串行循环
    unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int k = blockIdx.x * blockDim.x + threadIdx.x;

    if (j >= 1 && j < N-1 && k >= 1 && k < N-1) {
        // Register Tiling: 利用寄存器保存Z轴的历史数据
        float bottom = in[0 * N * N + j * N + k];
        float current = in[1 * N * N + j * N + k];
        float top;

        for (unsigned int i = 1; i < N-1; i++) {
            top = in[(i+1) * N * N + j * N + k]; // 每次循环只加载1次Z方向的新数据
            out[i * N * N + j * N + k] = C0 * current +
                C1 * (bottom + top +
                in[i * N * N + (j-1) * N + k] +
                in[i * N * N + (j+1) * N + k] +
                in[i * N * N + j * N + (k-1)] +
                in[i * N * N + j * N + (k+1)]);
            
            bottom = current;
            current = top;
        }
    }
}

void stencil_gpu(float *in, float *out, unsigned int N) {
    float *in_d, *out_d;
    size_t size = N * N * N * sizeof(float);
    cudaMalloc(&in_d, size);
    cudaMalloc(&out_d, size);
    cudaMemcpy(in_d, in, size, cudaMemcpyHostToDevice);
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    dim3 block(BLOCK_DIM, BLOCK_DIM, BLOCK_DIM);
    dim3 grid((N + block.x - 1) / block.x,
                (N + block.y - 1) / block.y,
                (N + block.z - 1) / block.z);
    stencil_3d<<<grid, block>>>(in_d, out_d, N);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Time taken by GPU kernel: %f ms\n", milliseconds);

    cudaMemcpy(out, out_d, size, cudaMemcpyDeviceToHost);
    cudaFree(in_d);
    cudaFree(out_d);
}

void stencil_gpu_tile(float *in, float *out, unsigned int N) {
    float *in_d, *out_d;
    size_t size = N * N * N * sizeof(float);
    cudaMalloc(&in_d, size);
    cudaMalloc(&out_d, size);
    cudaMemcpy(in_d, in, size, cudaMemcpyHostToDevice);
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    dim3 block(BLOCK_DIM, BLOCK_DIM, BLOCK_DIM);
    dim3 grid((N + OUT_TILE_DIM - 1) / OUT_TILE_DIM,
                (N + OUT_TILE_DIM - 1) / OUT_TILE_DIM,
                (N + OUT_TILE_DIM - 1) / OUT_TILE_DIM);
    stencil_3d_tile<<<grid, block>>>(in_d, out_d, N);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Time taken by GPU tiled kernel: %f ms\n", milliseconds);

    cudaMemcpy(out, out_d, size, cudaMemcpyDeviceToHost);
    cudaFree(in_d);
    cudaFree(out_d);
}

void stencil_gpu_coarsen_z(float *in, float *out, unsigned int N) {
    float *in_d, *out_d;
    size_t size = N * N * N * sizeof(float);
    cudaMalloc(&in_d, size);
    cudaMalloc(&out_d, size);
    cudaMemcpy(in_d, in, size, cudaMemcpyHostToDevice);
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // Z轴在核函数内通过循环处理，因此Grid配置为二维即可
    dim3 block(16, 16, 1); 
    dim3 grid((N + block.x - 1) / block.x,
              (N + block.y - 1) / block.y,
              1);
    stencil_3d_coarsen_z<<<grid, block>>>(in_d, out_d, N);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Time taken by GPU coarsened (register tiling) kernel: %f ms\n", milliseconds);

    cudaMemcpy(out, out_d, size, cudaMemcpyDeviceToHost);
    cudaFree(in_d);
    cudaFree(out_d);
}

void stencil_cpu(float *in, float *out, unsigned int N) {
    for (unsigned int i = 1; i < N-1; i++) {
        for (unsigned int j = 1; j < N-1; j++) {
            for (unsigned int k = 1; k < N-1; k++) {
                out[i * N * N + j * N + k] = C0 * in[i * N * N + j * N + k] +
                    C1 * (in[(i-1) * N * N + j * N + k] +
                    in[(i+1) * N * N + j * N + k] +
                    in[i * N * N + (j-1) * N + k] +
                    in[i * N * N + (j+1) * N + k] +
                    in[i * N * N + j * N + (k-1)] +
                    in[i * N * N + j * N + (k+1)]);
            }
        }
    }
}

int main() {
    unsigned int N = 512; // example size
    float *in = (float*)malloc(N * N * N * sizeof(float));
    float *out_cpu = (float*)malloc(N * N * N * sizeof(float));
    float *out_gpu = (float*)malloc(N * N * N * sizeof(float));
    float *out_gpu_tile = (float*)malloc(N * N * N * sizeof(float));
    float *out_gpu_coarsen = (float*)malloc(N * N * N * sizeof(float));
    // Initialize in with some values
    for (unsigned int i = 0; i < N * N * N; i++) {
        in[i] = rand() / (float)RAND_MAX;
    }
    time_t start = clock();
    stencil_cpu(in, out_cpu , N);
    time_t end = clock();
    double cpu_time = double(end - start) / CLOCKS_PER_SEC * 1000.0;
    printf("Time taken by CPU: %f ms\n", cpu_time);
    start = clock();
    stencil_gpu(in, out_gpu, N);
    end = clock();
    double gpu_time = double(end - start) / CLOCKS_PER_SEC * 1000.0;
    printf("Time taken by GPU: %f ms\n", gpu_time);
    start = clock();
    stencil_gpu_tile(in, out_gpu_tile, N);
    cudaDeviceSynchronize();
    end = clock();
    double gpu_tile_time = double(end - start) / CLOCKS_PER_SEC * 1000.0;
    printf("Time taken by GPU tiled: %f ms\n", gpu_tile_time);

    start = clock();
    stencil_gpu_coarsen_z(in, out_gpu_coarsen, N);
    cudaDeviceSynchronize();
    end = clock();
    double gpu_coarsen_time = double(end - start) / CLOCKS_PER_SEC * 1000.0;
    printf("Time taken by GPU coarsened: %f ms\n", gpu_coarsen_time);

    for (unsigned int i = 0; i < N * N * N; i++) {
        if (out_cpu[i] != out_gpu[i]) {
            printf("Mismatch at index %u: CPU %f, GPU %f\n", i, out_cpu[i], out_gpu[i]);
            break;
        }
        if (out_cpu[i] != out_gpu_tile[i]) {
            printf("Mismatch at index %u (%u, %u, %u): CPU %f, GPU tiled %f\n", i, i / (N * N), (i % (N * N)) / N, i % N, out_cpu[i], out_gpu_tile[i]);
            break;
        }
        if (out_cpu[i] != out_gpu_coarsen[i]) {
            printf("Mismatch at index %u (%u, %u, %u): CPU %f, GPU coarsened %f\n", i, i / (N * N), (i % (N * N)) / N, i % N, out_cpu[i], out_gpu_coarsen[i]);
            break;
        }
    }

    free(in);
    free(out_cpu);
    free(out_gpu);
    free(out_gpu_tile);
    free(out_gpu_coarsen);
    return 0;
}
