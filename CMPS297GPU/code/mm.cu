
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda_profiler_api.h>
#include <time.h>

#define TILE_DIM 32
#define COARSE_FACTOR 4

#define bM 128
#define bN 128
#define bK 8
#define tM 8
#define tN 8
#define NUM_THREADS_PER_BLOCK__TILEMNK 256

__global__ void matrixMulByTileCoarsed(float *A, float *B, float *C, unsigned int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int colStart = blockIdx.x * blockDim.x * COARSE_FACTOR + threadIdx.x;

    __shared__ float A_s[TILE_DIM][TILE_DIM];
    __shared__ float B_s[TILE_DIM][TILE_DIM];

    float sum[COARSE_FACTOR] = {0.0f};

    for (unsigned int tile = 0; tile < N/TILE_DIM; ++tile) {
        A_s[threadIdx.y][threadIdx.x] = A[row * N + tile * TILE_DIM + threadIdx.x];
        for (unsigned int c = 0; c < COARSE_FACTOR; ++c) {
            unsigned int col = colStart + c * TILE_DIM;
            B_s[threadIdx.y][threadIdx.x] = B[(tile * TILE_DIM + threadIdx.y) * N + col];
            __syncthreads();
            for (unsigned int i = 0; i < TILE_DIM; ++i) {
                sum[c] += A_s[threadIdx.y][i] * B_s[i][threadIdx.x];
            }
            __syncthreads();
        }
    }
    for (unsigned int c = 0; c < COARSE_FACTOR; ++c) {
        unsigned int col = colStart + c * TILE_DIM;
        if (col < N) {
            C[row * N + col] = sum[c];
        }
    }
}

void matrixMultiplyByTileCoarsed(float *A, float *B, float *C, unsigned int N) {
    float *d_A, *d_B, *d_C;
    size_t size = N * N * sizeof(float);

    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);

    dim3 blockSize(TILE_DIM, TILE_DIM);
    dim3 gridSize((N + TILE_DIM * COARSE_FACTOR - 1) / (TILE_DIM * COARSE_FACTOR), (N + TILE_DIM - 1) / TILE_DIM);
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    matrixMulByTileCoarsed<<<gridSize, blockSize>>>(d_A, d_B, d_C, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Coarsened tiled matrix multiplication kernel time: %f milliseconds\n", milliseconds);
    cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

__global__ void matrixMulByTile(float *A, float *B, float *C, unsigned int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ float A_s[TILE_DIM][TILE_DIM];
    __shared__ float B_s[TILE_DIM][TILE_DIM];

    float sum = 0.0f;

    for (unsigned int tile = 0; tile < N/TILE_DIM; ++tile) {
        A_s[threadIdx.y][threadIdx.x] = A[row * N + tile * TILE_DIM + threadIdx.x];
        B_s[threadIdx.y][threadIdx.x] = B[(tile * TILE_DIM + threadIdx.y) * N + col];
        __syncthreads();
        for (unsigned int i = 0; i < TILE_DIM; ++i) {
            sum += A_s[threadIdx.y][i] * B_s[i][threadIdx.x];
        }
        __syncthreads();
    }
    C[row * N + col] = sum;
}

__device__ __forceinline__ void clear(float C_r[][tN], unsigned int m, unsigned int n) {
    #pragma unroll
    for (unsigned int i = 0; i < tM; ++i) {
        #pragma unroll
        for (unsigned int j = 0; j < tN; ++j) {
            if (i < m && j < n) {
                C_r[i][j] = 0.0f;
            }
        }
    }
}

__device__ __forceinline__ void loadTile(float *A, unsigned int lda, unsigned int maxRow, unsigned int maxCol, float *A_s, unsigned int ldas, unsigned int height, unsigned int width) {
    unsigned int rowPerSubTile = NUM_THREADS_PER_BLOCK__TILEMNK / width;
    unsigned int numSubTiles = height / rowPerSubTile;
    #pragma unroll
    for (unsigned int subTile = 0; subTile < numSubTiles; ++subTile) {
        unsigned int row = subTile * rowPerSubTile + threadIdx.x / width;
        unsigned int col = threadIdx.x % width;
        if (row < maxRow && col < maxCol) {
            A_s[row * ldas + col] = A[row * lda + col];
        }
        else {
            A_s[row * ldas + col] = 0.0f;
        }
    }
}

__device__ __forceinline__ void mm(unsigned int m, unsigned int n, unsigned int k, float *A_s, unsigned int lda_s, float *B_s, unsigned int ldb_s, float C_r[][tN]) {
    #pragma unroll
    for (unsigned int i = 0; i < m; ++i) {
        #pragma unroll
        for (unsigned int j = 0; j < n; ++j) {
            float sum = 0.0f;
            #pragma unroll
            for (unsigned int p = 0; p < k; ++p) {
                sum += A_s[i * lda_s + p] * B_s[p * ldb_s + j];
            }
            C_r[i][j] += sum;
        }
    }
}

__device__ __forceinline__ void writeTile(float *C, unsigned int ldc, unsigned int maxRow, unsigned int maxCol, float C_r[][tN], unsigned int m, unsigned int n) {
    #pragma unroll
    for (unsigned int row = 0; row < m; ++row) {
        #pragma unroll
        for (unsigned int col = 0; col < n; ++col) {
            if (row < maxRow && col < maxCol) {
                C[row * ldc + col] = C_r[row][col];
            }
        }
    }
}


__global__ void matrixMulByTileMNK(float *A, float *B, float *C, unsigned int M, unsigned int N, unsigned int K) {
    // Identify the block's tile
    unsigned int bRow = blockIdx.y * bM;
    unsigned int bCol = blockIdx.x * bN;

    // Identify the thread's tile within the block
    unsigned int tilesPerBlockX = bN / tN;
    unsigned int ty = threadIdx.x / tilesPerBlockX;
    unsigned int tx = threadIdx.x % tilesPerBlockX;
    unsigned int tRow = ty * tM;
    unsigned int tCol = tx * tN;


    float C_r[tM][tN];
    clear(C_r, tM, tN);

    __shared__ float A_s[bM*bK];
    __shared__ float B_s[bK*bN];
    for (unsigned int tile = 0; tile < (K + bK - 1) / bK; ++tile) {
        // load 128x8 elements of A each tile but only 256 threads per block
        loadTile(&A[bRow * N + tile * bK], K, M - bRow, K - tile * bK, &A_s[0], bK, bM, bK);
        loadTile(&B[tile * bK * N + bCol], N, K - tile * bK, N - bCol, &B_s[0], bN, bK, bN);
        __syncthreads();
        mm(tM, tN, bK, &A_s[tRow * bK], bK, &B_s[tCol], bN, C_r);
        __syncthreads();
    }

    float *C_base = &C[(bRow + tRow) * N + bCol + tCol];
    unsigned int maxRow = (bRow + tRow < M) ? (M - bRow - tRow) : 0;
    unsigned int maxCol = (bCol + tCol < N) ? (N - bCol - tCol) : 0;
    writeTile(C_base, N, maxRow, maxCol, C_r, tM, tN);    
}

__global__ void matrixMulSimple(float *A, float *B, float *C, unsigned int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    float sum = 0.0f;
    for (unsigned int k = 0; k < N; ++k) {
        sum += A[row * N + k] * B[k * N + col];
    }
    C[row * N + col] = sum;
}

void matrixMultiplyByTile(float *A, float *B, float *C, unsigned int N) {
    float *d_A, *d_B, *d_C;
    size_t size = N * N * sizeof(float);

    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);

    dim3 blockSize(TILE_DIM, TILE_DIM);
    dim3 gridSize((N + TILE_DIM - 1) / TILE_DIM, (N + TILE_DIM - 1) / TILE_DIM);
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    matrixMulByTile<<<gridSize, blockSize>>>(d_A, d_B, d_C, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Tiled matrix multiplication kernel time: %f milliseconds\n", milliseconds);

    cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

void matrixMultiplySimple(float *A, float *B, float *C, unsigned int N) {
    float *d_A, *d_B, *d_C;
    size_t size = N * N * sizeof(float);

    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);

    dim3 blockSize(TILE_DIM, TILE_DIM);
    dim3 gridSize((N + TILE_DIM - 1) / TILE_DIM, (N + TILE_DIM - 1) / TILE_DIM);
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    matrixMulSimple<<<gridSize, blockSize>>>(d_A, d_B, d_C, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Simple matrix multiplication kernel time: %f milliseconds\n", milliseconds);
    cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

void matrixMultiplyCPU(float *A, float *B, float *C, unsigned int N) {
    for (unsigned int i = 0; i < N; ++i) {
        for (unsigned int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (unsigned int k = 0; k < N; ++k) {
                sum += A[i * N + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

void matrixMultiplyTiledCPU(float *A, float *B, float *C, unsigned int N) {
    for (unsigned int rowTile = 0; rowTile < N/TILE_DIM; rowTile++) {
        for (unsigned int colTile = 0; colTile < N/TILE_DIM; colTile++) {
            for (unsigned int iTile = 0; iTile < N/TILE_DIM; iTile++) {
                for (unsigned int row = rowTile * TILE_DIM; row < (rowTile + 1) * TILE_DIM; ++row) {
                    for (unsigned int col = colTile * TILE_DIM; col < (colTile + 1) * TILE_DIM; ++col) {
                        float sum = 0.0f;
                        for (unsigned int k = 0; k < TILE_DIM; ++k) {
                            sum += A[row * N + (iTile * TILE_DIM + k)] *
                                   B[(iTile * TILE_DIM + k) * N + col];
                        }
                        if (iTile == 0) {
                            C[row * N + col] = sum;
                        } else {
                            C[row * N + col] += sum;
                        }
                    }
                }
            }
        }
    }
}

int main() {
    const unsigned int N = 8192; // Matrix size (N x N)
    float *A = (float*)malloc(N * N * sizeof(float));
    float *B = (float*)malloc(N * N * sizeof(float));
    float *C = (float*)malloc(N * N * sizeof(float));
    float *C0 = (float*)malloc(N * N * sizeof(float));
    float *C1 = (float*)malloc(N * N * sizeof(float));
    float *C2 = (float*)malloc(N * N * sizeof(float));
    float *C3 = (float*)malloc(N * N * sizeof(float));

    // Initialize A and B with random values
    for (unsigned int i = 0; i < N * N; ++i) {
        A[i] = static_cast<float>(rand()) / RAND_MAX;
        B[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    // time the matrix multiplication
    clock_t start = clock();
    matrixMultiplyByTile(A, B, C1, N);
    clock_t end = clock();
    double time = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("Tiled matrix multiplication time: %f seconds\n", time);
    start = clock();
    matrixMultiplyByTileCoarsed(A, B, C0, N);
    end = clock();
    time = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("Coarsened tiled matrix multiplication time: %f seconds\n", time);
    clock_t start_simple = clock();
    matrixMultiplySimple(A, B, C2, N);
    clock_t end_simple = clock();
    double time_simple = ((double)(end_simple - start_simple)) / CLOCKS_PER_SEC;
    printf("Simple matrix multiplication time: %f seconds\n", time_simple);
    // clock_t start_cpu = clock();
    // matrixMultiplyCPU(A, B, C, N);
    // clock_t end_cpu = clock();
    // double time_cpu = ((double)(end_cpu - start_cpu)) / CLOCKS_PER_SEC;
    // printf("CPU matrix multiplication time: %f seconds\n", time_cpu);
    // clock_t start_tiled_cpu = clock();
    // matrixMultiplyTiledCPU(A, B, C3, N);
    // clock_t end_tiled_cpu = clock();
    // double time_tiled_cpu = ((double)(end_tiled_cpu - start_tiled_cpu)) / CLOCKS_PER_SEC;
    // printf("Tiled CPU matrix multiplication time: %f seconds\n", time_tiled_cpu);
    for (unsigned int i = 0; i < N * N; ++i) {
    //     if (abs(C[i] - C1[i]) > 1e-4 || abs(C[i] - C2[i]) > 1e-4 || abs(C[i] - C3[i]) > 1e-4) {
    //         printf("Mismatch at index %u: CPU=%f, Tiled GPU=%f, Simple GPU=%f, Tiled CPU=%f\n", i, C[i], C1[i], C2[i], C3[i]);
        if (abs(C1[i] - C2[i]) > 1e-4 || abs(C1[i] - C0[i]) > 1e-4) {
            printf("Mismatch at index %u: Tiled GPU=%f, Simple GPU=%f, Coarsened Tiled GPU=%f\n", i, C1[i], C2[i], C0[i]);
            break;
        }
    }

    free(A);
    free(B);
    free(C1);
    free(C2);
    free(C3);
    free(C);

    cudaProfilerStop();

    return 0;
}

