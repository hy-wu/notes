#include <cstdio>
#include <algorithm>
#include <unordered_set>
#include <cuda_runtime.h>
#include <cusparse.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define THREADS_PER_BLOCK__JDS 1024
#define THREADS_PER_BLOCK__ELL 1024
#define THREADS_PER_BLOCK__CSR 1024
#define THREADS_PER_BLOCK__COO 1024

#define MY_DEBUG 0

struct COOMatrix {
  unsigned int *rowIdxs;
  unsigned int *colIdxs;
  float *values;
  unsigned int numRows;
  unsigned int numCols;
  unsigned int numNonZeros;
};

struct CSRMatrix {
  unsigned int *rowPtrs;
  unsigned int *colIdxs;
  float *values;
  unsigned int numRows;
  unsigned int numCols;
  unsigned int numNonZeros;
};

struct ELLMatrix {
  unsigned int numRows;
  unsigned int numCols;
  unsigned int maxNNZPerRow;
  unsigned int *nnzPerRow;
  unsigned int *colIdxs;
  float *values;
};

struct JDSMatrix {
  unsigned int numRows;
  unsigned int numCols;
  unsigned int numNonZeros;
  unsigned int maxNNZPerRow;
  unsigned int *nnzPerRow;
  unsigned int *rowIdxs;
  unsigned int *iterPtr;
  unsigned int *colIdxs;
  float *values;
};

__global__ void spmv_jds_kernel(JDSMatrix jds, float *inVec, float *outVec) {
  unsigned int r = blockIdx.x * blockDim.x + threadIdx.x;
  if (r < jds.numRows) {
    float sum = 0.0f;
    for (unsigned int i = 0; i < jds.nnzPerRow[r]; ++i) {
      unsigned int idx = jds.iterPtr[i] + r;
      if (idx < jds.numNonZeros) {
        unsigned int col = jds.colIdxs[idx];
        float val = jds.values[idx];
        sum += val * inVec[col];
      }
    }
    outVec[jds.rowIdxs[r]] = sum;
  }
}

void spmv_jds_gpu(JDSMatrix jds, float *inVec, float *outVec) {
  JDSMatrix jds_d;
  cudaMalloc(&jds_d.nnzPerRow, jds.numRows * sizeof(unsigned int));
  cudaMalloc(&jds_d.rowIdxs, jds.numRows * sizeof(unsigned int));
  cudaMalloc(&jds_d.iterPtr, (jds.maxNNZPerRow + 1) * sizeof(unsigned int));
  cudaMalloc(&jds_d.colIdxs, jds.numNonZeros * sizeof(unsigned int));
  cudaMalloc(&jds_d.values, jds.numNonZeros * sizeof(float));
  cudaMemcpy(jds_d.nnzPerRow, jds.nnzPerRow, jds.numRows * sizeof(unsigned int),
             cudaMemcpyHostToDevice);
  cudaMemcpy(jds_d.rowIdxs, jds.rowIdxs, jds.numRows * sizeof(unsigned int),
             cudaMemcpyHostToDevice);
  cudaMemcpy(jds_d.iterPtr, jds.iterPtr,
             (jds.maxNNZPerRow + 1) * sizeof(unsigned int),
             cudaMemcpyHostToDevice);
  cudaMemcpy(jds_d.colIdxs, jds.colIdxs, jds.numNonZeros * sizeof(unsigned int),
             cudaMemcpyHostToDevice);
  cudaMemcpy(jds_d.values, jds.values, jds.numNonZeros * sizeof(float),
             cudaMemcpyHostToDevice);
  jds_d.numRows = jds.numRows;
  jds_d.numCols = jds.numCols;
  jds_d.numNonZeros = jds.numNonZeros;
  jds_d.maxNNZPerRow = jds.maxNNZPerRow;

  float *inVec_d, *outVec_d;
  cudaMalloc(&inVec_d, jds.numCols * sizeof(float));
  cudaMalloc(&outVec_d, jds.numRows * sizeof(float));
  cudaMemcpy(inVec_d, inVec, jds.numCols * sizeof(float),
             cudaMemcpyHostToDevice);
  cudaMemset(outVec_d, 0, jds.numRows * sizeof(float));

  dim3 blockDim(THREADS_PER_BLOCK__JDS);
  unsigned int numBlocks = (jds.numRows + blockDim.x - 1) / blockDim.x;
  spmv_jds_kernel<<<numBlocks, blockDim>>>(jds_d, inVec_d, outVec_d);
  cudaDeviceSynchronize();
  cudaMemcpy(outVec, outVec_d, jds.numRows * sizeof(float),
             cudaMemcpyDeviceToHost);
  if (MY_DEBUG) {
    printf("SpMV (JDS) outVec: ");
    for (unsigned int i = 0; i < jds.numRows; ++i) {
      printf("%.4f\t", outVec[i]);
    }
    printf("\n");
  }
  cudaFree(jds_d.nnzPerRow);
  cudaFree(jds_d.rowIdxs);
  cudaFree(jds_d.iterPtr);
  cudaFree(jds_d.colIdxs);
  cudaFree(jds_d.values);
  cudaFree(inVec_d);
  cudaFree(outVec_d);
}

void ell2jds(ELLMatrix ell, JDSMatrix *jds, bool useLibrarySort = true) {
  jds->numRows = ell.numRows;
  jds->numCols = ell.numCols;
  jds->maxNNZPerRow = ell.maxNNZPerRow;

  jds->nnzPerRow = (unsigned int *)malloc(ell.numRows * sizeof(unsigned int));
  jds->rowIdxs = (unsigned int *)malloc(ell.numRows * sizeof(unsigned int));

  if (useLibrarySort) {
    for (unsigned int i = 0; i < ell.numRows; ++i) {
      jds->rowIdxs[i] = i;
    }
    // 使用 std::sort 和 Lambda 表达式通过比较非零元素数量来降序排列行索引
    std::sort(jds->rowIdxs, jds->rowIdxs + ell.numRows, [&ell](unsigned int a, unsigned int b) {
      return ell.nnzPerRow[a] > ell.nnzPerRow[b];
    });
    for (unsigned int i = 0; i < ell.numRows; ++i) {
      jds->nnzPerRow[i] = ell.nnzPerRow[jds->rowIdxs[i]];
    }
  } else {
    for (unsigned int i = 0; i < ell.numRows; ++i) {
      jds->nnzPerRow[i] = ell.nnzPerRow[i];
      jds->rowIdxs[i] = i;
    }
    // 原来的冒泡排序（降序）
    for (unsigned int i = 0; i < ell.numRows - 1; ++i) {
      for (unsigned int j = 0; j < ell.numRows - i - 1; ++j) {
        if (jds->nnzPerRow[j] < jds->nnzPerRow[j + 1]) {
          // Swap nnzPerRow
          unsigned int tempNNZ = jds->nnzPerRow[j];
          jds->nnzPerRow[j] = jds->nnzPerRow[j + 1];
          jds->nnzPerRow[j + 1] = tempNNZ;
          // Swap rowIdxs
          unsigned int tempIdx = jds->rowIdxs[j];
          jds->rowIdxs[j] = jds->rowIdxs[j + 1];
          jds->rowIdxs[j + 1] = tempIdx;
        }
      }
    }
  }

  jds->iterPtr =
      (unsigned int *)malloc((jds->maxNNZPerRow + 1) * sizeof(unsigned int));
  jds->iterPtr[0] = 0;
  for (unsigned int i = 1; i <= jds->maxNNZPerRow; ++i) {
    jds->iterPtr[i] = jds->iterPtr[i - 1];
    for (unsigned int row = 0; row < ell.numRows; ++row) {
      if (jds->nnzPerRow[row] > i - 1) {
        jds->iterPtr[i]++;
      } else {
        // 因为 nnzPerRow 是降序排列的，一旦遇到小于等于的，后面的必然也都小于等于
        break;
      }
    }
  }

  jds->numNonZeros = jds->iterPtr[jds->maxNNZPerRow];
  printf("JDS numNonZeros: %u\n", jds->numNonZeros);
  jds->colIdxs =
      (unsigned int *)malloc(jds->numNonZeros * sizeof(unsigned int));
  jds->values = (float *)malloc(jds->numNonZeros * sizeof(float));
  unsigned int idx = 0;
  for (unsigned int i = 0; i < jds->maxNNZPerRow; ++i) {
    for (unsigned int row = 0; row < ell.numRows; ++row) {
      if (jds->nnzPerRow[row] > i) {
        unsigned int ellRow = jds->rowIdxs[row];
        jds->colIdxs[idx] = ell.colIdxs[ellRow * ell.maxNNZPerRow + i];
        jds->values[idx] = ell.values[ellRow * ell.maxNNZPerRow + i];
        idx++;
      } else {
        // 同样利用降序特性，提前结束内层循环
        break;
      }
    }
  }
}

__global__ void spmv_ell_kernel(ELLMatrix ell, float *inVec, float *outVec) {
  unsigned int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < ell.numRows) {
    float sum = 0.0f;
    for (unsigned int idx = 0; idx < ell.nnzPerRow[row]; ++idx) {
      unsigned int col = ell.colIdxs[row * ell.maxNNZPerRow + idx];
      float val = ell.values[row * ell.maxNNZPerRow + idx];
      if (col < ell.numCols) {
        sum += val * inVec[col];
      }
    }
    outVec[row] = sum;
  }
}

void spmv_ell_gpu(ELLMatrix ell, float *inVec, float *outVec) {

  ELLMatrix ell_d;
  cudaMalloc(&ell_d.nnzPerRow, ell.numRows * sizeof(unsigned int));
  cudaMalloc(&ell_d.colIdxs,
             ell.maxNNZPerRow * ell.numRows * sizeof(unsigned int));
  cudaMalloc(&ell_d.values, ell.maxNNZPerRow * ell.numRows * sizeof(float));
  cudaMemcpy(ell_d.nnzPerRow, ell.nnzPerRow, ell.numRows * sizeof(unsigned int),
             cudaMemcpyHostToDevice);
  cudaMemcpy(ell_d.colIdxs, ell.colIdxs,
             ell.maxNNZPerRow * ell.numRows * sizeof(unsigned int),
             cudaMemcpyHostToDevice);
  cudaMemcpy(ell_d.values, ell.values,
             ell.maxNNZPerRow * ell.numRows * sizeof(float),
             cudaMemcpyHostToDevice);
  ell_d.numRows = ell.numRows;
  ell_d.numCols = ell.numCols;
  ell_d.maxNNZPerRow = ell.maxNNZPerRow;
  float *inVec_d, *outVec_d;
  cudaMalloc(&inVec_d, ell.numCols * sizeof(float));
  cudaMalloc(&outVec_d, ell.numRows * sizeof(float));
  cudaMemcpy(inVec_d, inVec, ell.numCols * sizeof(float),
             cudaMemcpyHostToDevice);
  cudaMemset(outVec_d, 0, ell.numRows * sizeof(float));

  dim3 blockDim(THREADS_PER_BLOCK__ELL);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);

  unsigned int numBlocks = (ell.numRows + blockDim.x - 1) / blockDim.x;
  spmv_ell_kernel<<<numBlocks, blockDim>>>(ell_d, inVec_d, outVec_d);
  cudaDeviceSynchronize();
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  printf("SpMV (ELL) kernel: %f ms\n", milliseconds);
  cudaMemcpy(outVec, outVec_d, ell.numRows * sizeof(float),
             cudaMemcpyDeviceToHost);
  cudaFree(ell_d.nnzPerRow);
  cudaFree(ell_d.colIdxs);
  cudaFree(ell_d.values);
  cudaFree(inVec_d);
  cudaFree(outVec_d);
}

void csr2ell(CSRMatrix csr, ELLMatrix *ell) {
  ell->numRows = csr.numRows;
  ell->numCols = csr.numCols;

  ell->nnzPerRow = (unsigned int *)malloc(csr.numRows * sizeof(unsigned int));
  ell->maxNNZPerRow = 0;
  for (unsigned int i = 0; i < csr.numRows; ++i) {
    ell->nnzPerRow[i] = csr.rowPtrs[i + 1] - csr.rowPtrs[i];
    if (ell->nnzPerRow[i] > ell->maxNNZPerRow) {
      ell->maxNNZPerRow = ell->nnzPerRow[i];
    }
  }

  ell->colIdxs = (unsigned int *)malloc(ell->maxNNZPerRow * ell->numRows *
                                        sizeof(unsigned int));
  ell->values =
      (float *)malloc(ell->maxNNZPerRow * ell->numRows * sizeof(float));
  for (unsigned int i = 0; i < ell->numRows; ++i) {
    unsigned int idx = 0;
    for (unsigned int j = csr.rowPtrs[i]; j < csr.rowPtrs[i + 1]; ++j) {
      ell->colIdxs[i * ell->maxNNZPerRow + idx] = csr.colIdxs[j];
      ell->values[i * ell->maxNNZPerRow + idx] = csr.values[j];
      idx++;
    }
    // Pad remaining entries with zeros
    for (unsigned int j = idx; j < ell->maxNNZPerRow; ++j) {
      ell->colIdxs[i * ell->maxNNZPerRow + j] = 0;
      ell->values[i * ell->maxNNZPerRow + j] = 0.0f;
    }
  }
}

__global__ void spmv_csr_kernel(CSRMatrix csr, float *inVec, float *outVec) {
  unsigned int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < csr.numRows) {
    float sum = 0.0f;
    for (unsigned int idx = csr.rowPtrs[row]; idx < csr.rowPtrs[row + 1];
         ++idx) {
      unsigned int col = csr.colIdxs[idx];
      float val = csr.values[idx];
      sum += val * inVec[col];
    }
    outVec[row] = sum;
  }
}

void spmv_csr_gpu(CSRMatrix csr, float *inVec, float *outVec) {

  CSRMatrix csr_d;
  cudaMalloc(&csr_d.rowPtrs, (csr.numRows + 1) * sizeof(unsigned int));
  cudaMalloc(&csr_d.colIdxs, csr.numNonZeros * sizeof(unsigned int));
  cudaMalloc(&csr_d.values, csr.numNonZeros * sizeof(float));
  cudaMemcpy(csr_d.rowPtrs, csr.rowPtrs,
             (csr.numRows + 1) * sizeof(unsigned int), cudaMemcpyHostToDevice);
  cudaMemcpy(csr_d.colIdxs, csr.colIdxs, csr.numNonZeros * sizeof(unsigned int),
             cudaMemcpyHostToDevice);
  cudaMemcpy(csr_d.values, csr.values, csr.numNonZeros * sizeof(float),
             cudaMemcpyHostToDevice);
  csr_d.numRows = csr.numRows;
  csr_d.numCols = csr.numCols;
  csr_d.numNonZeros = csr.numNonZeros;

  float *inVec_d, *outVec_d;
  cudaMalloc(&inVec_d, csr.numCols * sizeof(float));
  cudaMalloc(&outVec_d, csr.numRows * sizeof(float));
  cudaMemcpy(inVec_d, inVec, csr.numCols * sizeof(float),
             cudaMemcpyHostToDevice);
  cudaMemset(outVec_d, 0, csr.numRows * sizeof(float));

  dim3 blockDim(THREADS_PER_BLOCK__CSR);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);

  unsigned int numBlocks = (csr.numRows + blockDim.x - 1) / blockDim.x;
  spmv_csr_kernel<<<numBlocks, blockDim>>>(csr_d, inVec_d, outVec_d);
  cudaDeviceSynchronize();
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  printf("SpMV (CSR) kernel: %f ms\n", milliseconds);
  cudaMemcpy(outVec, outVec_d, csr.numRows * sizeof(float),
             cudaMemcpyDeviceToHost);
  cudaFree(csr_d.rowPtrs);
  cudaFree(csr_d.colIdxs);
  cudaFree(csr_d.values);
  cudaFree(inVec_d);
  cudaFree(outVec_d);
}

void coo2csr(COOMatrix coo, CSRMatrix *csr) {
  csr->numRows = coo.numRows;
  csr->numCols = coo.numCols;
  csr->numNonZeros = coo.numNonZeros;

  csr->rowPtrs =
      (unsigned int *)malloc((csr->numRows + 1) * sizeof(unsigned int));
  csr->colIdxs =
      (unsigned int *)malloc(csr->numNonZeros * sizeof(unsigned int));
  csr->values = (float *)malloc(csr->numNonZeros * sizeof(float));

  // Initialize rowPtrs to 0
  for (unsigned int i = 0; i <= csr->numRows; ++i) {
    csr->rowPtrs[i] = 0;
  }

  // Count non-zeros per row
  for (unsigned int i = 0; i < coo.numNonZeros; ++i) {
    csr->rowPtrs[coo.rowIdxs[i] + 1]++;
  }

  // Cumulative sum to get row pointers
  for (unsigned int i = 1; i <= csr->numRows; ++i) {
    csr->rowPtrs[i] += csr->rowPtrs[i - 1];
  }

  // Fill colIdxs and values
  unsigned int *currentPos =
      (unsigned int *)malloc(csr->numRows * sizeof(unsigned int));
  for (unsigned int i = 0; i < coo.numRows; ++i) {
    currentPos[i] = csr->rowPtrs[i];
  }
  for (unsigned int i = 0; i < coo.numNonZeros; ++i) {
    unsigned int row = coo.rowIdxs[i];
    unsigned int destPos = currentPos[row]++;
    csr->colIdxs[destPos] = coo.colIdxs[i];
    csr->values[destPos] = coo.values[i];
  }

  free(currentPos);
}

__global__ void spmv_coo_kernel(COOMatrix coo, float *inVec, float *outVec) {
  unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < coo.numNonZeros) {
    unsigned int row = coo.rowIdxs[idx];
    unsigned int col = coo.colIdxs[idx];
    float val = coo.values[idx];
    atomicAdd(&outVec[row], val * inVec[col]);
  }
}

void spmv_coo_gpu(COOMatrix coo, float *inVec, float *outVec) {

  COOMatrix coo_d;
  cudaMalloc(&coo_d.rowIdxs, coo.numNonZeros * sizeof(unsigned int));
  cudaMalloc(&coo_d.colIdxs, coo.numNonZeros * sizeof(unsigned int));
  cudaMalloc(&coo_d.values, coo.numNonZeros * sizeof(float));
  cudaMemcpy(coo_d.rowIdxs, coo.rowIdxs, coo.numNonZeros * sizeof(unsigned int),
             cudaMemcpyHostToDevice);
  cudaMemcpy(coo_d.colIdxs, coo.colIdxs, coo.numNonZeros * sizeof(unsigned int),
             cudaMemcpyHostToDevice);
  cudaMemcpy(coo_d.values, coo.values, coo.numNonZeros * sizeof(float),
             cudaMemcpyHostToDevice);
  coo_d.numRows = coo.numRows;
  coo_d.numCols = coo.numCols;
  coo_d.numNonZeros = coo.numNonZeros;

  float *inVec_d, *outVec_d;
  cudaMalloc(&inVec_d, coo.numCols * sizeof(float));
  cudaMalloc(&outVec_d, coo.numRows * sizeof(float));
  cudaMemcpy(inVec_d, inVec, coo.numCols * sizeof(float),
             cudaMemcpyHostToDevice);
  cudaMemset(outVec_d, 0, coo.numRows * sizeof(float));

  dim3 blockDim(THREADS_PER_BLOCK__COO);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);

  unsigned int numBlocks = (coo.numNonZeros + blockDim.x - 1) / blockDim.x;
  spmv_coo_kernel<<<numBlocks, blockDim>>>(coo_d, inVec_d, outVec_d);
  cudaDeviceSynchronize();
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  printf("SpMV (COO) kernel: %f ms\n", milliseconds);
  cudaMemcpy(outVec, outVec_d, coo.numRows * sizeof(float),
             cudaMemcpyDeviceToHost);
  cudaFree(coo_d.rowIdxs);
  cudaFree(coo_d.colIdxs);
  cudaFree(coo_d.values);
  cudaFree(inVec_d);
  cudaFree(outVec_d);
}

void spmv_cusparse_gpu(CSRMatrix csr, float *inVec, float *outVec) {
  CSRMatrix csr_d;
  cudaMalloc(&csr_d.rowPtrs, (csr.numRows + 1) * sizeof(unsigned int));
  cudaMalloc(&csr_d.colIdxs, csr.numNonZeros * sizeof(unsigned int));
  cudaMalloc(&csr_d.values, csr.numNonZeros * sizeof(float));
  cudaMemcpy(csr_d.rowPtrs, csr.rowPtrs,
             (csr.numRows + 1) * sizeof(unsigned int), cudaMemcpyHostToDevice);
  cudaMemcpy(csr_d.colIdxs, csr.colIdxs, csr.numNonZeros * sizeof(unsigned int),
             cudaMemcpyHostToDevice);
  cudaMemcpy(csr_d.values, csr.values, csr.numNonZeros * sizeof(float),
             cudaMemcpyHostToDevice);

  float *inVec_d, *outVec_d;
  cudaMalloc(&inVec_d, csr.numCols * sizeof(float));
  cudaMalloc(&outVec_d, csr.numRows * sizeof(float));
  cudaMemcpy(inVec_d, inVec, csr.numCols * sizeof(float),
             cudaMemcpyHostToDevice);
  cudaMemset(outVec_d, 0, csr.numRows * sizeof(float));

  cusparseHandle_t handle = NULL;
  cusparseCreate(&handle);

  cusparseSpMatDescr_t matA;
  cusparseCreateCsr(&matA, csr.numRows, csr.numCols, csr.numNonZeros,
                    csr_d.rowPtrs, csr_d.colIdxs, csr_d.values,
                    CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                    CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);

  cusparseDnVecDescr_t vecX, vecY;
  cusparseCreateDnVec(&vecX, csr.numCols, inVec_d, CUDA_R_32F);
  cusparseCreateDnVec(&vecY, csr.numRows, outVec_d, CUDA_R_32F);

  float alpha = 1.0f;
  float beta = 0.0f;
  size_t bufferSize = 0;
  cusparseSpMV_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha,
                          matA, vecX, &beta, vecY, CUDA_R_32F,
                          CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize);
  void *dBuffer = NULL;
  cudaMalloc(&dBuffer, bufferSize);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);

  cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA, vecX,
               &beta, vecY, CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, dBuffer);

  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  printf("cuSPARSE kernel: %f ms\n", milliseconds);

  cudaMemcpy(outVec, outVec_d, csr.numRows * sizeof(float),
             cudaMemcpyDeviceToHost);

  cusparseDestroySpMat(matA);
  cusparseDestroyDnVec(vecX);
  cusparseDestroyDnVec(vecY);
  cusparseDestroy(handle);
  cudaFree(dBuffer);

  cudaFree(csr_d.rowPtrs);
  cudaFree(csr_d.colIdxs);
  cudaFree(csr_d.values);
  cudaFree(inVec_d);
  cudaFree(outVec_d);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
}

int main() {
  // Example usage
  COOMatrix coo;
  if (MY_DEBUG) {
    coo.numRows = 10;
    coo.numCols = 5;
    coo.numNonZeros = 10;
  } else {
    coo.numRows = (1 << 18) - 1;
    coo.numCols = 1 << 17;
    coo.numNonZeros = 1 << 24;
  }

  unsigned int *rowIdxs =
      (unsigned int *)malloc(coo.numNonZeros * sizeof(unsigned int));
  unsigned int *colIdxs =
      (unsigned int *)malloc(coo.numNonZeros * sizeof(unsigned int));
  float *values = (float *)malloc(coo.numNonZeros * sizeof(float));
  coo.rowIdxs = rowIdxs;
  coo.colIdxs = colIdxs;
  coo.values = values;

  float *inVec = (float *)malloc(coo.numCols * sizeof(float));
  float *outVecByCOO = (float *)malloc(coo.numRows * sizeof(float));
  float *outVecByCSR = (float *)malloc(coo.numRows * sizeof(float));
  float *outVecByELL = (float *)malloc(coo.numRows * sizeof(float));
  float *outVecByJDS = (float *)malloc(coo.numRows * sizeof(float));

  std::unordered_set<size_t> occupied;
  occupied.reserve(coo.numNonZeros); // 预分配哈希表容量，防止频繁 rehash

  srand(time(NULL));

  for (unsigned int i = 0; i < coo.numNonZeros; ++i) {
    while (1) {
      rowIdxs[i] = rand() % coo.numRows;
      colIdxs[i] = rand() % coo.numCols;
      // Ensure no duplicate entries
      size_t flatIdx = (size_t)rowIdxs[i] * coo.numCols + colIdxs[i];
      if (occupied.insert(flatIdx).second) {
        break;
      }
    }
    values[i] = rand() / (float)RAND_MAX;
  }

  for (unsigned int i = 0; i < coo.numCols; ++i) {
    inVec[i] = rand() / (float)RAND_MAX;
    // inVec[i] = 1.0f; // Use a simple input vector for easier verification
  }

  printf("Size of matrix: %u x %u with %u non-zeros, inVec size: %u, outVec "
         "size: %u\n",
         coo.numRows, coo.numCols, coo.numNonZeros, coo.numCols, coo.numRows);

  time_t start = clock();
  spmv_coo_gpu(coo, inVec, outVecByCOO);
  cudaDeviceSynchronize();
  time_t end = clock();
  double elapsed_time = (double)(end - start) / CLOCKS_PER_SEC;
  printf("SpMV / COO   Elapsed time: %f seconds\n", elapsed_time);

  start = clock();
  CSRMatrix csr;
  coo2csr(coo, &csr);
  end = clock();
  elapsed_time = (double)(end - start) / CLOCKS_PER_SEC;
  printf("COO to CSR Convert time: %f seconds\n", elapsed_time);
  start = clock();
  spmv_csr_gpu(csr, inVec, outVecByCSR);
  cudaDeviceSynchronize();
  end = clock();
  elapsed_time = (double)(end - start) / CLOCKS_PER_SEC;
  printf("SpMV / CSR   Elapsed time: %f seconds\n", elapsed_time);

  start = clock();
  ELLMatrix ell;
  csr2ell(csr, &ell);
  end = clock();
  elapsed_time = (double)(end - start) / CLOCKS_PER_SEC;
  printf("CSR to ELL Convert time: %f seconds\n", elapsed_time);
  start = clock();
  spmv_ell_gpu(ell, inVec, outVecByELL);
  cudaDeviceSynchronize();
  end = clock();
  elapsed_time = (double)(end - start) / CLOCKS_PER_SEC;
  printf("SpMV / ELL   Elapsed time: %f seconds\n", elapsed_time);

  start = clock();
  JDSMatrix jds;
  ell2jds(ell, &jds);
  end = clock();
  elapsed_time = (double)(end - start) / CLOCKS_PER_SEC;
  printf("ELL to JDS Convert time: %f seconds\n", elapsed_time);
  start = clock();
  spmv_jds_gpu(jds, inVec, outVecByJDS);
  cudaDeviceSynchronize();
  end = clock();
  elapsed_time = (double)(end - start) / CLOCKS_PER_SEC;
  printf("SpMV / JDS   Elapsed time: %f seconds\n", elapsed_time);
  if (MY_DEBUG) {
    printf("Dense Matrix:\n");
    for (unsigned int i = 0; i < coo.numRows; ++i) {
      for (unsigned int j = 0; j < coo.numCols; ++j) {
        bool found = false;
        for (unsigned int k = 0; k < coo.numNonZeros; ++k) {
          if (coo.rowIdxs[k] == i && coo.colIdxs[k] == j) {
            printf("%.4f\t", coo.values[k]);
            found = true;
            break;
          }
        }
        if (!found) {
          printf(" .    \t");
        }
      }
      printf("\n");
    }
    printf("ELL colIdxs:\t");
    for (unsigned int i = 0; i < ell.maxNNZPerRow * ell.numRows; ++i) {
      printf("%u\t", ell.colIdxs[i]);
    }
    printf("\n");
    printf("ELL values:\t");
    for (unsigned int i = 0; i < ell.maxNNZPerRow * ell.numRows; ++i) {
      printf("%.4f\t", ell.values[i]);
    }
    printf("\n");
    printf("JDS nnzPerRow: ");
    for (unsigned int i = 0; i < jds.numRows; ++i) {
      printf("%u ", jds.nnzPerRow[i]);
    }
    printf("\n");
    printf("JDS rowIdxs: ");
    for (unsigned int i = 0; i < jds.numRows; ++i) {
      printf("%u ", jds.rowIdxs[i]);
    }
    printf("\n");
    printf("JDS iterPtr: ");
    for (unsigned int i = 0; i <= jds.maxNNZPerRow; ++i) {
      printf("%u ", jds.iterPtr[i]);
    }
    printf("\n");
    printf("            \t");
    for (unsigned int i = 0; i < jds.numNonZeros; ++i) {
      printf("%u\t", i);
    }
    printf("\n");
    printf("JDS colIdxs:\t");
    for (unsigned int i = 0; i < jds.numNonZeros; ++i) {
      printf("%u\t", jds.colIdxs[i]);
    }
    printf("\n");
    printf("JDS values:\t");
    for (unsigned int i = 0; i < jds.numNonZeros; ++i) {
      printf("%.4f\t", jds.values[i]);
    }
    printf("\n");
  }

  float *outVecByCuSparse = (float *)malloc(coo.numRows * sizeof(float));
  start = clock();
  spmv_cusparse_gpu(csr, inVec, outVecByCuSparse);
  cudaDeviceSynchronize();
  end = clock();
  elapsed_time = (double)(end - start) / CLOCKS_PER_SEC;
  printf("cusparseSpMV Elapsed time: %f seconds\n", elapsed_time);

  printf("Memory usage (bytes):\n"
         "  COO: %lu\n  CSR: %lu\n  ELL: %lu\n  JDS: %lu\n",
         coo.numNonZeros * (sizeof(unsigned int) * 2 + sizeof(float)),
         (csr.numRows + 1) * sizeof(unsigned int) + csr.numNonZeros * (sizeof(unsigned int) + sizeof(float)),
         ell.numRows * sizeof(unsigned int) + ell.maxNNZPerRow * ell.numRows * (sizeof(unsigned int) + sizeof(float)),
         jds.numRows * sizeof(unsigned int) * 2 + (jds.maxNNZPerRow + 1) * sizeof(unsigned int) + jds.numNonZeros * (sizeof(unsigned int) + sizeof(float)));

  int match = 1;
  for (unsigned int i = 0; i < coo.numRows; ++i) {
    if (fabs(outVecByCOO[i] - outVecByCuSparse[i]) > 1e-4 ||
        fabs(outVecByCSR[i] - outVecByCuSparse[i]) > 1e-4 ||
        fabs(outVecByELL[i] - outVecByCuSparse[i]) > 1e-4 ||
        fabs(outVecByJDS[i] - outVecByCuSparse[i]) > 1e-4) {
      match = 0;
      printf(
          "Mismatch at index %u: COO %f, CSR %f, ELL %f, JDS %f, cuSPARSE %f\n",
          i, outVecByCOO[i], outVecByCSR[i], outVecByELL[i], outVecByJDS[i],
          outVecByCuSparse[i]);
      break;
    }
  }
  if (match)
    printf("Results match with cuSPARSE!\n");

  free(outVecByCuSparse);

  // Free allocated memory
  free(rowIdxs);
  free(colIdxs);
  free(values);
  free(inVec);
  free(outVecByCOO);
  free(outVecByCSR);
  free(outVecByELL);
  free(outVecByJDS);
  free(csr.rowPtrs);
  free(csr.colIdxs);
  free(csr.values);
  free(ell.nnzPerRow);
  free(ell.colIdxs);
  free(ell.values);
  return 0;
}
