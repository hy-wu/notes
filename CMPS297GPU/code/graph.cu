#include <chrono>
#include <cstdio>
#include <cuda_runtime.h>
#include <fstream>
#include <sstream>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <vector>

#define THREADS_PER_BLOCK__CSR 256
#define THREADS_PER_BLOCK__CSR_COARSED 1024
#define THREADS_PER_BLOCK__COO 256
#define LOCAL_QUEUE_SIZE 2048
#define DYNAMIC_PARALLEL_THRESHOLD 1200

struct CSRGraph {
  unsigned int numVertices;
  unsigned int numEdges;
  unsigned int *srcPtrs;
  unsigned int *dst;
};

struct COOGraph {
  unsigned int numVertices;
  unsigned int numEdges;
  unsigned int *src;
  unsigned int *dst;
};

void bfs_cpu(CSRGraph graph, unsigned int source, unsigned int *level) {
  unsigned int *queue =
      (unsigned int *)malloc(graph.numVertices * sizeof(unsigned int));
  unsigned int head = 0;
  unsigned int tail = 0;

  for (unsigned int i = 0; i < graph.numVertices; ++i) {
    level[i] = UINT_MAX;
  }
  level[source] = 0;
  queue[tail++] = source;

  while (head < tail) {
    unsigned int u = queue[head++];
    unsigned int current_level = level[u];
    unsigned int start_edge = graph.srcPtrs[u];
    unsigned int end_edge = graph.srcPtrs[u + 1];

    for (unsigned int e = start_edge; e < end_edge; ++e) {
      unsigned int v = graph.dst[e];
      if (level[v] == UINT_MAX) {
        level[v] = current_level + 1;
        queue[tail++] = v;
      }
    }
  }
  free(queue);
}

__global__ void bfs_coo_kernel(COOGraph cooGraph, unsigned int *level,
                               unsigned int *newVertexVistied,
                               unsigned int currentLevel) {
  unsigned int edge = blockIdx.x * blockDim.x + threadIdx.x;
  if (edge < cooGraph.numEdges) {
    unsigned int vertex = cooGraph.src[edge];
    unsigned int neighbor = cooGraph.dst[edge];
    if (level[vertex] == currentLevel - 1 && level[neighbor] == UINT_MAX) {
      level[neighbor] = currentLevel;
      *newVertexVistied = 1;
    }
  }
}

void bfs_coo(COOGraph cooGraph, unsigned int source, unsigned int *level) {
  COOGraph cooGraph_d;
  cudaMalloc(&cooGraph_d.src, cooGraph.numEdges * sizeof(unsigned int));
  cudaMalloc(&cooGraph_d.dst, cooGraph.numEdges * sizeof(unsigned int));
  unsigned int *level_d;
  cudaMalloc(&level_d, cooGraph.numVertices * sizeof(unsigned int));
  unsigned int *newVertexVistied_d;
  cudaMalloc(&newVertexVistied_d, sizeof(unsigned int));

  cooGraph_d.numVertices = cooGraph.numVertices;
  cooGraph_d.numEdges = cooGraph.numEdges;
  cudaMemcpy(cooGraph_d.src, cooGraph.src,
             cooGraph.numEdges * sizeof(unsigned int), cudaMemcpyHostToDevice);
  cudaMemcpy(cooGraph_d.dst, cooGraph.dst,
             cooGraph.numEdges * sizeof(unsigned int), cudaMemcpyHostToDevice);
  cudaMemcpy(level_d, level, cooGraph.numVertices * sizeof(unsigned int),
             cudaMemcpyHostToDevice);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);
  unsigned int numThreadsPerBlock = THREADS_PER_BLOCK__COO;
  unsigned int numBlocks =
      (cooGraph_d.numEdges + numThreadsPerBlock - 1) / numThreadsPerBlock;
  unsigned int newVertexVistied = 1;
  for (unsigned int currentLevel = 1; newVertexVistied; ++currentLevel) {
    newVertexVistied = 0;
    cudaMemcpy(newVertexVistied_d, &newVertexVistied, sizeof(unsigned int),
               cudaMemcpyHostToDevice);
    bfs_coo_kernel<<<numBlocks, numThreadsPerBlock>>>(
        cooGraph_d, level_d, newVertexVistied_d, currentLevel);
    cudaDeviceSynchronize();
    cudaMemcpy(&newVertexVistied, newVertexVistied_d, sizeof(unsigned int),
               cudaMemcpyDeviceToHost);
  }
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  printf("BFS (COO) kernel: %f ms\n", milliseconds);
  cudaMemcpy(level, level_d, cooGraph.numVertices * sizeof(unsigned int),
             cudaMemcpyDeviceToHost);
  cudaFree(cooGraph_d.src);
  cudaFree(cooGraph_d.dst);
  cudaFree(level_d);
  cudaFree(newVertexVistied_d);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
}

__global__ void bfs_csr_top_down_kernel(CSRGraph graph, unsigned int *level,
                                        unsigned int *newVertexVistied,
                                        unsigned int currentLevel) {
  unsigned int vertex = blockIdx.x * blockDim.x + threadIdx.x;
  if (vertex < graph.numVertices) {
    if (level[vertex] == currentLevel - 1) {
      for (unsigned int edge = graph.srcPtrs[vertex];
           edge < graph.srcPtrs[vertex + 1]; ++edge) {
        unsigned int neighbor = graph.dst[edge];
        if (level[neighbor] == UINT_MAX) {
          level[neighbor] = currentLevel;
          *newVertexVistied = 1;
        }
      }
    }
  }
}

__global__ void bfs_csr_bottom_up_kernel(CSRGraph graph, unsigned int *level,
                                         unsigned int *newVertexVistied,
                                         unsigned int currentLevel) {
  unsigned int vertex = blockIdx.x * blockDim.x + threadIdx.x;
  if (vertex < graph.numVertices) {
    if (level[vertex] == UINT_MAX) {
      for (unsigned int edge = graph.srcPtrs[vertex];
           edge < graph.srcPtrs[vertex + 1]; ++edge) {
        unsigned int neighbor = graph.dst[edge];
        if (level[neighbor] == currentLevel - 1) {
          level[vertex] = currentLevel;
          *newVertexVistied = 1;
          break;
        }
      }
    }
  }
}

__global__ void bfs_csr_frontier_kernel(CSRGraph graph, unsigned int *level,
                                        unsigned int *prevFrontier,
                                        unsigned int *currFrontier,
                                        unsigned int numPrevFrontier,
                                        unsigned int *numCurrFrontier,
                                        unsigned int currentLevel) {
  __shared__ unsigned int currFrontier_s[LOCAL_QUEUE_SIZE];
  __shared__ unsigned int numCurrFrontier_s;
  if (threadIdx.x == 0) {
    numCurrFrontier_s = 0;
  }
  __syncthreads();

  unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < numPrevFrontier) {
    unsigned int vertex = prevFrontier[idx];
    for (unsigned int edge = graph.srcPtrs[vertex];
         edge < graph.srcPtrs[vertex + 1]; ++edge) {
      unsigned int neighbor = graph.dst[edge];
      // if (level[neighbor] == UINT_MAX) {
      //   level[neighbor] = currentLevel;
      if (atomicCAS(&level[neighbor], UINT_MAX, currentLevel) == UINT_MAX) {
        // unsigned int pos = atomicAdd(numCurrFrontier, 1);
        // currFrontier[pos] = neighbor;
        unsigned int i = atomicAdd(&numCurrFrontier_s, 1);
        if (i < LOCAL_QUEUE_SIZE) {
          currFrontier_s[i] = neighbor;
        } else {
          unsigned int pos = atomicAdd(numCurrFrontier, 1);
          currFrontier[pos] = neighbor;
        }
      }
    }
  }
  __syncthreads();
  __shared__ unsigned int currFrontierStartIdx;
        __shared__ unsigned int limit;
  if (threadIdx.x == 0) {
          limit = numCurrFrontier_s > LOCAL_QUEUE_SIZE ? LOCAL_QUEUE_SIZE : numCurrFrontier_s;
          currFrontierStartIdx = atomicAdd(numCurrFrontier, limit);
  }
  __syncthreads();
        for (unsigned int i = threadIdx.x; i < limit; i += blockDim.x) {
    unsigned int pos = currFrontierStartIdx + i;
    currFrontier[pos] = currFrontier_s[i];
  }
  __syncthreads();
}

__global__ void bfs_simple_kernel(CSRGraph graph, unsigned int *level,
                                  unsigned int *prevFrontier,
                                  unsigned int *currFrontier,
                                  unsigned int numPrevFrontier,
                                  unsigned int *numCurrFrontier,
                                  unsigned int currentLevel) {
  unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < numPrevFrontier) {
    unsigned int vertex = prevFrontier[idx];
    unsigned int start = graph.srcPtrs[vertex];
    unsigned int numNeighbors = graph.srcPtrs[vertex + 1] - start;
    for (unsigned int i = 0; i < numNeighbors; ++i) {
      unsigned int neighbor = graph.dst[start + i];
      if (atomicCAS(&level[neighbor], UINT_MAX, currentLevel) == UINT_MAX) {
        unsigned int currFrontierIdx = atomicAdd(numCurrFrontier, 1);
        currFrontier[currFrontierIdx] = neighbor;
      }
    }
  }
}

#if __CUDACC_RDC__
__global__ void
bfs_child_kernel(CSRGraph graph, unsigned int *level,
                 unsigned int *currFrontier, unsigned int numPrevFrontier,
                 unsigned int *numCurrFrontier, unsigned int currentLevel,
                 unsigned int numNeighbors, unsigned int start) {
  unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < numNeighbors) {
    unsigned int neighbor = graph.dst[start + idx];
    if (atomicCAS(&level[neighbor], UINT_MAX, currentLevel) == UINT_MAX) {
      unsigned int currFrontierIdx = atomicAdd(numCurrFrontier, 1);
      currFrontier[currFrontierIdx] = neighbor;
    }
  }
}

__global__ void bfs_dyn_paral_kernel(CSRGraph graph, unsigned int *level,
                                     unsigned int *prevFrontier,
                                     unsigned int *currFrontier,
                                     unsigned int numPrevFrontier,
                                     unsigned int *numCurrFrontier,
                                     unsigned int currentLevel) {
  unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < numPrevFrontier) {
    unsigned int vertex = prevFrontier[idx];
    unsigned int start = graph.srcPtrs[vertex];
    unsigned int numNeighbors = graph.srcPtrs[vertex + 1] - start;
    unsigned int numThreadsPerBlock = 1024;
    if (numNeighbors > DYNAMIC_PARALLEL_THRESHOLD) {
      unsigned int numBlocks =
          (numNeighbors + numThreadsPerBlock - 1) / numThreadsPerBlock;
      bfs_child_kernel<<<numBlocks, numThreadsPerBlock>>>(
          graph, level, currFrontier, numPrevFrontier, numCurrFrontier,
          currentLevel, numNeighbors, start);
    } else {
      for (unsigned int i = 0; i < numNeighbors; ++i) {
        unsigned int neighbor = graph.dst[start + i];
        if (atomicCAS(&level[neighbor], UINT_MAX, currentLevel) == UINT_MAX) {
          unsigned int currFrontierIdx = atomicAdd(numCurrFrontier, 1);
          currFrontier[currFrontierIdx] = neighbor;
        }
      }
    }
  }
}
#endif

__global__ void bfs_csr_frontier_coarsed_kernel(CSRGraph graph,
                                                unsigned int *level,
                                                unsigned int *buffer1,
                                                unsigned int *buffer2) {

  __shared__ unsigned int currentLevel;
  __shared__ unsigned int currFrontier_s[LOCAL_QUEUE_SIZE];
  __shared__ unsigned int *prevFrontier;
  __shared__ unsigned int *currFrontier;
  __shared__ unsigned int numPrevFrontier;
  __shared__ unsigned int numCurrFrontier;
  __shared__ unsigned int numCurrFrontier_s;
  if (threadIdx.x == 0) {
    prevFrontier = buffer1;
    currFrontier = buffer2;
    numPrevFrontier = 1;
    numCurrFrontier = 0;
    numCurrFrontier_s = 0;
    currentLevel = 1;
  }
  __syncthreads();
  while (numPrevFrontier > 0) {
    for (unsigned int idx = threadIdx.x; idx < numPrevFrontier;
         idx += blockDim.x) {
      unsigned int vertex = prevFrontier[idx];
      for (unsigned int edge = graph.srcPtrs[vertex];
           edge < graph.srcPtrs[vertex + 1]; ++edge) {
        unsigned int neighbor = graph.dst[edge];
        if (atomicCAS(&level[neighbor], UINT_MAX, currentLevel) == UINT_MAX) {
          unsigned int i = atomicAdd(&numCurrFrontier_s, 1);
          if (i < LOCAL_QUEUE_SIZE) {
            currFrontier_s[i] = neighbor;
          } else {
            unsigned int pos = atomicAdd(&numCurrFrontier, 1);
            currFrontier[pos] = neighbor;
          }
        }
      }
    }
    __syncthreads();
    __shared__ unsigned int currFrontierStartIdx;
            __shared__ unsigned int limit;
    if (threadIdx.x == 0) {
              limit = numCurrFrontier_s > LOCAL_QUEUE_SIZE ? LOCAL_QUEUE_SIZE : numCurrFrontier_s;
              currFrontierStartIdx = atomicAdd(&numCurrFrontier, limit);
    }
    __syncthreads();
            for (unsigned int i = threadIdx.x; i < limit; i += blockDim.x) {
      unsigned int pos = currFrontierStartIdx + i;
      currFrontier[pos] = currFrontier_s[i];
    }
    if (threadIdx.x == 0) {
      numPrevFrontier = numCurrFrontier;
      numCurrFrontier_s = 0;
      unsigned int *temp = prevFrontier;
      prevFrontier = currFrontier;
      currFrontier = temp;
      numCurrFrontier = 0;
      atomicAdd(&currentLevel, 1);
    }
    __syncthreads();
  }
}

void bfs_csr(CSRGraph graph, unsigned int source, unsigned int *level,
             char policy) {
  CSRGraph graph_d;
  cudaMalloc(&graph_d.srcPtrs, (graph.numVertices + 1) * sizeof(unsigned int));
  cudaMalloc(&graph_d.dst, graph.numEdges * sizeof(unsigned int));
  unsigned int *level_d;
  cudaMalloc(&level_d, graph.numVertices * sizeof(unsigned int));
  unsigned int *newVertexVistied_d;
  cudaMalloc(&newVertexVistied_d, sizeof(unsigned int));

  graph_d.numVertices = graph.numVertices;
  graph_d.numEdges = graph.numEdges;
  cudaMemcpy(graph_d.srcPtrs, graph.srcPtrs,
             (graph.numVertices + 1) * sizeof(unsigned int),
             cudaMemcpyHostToDevice);
  cudaMemcpy(graph_d.dst, graph.dst, graph.numEdges * sizeof(unsigned int),
             cudaMemcpyHostToDevice);
  cudaMemcpy(level_d, level, graph.numVertices * sizeof(unsigned int),
             cudaMemcpyHostToDevice);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);

  if (policy == 't' || policy == 'b' || policy == 'd') {
    unsigned int numThreadsPerBlock = THREADS_PER_BLOCK__CSR;
    unsigned int numBlocks =
        (graph_d.numVertices + numThreadsPerBlock - 1) / numThreadsPerBlock;
    unsigned int newVertexVistied = 1;
    for (unsigned int currentLevel = 1; newVertexVistied; ++currentLevel) {
      newVertexVistied = 0;
      cudaMemcpy(newVertexVistied_d, &newVertexVistied, sizeof(unsigned int),
                 cudaMemcpyHostToDevice);
      if (policy == 't') {
        bfs_csr_top_down_kernel<<<numBlocks, numThreadsPerBlock>>>(
            graph_d, level_d, newVertexVistied_d, currentLevel);
      } else if (policy == 'b') {
        bfs_csr_bottom_up_kernel<<<numBlocks, numThreadsPerBlock>>>(
            graph_d, level_d, newVertexVistied_d, currentLevel);
      } else if (policy == 'd') {
        if (currentLevel == 1) {
          bfs_csr_top_down_kernel<<<numBlocks, numThreadsPerBlock>>>(
              graph_d, level_d, newVertexVistied_d, currentLevel);
        } else {
          bfs_csr_bottom_up_kernel<<<numBlocks, numThreadsPerBlock>>>(
              graph_d, level_d, newVertexVistied_d, currentLevel);
        }
      } else {
        printf("Invalid policy: %c\n", policy);
        exit(EXIT_FAILURE);
      }
      cudaDeviceSynchronize();
      cudaMemcpy(&newVertexVistied, newVertexVistied_d, sizeof(unsigned int),
                 cudaMemcpyDeviceToHost);
    }
  } else if (policy == 'f' || policy == 'i' || policy == 'p') {
    unsigned int *buffer1_d, *buffer2_d;
    cudaMalloc(&buffer1_d, graph_d.numVertices * sizeof(unsigned int));
    cudaMalloc(&buffer2_d, graph_d.numVertices * sizeof(unsigned int));
    unsigned int *numCurrFrontier_d;
    cudaMalloc(&numCurrFrontier_d, sizeof(unsigned int));
    unsigned int *prevFrontier_d = buffer1_d;
    unsigned int *currFrontier_d = buffer2_d;
    cudaMemcpy(prevFrontier_d, &source, sizeof(unsigned int),
               cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    unsigned int numPrevFrontier = 1;
    unsigned int currentLevel = 1;

    cudaDeviceSetLimit(cudaLimitDevRuntimePendingLaunchCount,
                       graph.numVertices);

    unsigned int numThreadsPerBlock = THREADS_PER_BLOCK__CSR;
    // printf("Number of frontier vertices: ");
    while (numPrevFrontier > 0) {
      // printf("%u ", numPrevFrontier);
      // 使用 cudaMemset 将设备上的 frontier 计数器正确清零
      cudaMemset(numCurrFrontier_d, 0, sizeof(unsigned int));
      unsigned int numBlocks =
          (numPrevFrontier + numThreadsPerBlock - 1) / numThreadsPerBlock;
      if (policy == 'f') {
        bfs_csr_frontier_kernel<<<numBlocks, numThreadsPerBlock>>>(
            graph_d, level_d, prevFrontier_d, currFrontier_d, numPrevFrontier,
            numCurrFrontier_d, currentLevel);
      } else if (policy == 'i') {
        bfs_simple_kernel<<<numBlocks, numThreadsPerBlock>>>(
            graph_d, level_d, prevFrontier_d, currFrontier_d, numPrevFrontier,
            numCurrFrontier_d, currentLevel);
      } else if (policy == 'p') {
#if __CUDACC_RDC__
        bfs_dyn_paral_kernel<<<numBlocks, numThreadsPerBlock>>>(
            graph_d, level_d, prevFrontier_d, currFrontier_d, numPrevFrontier,
            numCurrFrontier_d, currentLevel);
#else
        if (currentLevel == 1) {
          printf("Warning: Dynamic-Parallel requires compiling with -rdc=true. Skipping...\n");
        }
#endif
      }
      // 交换前后 frontier 指针，为下一次迭代做准备
      unsigned int *temp = prevFrontier_d;
      prevFrontier_d = currFrontier_d;
      currFrontier_d = temp;
      cudaDeviceSynchronize(); // 等待内核执行完毕，确保 numCurrFrontier_d
                               // 结果正确
      cudaMemcpy(&numPrevFrontier, numCurrFrontier_d, sizeof(unsigned int),
                 cudaMemcpyDeviceToHost);
      currentLevel++;
    }
    // printf("\n");
    cudaFree(buffer1_d);
    cudaFree(buffer2_d);
    cudaFree(numCurrFrontier_d);
  } else if (policy == 'c') {
    unsigned int *buffer1_d, *buffer2_d;
    cudaMalloc(&buffer1_d, graph_d.numVertices * sizeof(unsigned int));
    cudaMalloc(&buffer2_d, graph_d.numVertices * sizeof(unsigned int));
    cudaMemcpy(buffer1_d, &source, sizeof(unsigned int),
               cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();

    unsigned int numThreadsPerBlock = THREADS_PER_BLOCK__CSR_COARSED;
    unsigned int numBlocks = 1;
    bfs_csr_frontier_coarsed_kernel<<<numBlocks, numThreadsPerBlock>>>(
        graph_d, level_d, buffer1_d, buffer2_d);
    cudaDeviceSynchronize();
    cudaFree(buffer1_d);
    cudaFree(buffer2_d);
  } else {
    printf("Invalid policy: %c\n", policy);
    exit(EXIT_FAILURE);
  }

  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  const char *policyName = (policy == 't')   ? "Top-Down"
                           : (policy == 'b') ? "Bottom-Up"
                           : (policy == 'd') ? "Direction-Optimized"
                           : (policy == 'f') ? "Frontier-Based"
                           : (policy == 'c') ? "Frontier-Coarsed"
                           : (policy == 'i') ? "Frontier-Interleaved"
                           : (policy == 'p') ? "Dynamic-Parallel"
                                             : "Unknown";
  printf("BFS (CSR) kernel: %f ms using policy %s\n", milliseconds, policyName);

  cudaMemcpy(level, level_d, graph.numVertices * sizeof(unsigned int),
             cudaMemcpyDeviceToHost);
  cudaFree(graph_d.srcPtrs);
  cudaFree(graph_d.dst);
  cudaFree(level_d);
  cudaFree(newVertexVistied_d);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
}

// R-MAT (Kronecker) 幂律图生成器
void generate_rmat_edges(
    unsigned int scale, unsigned int edgeFactor,
    std::vector<std::pair<unsigned int, unsigned int>> &edges) {
  unsigned int numVertices = 1 << scale;
  unsigned int numEdges = numVertices * edgeFactor;
  // R-MAT 参数：A, B, C, D
  // 决定了边落入邻接矩阵四个象限的概率，高度倾斜导致Hub节点
  float A = 0.57f, B = 0.19f, C = 0.19f; // D = 1 - (A+B+C) = 0.05

  for (unsigned int i = 0; i < numEdges; ++i) {
    unsigned int u = 0, v = 0;
    unsigned int step = numVertices / 2;
    while (step >= 1) {
      float p = rand() / (float)RAND_MAX;
      if (p < A) {
        // Top-left: u, v 不变
      } else if (p < A + B) {
        // Top-right
        v += step;
      } else if (p < A + B + C) {
        // Bottom-left
        u += step;
      } else {
        // Bottom-right
        u += step;
        v += step;
      }
      step /= 2;
    }
    edges.push_back({u, v});
    edges.push_back({v, u}); // 无向化
  }
}

// 从本地加载真实数据集边表
void load_edges_from_file(
    const char *filename, unsigned int &numVertices,
    std::vector<std::pair<unsigned int, unsigned int>> &edges) {
  std::ifstream infile(filename);
  if (!infile.is_open()) {
    printf("Failed to open file %s\n", filename);
    exit(EXIT_FAILURE);
  }
  std::string line;
  unsigned int max_v = 0;
  while (std::getline(infile, line)) {
    if (line.empty() || line[0] == '#')
      continue;
    std::istringstream iss(line);
    unsigned int u, v;
    if (iss >> u >> v) {
      edges.push_back({u, v});
      edges.push_back({v, u}); // 无向化
      if (u > max_v)
        max_v = u;
      if (v > max_v)
        max_v = v;
    }
  }
  numVertices = max_v + 1;
  printf("Loaded %lu directed edges (after making undirected) and %u vertices "
         "from %s\n",
         edges.size(), numVertices, filename);
}

// 统构图函数：将边数组转为 CSR 和 COO
void build_graphs_from_edges(
    unsigned int numVertices,
    const std::vector<std::pair<unsigned int, unsigned int>> &edges,
    CSRGraph &csrGraph, COOGraph &cooGraph) {
  std::vector<std::vector<unsigned int>> adj(numVertices);
  for (auto &e : edges) {
    adj[e.first].push_back(e.second);
  }

  csrGraph.numVertices = numVertices;
  csrGraph.numEdges = 0;
  csrGraph.srcPtrs =
      (unsigned int *)malloc((numVertices + 1) * sizeof(unsigned int));
  for (unsigned int i = 0; i < numVertices; ++i) {
    csrGraph.srcPtrs[i] = csrGraph.numEdges;
    csrGraph.numEdges += adj[i].size();
  }
  csrGraph.srcPtrs[numVertices] = csrGraph.numEdges;

  csrGraph.dst =
      (unsigned int *)malloc(csrGraph.numEdges * sizeof(unsigned int));
  unsigned int edgeIdx = 0;
  for (unsigned int i = 0; i < numVertices; ++i) {
    for (unsigned int neighbor : adj[i]) {
      csrGraph.dst[edgeIdx++] = neighbor;
    }
  }

  cooGraph.numVertices = numVertices;
  cooGraph.numEdges = csrGraph.numEdges;
  cooGraph.src =
      (unsigned int *)malloc(cooGraph.numEdges * sizeof(unsigned int));
  cooGraph.dst =
      (unsigned int *)malloc(cooGraph.numEdges * sizeof(unsigned int));
  unsigned int cooEdgeIdx = 0;
  for (unsigned int i = 0; i < numVertices; ++i) {
    for (unsigned int neighbor : adj[i]) {
      cooGraph.src[cooEdgeIdx] = i;
      cooGraph.dst[cooEdgeIdx] = neighbor;
      cooEdgeIdx++;
    }
  }
}

int main(int argc, char **argv) {
  int graph_type = 0; // 0=Uniform, 1=R-MAT, 2=File
  if (argc > 1)
    graph_type = atoi(argv[1]);

  srand(12345); // 固定种子以便比对

  CSRGraph csrGraph;
  COOGraph cooGraph;
  unsigned int numVertices = 0;
  std::vector<std::pair<unsigned int, unsigned int>> edges;

  if (graph_type == 0) {
    printf("--- Generating Uniform Random Graph ---\n");
    numVertices = 1 << 18;
    unsigned int targetEdges = 1 << 23;
    for (unsigned int i = 0; i < targetEdges; ++i) {
      unsigned int u = rand() % numVertices;
      unsigned int v = rand() % numVertices;
      edges.push_back({u, v});
      edges.push_back({v, u});
    }
  } else if (graph_type == 1) {
    printf("--- Generating R-MAT Power-law Graph ---\n");
    numVertices = 1 << 18;
    unsigned int edgeFactor = 32;
    generate_rmat_edges(18, edgeFactor, edges);
  } else if (graph_type == 2) {
    if (argc < 3) {
      printf("Usage: ./graph 2 <filename>\n");
      return 1;
    }
    printf("--- Loading Graph from File: %s ---\n", argv[2]);
    load_edges_from_file(argv[2], numVertices, edges);
  } else {
    printf("Invalid graph type.\n");
    return 1;
  }

  build_graphs_from_edges(numVertices, edges, csrGraph, cooGraph);
  printf("Graph built: Vertices = %u, Total Directed Edges = %u\n",
         csrGraph.numVertices, csrGraph.numEdges);

  unsigned int *level =
      (unsigned int *)malloc(csrGraph.numVertices * sizeof(unsigned int));
  for (unsigned int i = 0; i < csrGraph.numVertices; ++i) {
    level[i] = UINT_MAX;
  }

  // 选一个有出度的源节点，防止瞬间BFS结束
  unsigned int source;
  do {
    source = rand() % csrGraph.numVertices;
  } while (csrGraph.srcPtrs[source] == csrGraph.srcPtrs[source + 1]);
  printf("Selected BFS Source: %u\n", source);

  level[source] = 0;

  unsigned int *level1 =
      (unsigned int *)malloc(csrGraph.numVertices * sizeof(unsigned int));
  unsigned int *level2 =
      (unsigned int *)malloc(csrGraph.numVertices * sizeof(unsigned int));
  unsigned int *level3 =
      (unsigned int *)malloc(csrGraph.numVertices * sizeof(unsigned int));
  unsigned int *level4 =
      (unsigned int *)malloc(csrGraph.numVertices * sizeof(unsigned int));
  unsigned int *level5 =
      (unsigned int *)malloc(csrGraph.numVertices * sizeof(unsigned int));
  unsigned int *level6 =
      (unsigned int *)malloc(csrGraph.numVertices * sizeof(unsigned int));
  unsigned int *level7 =
      (unsigned int *)malloc(csrGraph.numVertices * sizeof(unsigned int));
  memcpy(level1, level, csrGraph.numVertices * sizeof(unsigned int));
  memcpy(level2, level, csrGraph.numVertices * sizeof(unsigned int));
  memcpy(level3, level, csrGraph.numVertices * sizeof(unsigned int));
  memcpy(level4, level, csrGraph.numVertices * sizeof(unsigned int));
  memcpy(level5, level, csrGraph.numVertices * sizeof(unsigned int));
  memcpy(level6, level, csrGraph.numVertices * sizeof(unsigned int));
  memcpy(level7, level, csrGraph.numVertices * sizeof(unsigned int));

  printf("\nStarting Benchmarks...\n");

  unsigned int *level_cpu =
      (unsigned int *)malloc(csrGraph.numVertices * sizeof(unsigned int));
  auto start_cpu = std::chrono::high_resolution_clock::now();
  bfs_cpu(csrGraph, source, level_cpu);
  auto end_cpu = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> cpu_ms = end_cpu - start_cpu;
  printf("BFS (CPU): %f ms\n", cpu_ms.count());

  bfs_coo(cooGraph, source, level);
  bfs_csr(csrGraph, source, level1, 't');
  bfs_csr(csrGraph, source, level2, 'b');
  bfs_csr(csrGraph, source, level3, 'd');
  bfs_csr(csrGraph, source, level4, 'f');
  bfs_csr(csrGraph, source, level5, 'c');
  bfs_csr(csrGraph, source, level6, 'i');
  bfs_csr(csrGraph, source, level7, 'p');
  // Verify the results
  bool match = true;
  for (unsigned int i = 0; i < csrGraph.numVertices; ++i) {
    if (level_cpu[i] != level[i] || level_cpu[i] != level1[i] ||
        level_cpu[i] != level2[i] || level_cpu[i] != level3[i] ||
        level_cpu[i] != level4[i] || level_cpu[i] != level5[i] ||
        level_cpu[i] != level6[i] || level_cpu[i] != level7[i]) {
      printf("Mismatch at vertex %u: CPU=%u, COO=%u, CSR_T=%u, CSR_B=%u, "
             "CSR_D=%u, CSR_F=%u, CSR_C=%u, CSR_I=%u, CSR_P=%u\n",
             i, level_cpu[i], level[i], level1[i], level2[i], level3[i],
             level4[i], level5[i], level6[i], level7[i]);
      match = false;
      break;
    }
  }
  if (match) {
    printf("All policies matched correctly!\n");
  }

  free(level1);
  free(level2);
  free(level3);
  free(level4);
  free(level5);
  free(level6);
  free(level7);
  free(level_cpu);

  free(csrGraph.srcPtrs);
  free(csrGraph.dst);
  free(level);
  free(cooGraph.src);
  free(cooGraph.dst);
  return 0;
}
