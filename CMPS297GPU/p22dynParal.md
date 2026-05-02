
Dynamic Parallism: CUDA dynamic parallelism refers to the ability of threads executing on the GPU to launch new grids

Nested Parallelism: Dynamic parallelism is useful for programming applications with nested parallelism, where each thread discovers more work that can be parallelized; particularly useful when the amount of nested work is unknown, so enough threads cannot be launched up front.

applications
- nested parallel work is **irregular** (varies cross threads)
  - e.g. graph algorithms (eacb vertex has a different number of neighbors)
  - e.g. Bezier curve (each curve needs different number of points to draw)
- nested parallel work is **recursive** with unknown depth
  - e.g. tree traversal algorithms (e.g., quadtrees and octrees)
  - e.g. divide and conquer algorithms (e.g., quicksort)

![Applications of Dynamic Parallism](image-22.png)

- the device code for calling a kernel to launch a grid is the same as the host code
- memory is needed for buffering grid launches that have not started executing
  - The limit on the number of dynamic launches is referred to as the **pending launch count**
  - By default, the runtime supports 2048 launches, and exceeding this limit will cause an error
  - The limit can be increased by setting `cudaDeviceSetLimit(cudaLimitDevRuntimePendingLaunchCount, <new limit>)`

example: graph frontier BFS
- Directly launching child_kernel behaves worse
1. for device launches, threads in the same block share the same default stream (serialized)
- improve by creating a different stream per thread
  - use `cudaStreamCreate` as host, or
  - use compiler flag `--default-stream per-thread`
2. Pitfalls
  - launching very small grids may not be worth the overhead (more efficient to serialize)
  - launching too many grids causes queuing delays
3. Optimization: apply a threshold to the launch
  - only launch the large grids that are worth the overhead, and serialize the rest
  - threshold value is data dependent and can be tuned
4. Optimization: aggregate launched
  - have one thread collect the work of multiple threads and launch a single grid on their behalf

Offloading Driver Code
- free up the host to do other things
- notice: for single thread, CPU code is faster than GPU code, so it's unworthy in many cases

Memory Visibility
- Operations on global memory made by a parent thread before the launch are visible to the child threads
- Operations made by the child are visible to the parent after the child returns and the parent has synchronized
- A thread's local memory and a block's shared memory cannot be accessed by child threads

Nesting Depth: how deeply dynamically launched grids may launch other grids
- determined by the hardware (typically 24)





