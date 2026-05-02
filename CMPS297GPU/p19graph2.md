
Remark: if not symmetric, top-down BFS need CSR representation, while bottom-up BFS need CSC representation, (treating rows as source) dir-opt BFS need both CSR and CSC.

Reduce Redundancy
- objective: only check the vertices that are part of the previous level
- approach: each level places the vertices it visits in a queue for the next level to process
  - vertices enqueued at each level form that level's frontier
- overhead: synchronization across threads to add to a shared queue

![Vertex-Centric Frontier-Based BFS](images/image-19.png)

Queue Privatization
- all threads atomically increment the same global counter to insert elements into the queue, high latency due to global memory access and serialization due to high contention.
- optimization: each thread block maintains a private queue (in shared memory, with the counter) and commits entries to the global queue on completion

Minimizing Launch Overhead
- need to synchronize between levels
- so far, we have launched a new grid for each level
  - overhead of launching new grid and copying counter
- optimization: if consecutive levels have few enough vertices that can be executed by one thread block:
  - execute multiple levels in one single-block grid and synchronize between levels using __syncthreads()
  - reduce total number of grid launches

![Minimizing Launch Overhead](images/image-20.png)


