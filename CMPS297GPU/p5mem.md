
(v100 era)

- performance metrics
  - FLOPS rate; compute-bound
  - memory bandwidth; memory-bound
- Example: matrix multiplication
  - for $N\times N$ matrices
    - Data loaded: $8N^2$ B
    - Operations: $2N^3$ OP
    - potential compute-to-memory ratio: $0.25 N$ OP/B
  - opportunity: memory accesses in the kernel code to reuse input data loaded from global memory as much as possible
- memory in GPU
  - global memory: $\approx 500$ cycles
  - L1 cache: $\approx 5$ cycles, managed by hardware
  - shared memory: $\approx 5$ cycles, managed by programmer
  - constant cache: $\approx 5$ cycles, managed by programmer
  - L2 cache
  - registers: $\approx 1$ cycle

![alt text](<images/截屏2026-04-28 下午2.16.06.png>)
![alt text](<images/截屏2026-04-28 下午2.20.55.png>)
The data gets evicted from the L1 cache before another thread tries to load it, more likely than CPU because of the large number of threads and the small size of the L1 cache.

Dynamic shared memory
- declaration: `extern __shared__ float A_s[];`
- configuration: `kernel <<< numBlocks, numThreadsPerBlock, smemPerBlock >>> (...)`
