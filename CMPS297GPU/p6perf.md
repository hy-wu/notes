some Performance Optimizations

- Tuning resource usage to maximize occupancy
  - Threads per block, shared memory per block, registers per thread [p4]
- Minimizing control divergence to increase SIMD efficiency [p4]
- Shared memory tiling to increase data reuse [p5]
- Memory coalescing
- Thread coarsening

DRAM cell: consists of a capacitance that stores a charge and a three-state device (single-transistor) that allows data to be read/written (capacitor discharged/charged).

DRAM array, DRAM bank.

Row Address -> [Row Decoder] -> [DRAM Array] -> [Sense Amplifier] -> [Column Latches] -(burst)> [with Column Address -> Mux] -> Data

Accessing data in the same burst: no need to access the DRAM array again, just the multiplexer.

Memory Coalescing
- When threads in the same warp access consecutive memory locations in the same burst, the accesses can be combined and served by one burst
  - One DRAM transaction is needed
  - Known as **memory coalescing**
  - Appendix: threads with the same `y` value but consecutive `x` values will go to the same warp
- If threads in the same warp access locations not in the same burst, accesses cannot be combined
  - Multiple transactions are needed
  - Takes longer to service data to the warp
  - Sometimes called **memory divergence**

Latency Hiding with Multiple Banks
- Latency can be hidden by having multiple banks
- Need many threads to simultaneously access memory to keep all banks busy
  - by having high occupancy in SMs

Thread Granularity
- So far, parallelization approaches made threads as fine-grain as possible
  - Assign smallest possible unit of parallelism per thread
    - e.g., one vector element per thread in vector addition
    - e.g., one output pixel per thread in RGB to gray and in blur
    - e.g., one output element per thread in matrix multiplication
  - Advantages: provide hardware with as many threads as possible to fully utilize resources
    - transparent scalability
  - Disadvantages: common operations are performed redundantly across more threads
    - Suboptimal if threads are getting serialized by hardware
- Thread coarsening: assign multiple units of parallelism per thread
  - Advantages: reduce the price paid for parallelization
    - redundant memory
    - redundant computation
    - synchronization overhead or divergence
  - Disadvantages
    - underutilize resources
      - interfere with the transparent scalability
    - more resources per thread may limit occupancy

Checklist of Common Optimizations

| Optimization | Improves | Impact on Cores | Impact on Memory |
|--------------|----------|-----------------|------------------|
| Tuning resources to maximize occupancy | Latency hiding | More work to hide pipeline latency | More accesses to hide DRAM latency |
| Minimizing control divergence | SIMD efficiency | Fewer idle cores during SIMD execution | - |
| Uniform memory access pattern | Memory coalescing | Fewer pipeline stalls waiting for memory | Better utilization of bursts / cache lines |
| Shared memory tiling | Data reuse | Fewer pipeline stalls waiting for memory | Less global memory traffic |
| Thread coarsening | Price of parallelization | Less redundant work or synchronization | Less global memory traffic |
| Privatization *(covered later)* | Contention on common output | Fewer pipeline stalls waiting for memory | Less global memory traffic and serialization |

Tensions Between Optimizations
- Maximizing occupancy
  - Maximizing occupancy hides pipeline latency, but too many threads may compete for the cache, evicting each others' data (thrashing the cache)
- Shared memory tiling
  - Using more shared memory enables more data reuse, but may limit occupancy
- Thread coarsening
  - Coarsening reduces redundant work, but requires more resources per thread which may limit occupancy

Need to find the sweet spot that achieves the best compromise

Know Your Bottlenecks
- Optimizations trade one resource for another to alleviate the bottleneck
- Make sure to properly diagnose the bottleneck before applying optimizations
