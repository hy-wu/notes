
Reduction

reduction operation: reduces a set of values to one value, is
- associative
- commutative
- have a well-defined identity value

reduction is a fold

parallel reduction
- reduction tree: 3every thread adds two elements in each step
  - $log(N)$ steps and half the threads drop out every step
- segmented reduction
  - tree reduction on each segment -> partial sums
    - using stride as loop variable
  - then
    - launch a new reduction kernel, or
    - using atomic operations, or
    - add them on the CPU
  - problems
    - SIMD: control divergence
    - memory access pattern: not coalesced
- optimization
  - coalesced access: stride from BLOCK_DIM to 1
  - using shared memory
  - thread coarsening
    - price paid for parallelism: are synchronization and control divergence
    - each thread responsible for more data

![thread coarsening](images/reduction_coarsening.png)
- Coarsening Benefits
  - If blocks are all executed in parallel:
    - $\log(N)$ steps, $\log(N)$ synchronizations
  - If blocks serialized by the hardware by a factor of C:
    - $C*\log(N)$ steps, $C*\log(N)$ synchronizations
  - If blocks are coarsened by a factor of C:
    - $2*(C - 1) + \log(N)$ steps, $\log(N)$ synchronizations

