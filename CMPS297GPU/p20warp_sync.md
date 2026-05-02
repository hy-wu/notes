
##### intra-warp synchronization

take advantage of the special relationship between threads in the same warp to synchronize quickly

CUDA has several built-in functions to synchronize between warps
- sharing (shuffling) data across threads
- voting across threads

###### shuffle

built-in warp shuffle functions enable threads to share data with other threads in the same warp
  - faster than shared memory and `__syncthreads()` cross the block
- variants
  - `__shfl_sync()`: direct copy from indexed lane
  - `__shfl_up_sync()`: copy from a lane with lower ID relative to the caller
  - `__shfl_down_sync()`: copy from a lane with higher ID relative to the caller
  - `__shfl_xor_sync()`: copy from a lane based on a bitwise XOR of own lane ID

example: reduction/scan
- When one warp remains, use `__shfl_down_sync()` to share data from registers

###### voting

built-in warp vote functions enable threads to vote on which threads in the warp satisfy some condition
  - `__all_sync(unsigned mask, predicate)`: check if predicate is nonzero for all threads
  - `__any_sync(unsigned mask, predicate)`: check if predicate is nonzero for any thread
  - `__ballot_sync(unsigned mask, predicate)`: return mask of which threads evaluate predicate to nonzero
  - `__activemask()`: return mask of which threads are active

example: enqueue, have only one thread in the warp increment the counter on behalf of the others
1. assign a leader thread
  - find the active threads in the warp using `__activemask()`
  - disignate the first active thread as the leader using `__ffs(mask) - 1`
    - index of `__ffs(mask)` starts from 1, returns 0 if no bits are set
2. find how many threads need to enqueue
  - use `__popc(mask)` to count the number of set bits in the mask
3. have the leader perform the atomic operation
  - threads in the same warp are consecutive, so `threadIdx.x % WARP_SIZE` gives the position in the warp
4. broadcast the result to the other threads
  - use `__shfl_sync(mask, j, leader)`
5. find offset of each active thread and store the result
  - use bitwise AND to find the number of previous active threads

Gemini's suggestions:
1. Replaced `__activemask()` with `__ballot_sync(0xffffffff, is_active)`: On modern GPU architectures (Volta and above), independent thread scheduling can cause `__activemask()` to return only a subset of active threads taking a divergent branch. Using `__ballot_sync` guarantees all evaluating threads in the warp are grouped together, preventing multiple partial `atomicAdd`s.
2. Changed `1 << ...` to `1u << ...`: This avoids undefined behavior from signed integer overflow when the thread index is 31.
