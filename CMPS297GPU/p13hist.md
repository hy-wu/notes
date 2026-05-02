histogram

Data races may result in unpredictable program output.

mutual exclusion
- CPU: locks (mutex)
- atomic operation: read-modify-write with a single ISA instruction, serialized by the hardware

atomic operations in CUDA
- atomic add: `T atomicAdd(T* address, T val)`
- function call translated to a single ISA instruction: instrinsics
- sub, min, max, inc, dec, and, or, xor, exchange, compare and swap

atomic operations on global memory have high latency
- wait for both read and write to complete
- wait if there are any other threads accessing the same location: high probablity of contension
- optimize: privatization, coarsening, local counter

**privatization**: multiple private copies of an output are maintained, then the global copy is updated on completion
- operations on the output must be associative and commutative

