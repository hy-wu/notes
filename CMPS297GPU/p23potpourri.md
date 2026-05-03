
##### Multi-GPU programming

multiple GPUs on the same node
- `cudaGetDeviceCount`
- `cudaSetDevice` to select one
- typically use multiple CPU threads with one driving each GPU (e.g. using OpenMP)

multiple GPUs on multiple different nodes
- typically use MPI for programming multiple nodes, and each MPI rank interacts with its GPU normally
- important consideration: overlapping MPI communication with kernel execution

##### Interconnect

- PCIe: general, widely supported
- NVLink: faster between multi-GPUs, but only supported on some GPUs and some CPUs

##### Memory management

Unified Virtual Addressing (UVA)
- $\text{CPU} \longleftrightarrow \substack{\boxed{\text{Main Memory} \longleftrightarrow \text{Global Memory}}\\\text{UVA space}} \longleftrightarrow \text{GPU}$
- advantage: data location is known from the pointer value, so do not need to specify copy direction
  - `cudaMemcpyDefault`
- simpler code
- performance depends on situation

Zero-copy memory: enables devices to directly access host memory
- memory has to be pinned
- `cudaHostAlloc(&ptr, size, cudaHostAllocMapped)` to allocate the memory
- `cudaHostGetDevicePointer` to get corresponding device pointer
  - unnecessary if using UVA

##### Events

- synchronizing after every copy or kernel call interferes with execution and slows things down
- Events can be used to collect timing information without synchronizing
  - `cudaEventCreate(&event)`
  - `cudaEventRecord(event, stream=0)`
  - `cudaEventSynchronize(event)`
  - `cudaEventElapsedTime(&result, startEvent, endEvent)`
  - `cudaEventDestroy(event)`
- profilers can also be used

##### Tensor cores

programmable matrix multiplication and accumulation units
- accelerate DNN workloads
- in Volta v100, 640 cores (8 per SM), each core is capable of a 4x4 matrix multiply-accumulate operation D = A * B + C

##### Libraries

examples of libraries for common parallel primitives
- Thrust: reduction, scan, filter, and sort
- cuBLAS: basic dense linear algebra
  - NVBLAS: multi-GPU library on top of cuBLAS
- cuSPARSE: sparse-dense linear algebra
- cuSOLVER: factorization and solve routines, on top of cuBLAS and cuSPARSE
- cuDNN: deep neural networks
  - used by Caffe, TensorFlow, PyTorch, MxNet, etc.
- nvGRAPH: graph processing
- cuFFT: Fast Fourier Transform
- NPP: signal, image, and video processing

##### Other programming interfaces

- OpenCL: similar to CUDA, but open source and more portable across non-NVIDIA GPUs
- OpenACC: directive-based GPU programming
  - annotate code similar to OpenMP
- C++AMP: implementing GPU programs directly in C++

##### Other hardware

other GPUs
- AMD (Radeon)
- ARM (Mali)
- Intel

integrated GPUs

FPGA

##### Comparison with CPU

more fair to compare with parallel and vectorized CPU code
