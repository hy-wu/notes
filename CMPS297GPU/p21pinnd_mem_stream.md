
###### Pinned Memory

Direct Memory Access (DMA) allows some hardware to access memory without involving the CPU
- copying between CPU and GPU uses DMA
- advantage: CPU can be utilized for other tasks while the memory copy is taking place
- DMA use physical addresses to access memory, cannot detect if the OS swaps a virtual page with another virtual page at the same physical address
- to avoid data corruption, the OS must not be allowed to swap pages being accessed by DMA
- pages to be accessed by DMA must be allocated as pinned memory (also called page-locked memory), marked as non-swappable by the OS

cudaMemcpy
- host to device: CPU copies data to a pinned memory buffer, DMA copies data from pinned memory buffer to the device
- device to host: DMA copies data from the device to a pinned memory buffer, CPU copies data from pinned memory buffer
- disadvantage: every cudaMemcpy is actually two copies, which incurs overhead

faster copies: allocate host arrays directly in pinned memory
- `cudaError_t cudaMallocHost(void **devPtr, size_t size)`
- `cudaError_t cudaFreeHost(void *devPtr)`
- use with caution: only use for frequently copied data if overhead is high
- allocating too much pinned memory degrades overall system performance by reducing the virtual pages available for swapping

###### Stream

Typical system is simultaneously capable of
- executing grids on the GPU
- copying from host to device
- copying from device to host

Example: previous timeline of vector addition
- copy A to device
- copy B to device
  - not utilizing GPU and copyDeviceToHost
- execute grid to compute A+B
  - not utilizing copies
- copy C from device
  - not utilizing GPU and copyHostToDevice

pipelining vector addition
- divide arrays into segments
- overlap copy and computation of different segments

![Pipelining Vector Addition](image-21.png)

streams and asynchronous copies
- parallelism between grids and memory copies is achieved by using **streams**
  - tasks in different streams are executed in parallel
  - tasks in the same stream are serialized
  - if no stream is specified, tasks go into the default stream
- to overlap memory copies with host execution, use **asynchronous memory copies**
- `cudaError_t cudaStreamCreate(cudaStream_t *pStream)`
- `cudaError_t cudaMemcpyAsync(void *dst, const void *src, size_t count, cudaMemcpyKind kind, cudaStream_t stream=0)`
- `kernel<<<blocks, threads, smem, stream>>>(...)`
- cudaEvent_t can be used to measure the time taken in different streams


