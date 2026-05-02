convolution

usage: signal/image/video processing, e.g., Gaussian blur, sharpening, edge detection, etc.

a straightforward implementation: a thread per output element

observation: the mask is small, constant, and accessed by all threads,
store in constant memory
- `__constant__ float mask_c[MASK_DIM][MASK_DIM];`
- `cudaMemcpyToSymbol(mask_c, mask, MASK_DIM * MASK_DIM * sizeof(float));`
- can only allocate up to 64KB
- cache coherency

tiling (M*M mask)
- without tiling, ratio of computation to global loads: (2M^2 OP) / (4M^2 B) = 0.5 OP/B
- with tiling, dimensions: input = T, output = T-M+1
  - operations per block: (T-M+1)^2 * 2M^2 OP
  - global loads per block: T^2 * 4 B
  - ratio: 0.5M^2 * (1 - (M-1)/T)^2 OP/B

