stencil computation

- 5-point, 2D; 7-point, 3D
- grid of points stored as 2D or 3D array
- parallelizing
  - (assuming only inputs in bound)
  - Tiling
    - limited (max 1024 threads)
  - Thread coarsening
    - plane by plane
    - only the current slice is truly shared, save shared memory by putting them in registers: **register tiling**



