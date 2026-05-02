
- Threads are assigned to SMs at block granularity. (All threads in a block are assigned to the same SM.) Each SM has a limited amount of resources.
- Threads within the same block can collaborate in ways that threads in different blocks cannot:
    - barrier synchronization (e.g., __syncthreads())
    - shared memory
    - others
- Thread scheduling: warp. 32 threads in a warp are scheduled together and executed following a SIMD model.
    - control divergence: branches, loops
    - latency hiding: when some threads are stalled, the SM can switch to another warp that is ready to execute.
- Occupancy constraints

| GPU Name       | K40    | M40    | P100   | V100   | A100? | RTX 3090? | RTX 4090? | H100? | RTX 5090? |
|----------------|--------|--------|--------|--------|---------------|-------------------|----------------|---------------|-----------------------|
| Microarchitecture | Kepler | Maxwell | Pascal | Volta | Ampere        | Ampere            | Ada Lovelace   | Hopper        | Blackwell             |
| Compute Capability | 3.5    | 5.2    | 6.0    | 7.0    | 8.0           | 8.6               | 8.9            | 9.0           | 12.0                  |
| Number of SMs  | 15     | 24     | 56     | 80     | 108           | 82                | 128            | 144           | 192                   |
| Cores per SM   | 192    | 128    | 64     | 64     | 64            | 64                | 128            | 128           | 128                   |
| Max threads per SM | 2048 | 2048 | 2048 | 2048 | 2048          | 2048              | 2048           | 2048          | 1536 (48 warps)       |
| Max blocks per SM | 16    | 32     | 32     | 32     | 32            | 16                | 32             | 32            | 32                    |
| Max threads per block | 1024 | 1024 | 1024 | 1024 | 1024          | 1024              | 1024           | 1024          | 1024                  |
| Registers per SM | 64K    | 64K    | 64K    | 64K    | 64K           | 64K               | 64K            | 64K           | 64K                   |
| Shared memory per SM | 48KB | 96KB | 64KB | 96KB | 164KB         | 128KB             | 128KB          | 228KB         | 256KB                 |

