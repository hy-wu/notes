scan

scan operation
- takes
  - an input array $[x_0, x_1, \ldots, x_{n-1}]$
  - an associative operation $\oplus$, e.g. $+$, $\times$, $\max$, $\min$
- returns
  - an output array $[y_0, y_1, \ldots, y_{n-1}]$
    - inclusive scan: $y_i = x_0 \oplus x_1 \oplus \ldots \oplus x_i$
    - exclusive scan: $y_i = x_0 \oplus x_1 \oplus \ldots \oplus x_{i-1}$

segmented scan (also called hierarchical scan)

![segmented scan](images/image.png)

Kogge-Stone parallel (inclusive) scan

![Kogge-Stone parallel (inclusive) scan](images/image-1.png)

- careful with data races
- `__syncthreads()` in branch: undefined behavior
- shared memory: buffer
- double buffering: don't have to synchronize between read and write

exclusive scan: shift by one, insert identity element at the beginning

work efficiency
- A parallel algorithm is work efficient if it performs the same amount of work as the corresponding sequential algorithm

scan work efficiency
- sequential scan: $N$ additions
- Kogge-Stone parallel scan
  - $N-2^{step}$ operations per step, $\log N$ steps
  - total work: $N \log(N) - (N-1) = \mathcal{O}(N \log N)$ operations
  - not work efficient
- Brent-Kung parallel scan
  - Reduction: $\log(N)$ steps, $N-1$ operations
  - Post-reduction: $\log(N) - 1$ steps, $N-2 - (\log(N) - 1)$ operations
  - total: $2\log(N) - 1$ steps, $2N - \log(N) - 2 = \mathcal{O}(N)$ operations
  - takes more steps but more work efficient
