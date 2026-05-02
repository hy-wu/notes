
ordered merge

parallelization
- each thread for a segment of output array
- finding input segments, binary search co-rank $i$ ($j=k-i$)
  - bound on $i$: $\max(0,k-n)\leq i \leq \min(k,m)$
  - guess is correct: $A[i-1]\leq B[j]$ and $B[j-1]\leq A[i]$
  - guess is too high: $A[i'-1]>B[j']$
  - guess is too low: $B[j''-1]>A[i'']$

![Finding Input Segments](images/image-8.png)

memory accesses are not coalesced, optimization
- load the block's segment to shared memory
- do the per-thread co-rank and merge in shared memory
- already applied thread coarsening

![Shared Memory Tiling](images/image-9.png)
