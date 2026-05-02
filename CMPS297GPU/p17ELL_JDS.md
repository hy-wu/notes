
ELLPACK Format (ELL)
- pad each row to the same size
- store padded array in column-major order

SpMV/ELL
- assign one thread to loop over each input row sequentially

![SpMV ELL](image-12.png)

ELL tradeoffs
- advantages
  - flexibility: can add new elements as long as row not full
  - accessibility: given row, easy to find all nonzeros; given non-zero, easy to find its row and column
  - SpMV/ELL memory access is coalesced
- disadvantages
  - space efficiency: overhead due to padding
  - accessibility: given column, hard to find all nonzeros
  - SpMV/ELL has control divergence

Hybrid ELL + COO: use COO for very long rows

![Hybrid ELL COO](image-13.png)

- benefits
  - space efficient: less padding
  - flexibility: can add new elements to any row
- still has control divergence

Jagged Diagonal Storage (JDS)
- sort rows by size and remember the original row index
- store nonzeros in column major
- remember where the nonzeros of each iteration start (`IterPtr`)

SpMV/JDS
- assign one thread to loop over each input row sequentially and update corresponding output element

![SpMV JDS Kernel](image-14.png)

JDS tradeoffs
- advantages
  - space efficient: no padding
  - accessibility: given a row, easy to find all nonzeros
  - SpMV/JDS memory access is coalesced
  - SpMV/JDS minimizes control divergence
- disadvantages
  - flexibility: hard to add new elements to the matrix
  - accessibility: given nonzero, hard to find row; given a column, hard to find all nonzeros
