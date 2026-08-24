# GATE Computer Science - Previous Year Solved Questions (PYQ Archive)

## Question 1: OS CPU Scheduling [GATE 2023 || 2 Marks]
**Topic**: Operating Systems
**Subtopic**: CPU Scheduling
**Question**:
Consider three processes P1, P2, and P3 arriving at time 0 with CPU burst times of 10 ms, 5 ms, and 2 ms respectively. If Shortest Remaining Time First (SRTF) scheduling is used, what is the average waiting time of the processes?
Options:
(A) 2.33 ms
(B) 4.0 ms
(C) 5.66 ms
(D) 3.0 ms

**Answer and Reasoning**:
- At time t = 0, all three processes arrive: P1 (10 ms), P2 (5 ms), P3 (2 ms).
- SRTF selects P3 (burst 2 ms), finishing at t = 2. Waiting time for P3 = 0.
- Next, SRTF selects P2 (burst 5 ms), running from t = 2 to t = 7. Waiting time for P2 = 2 - 0 = 2 ms.
- Finally, P1 runs from t = 7 to t = 17. Waiting time for P1 = 7 - 0 = 7 ms.
- Average waiting time = (0 + 2 + 7) / 3 = 9 / 3 = 3.0 ms.
Correct Answer: (D) 3.0 ms

## Question 2: DBMS Normalization [GATE 2022 || 2 Marks]
**Topic**: Database Management Systems
**Subtopic**: Normalization
**Question**:
Given relation R(A, B, C, D, E) with functional dependencies F = {A -> BC, CD -> E, B -> D, E -> A}. What is the highest normal form satisfied by relation R?
Options:
(A) 1NF
(B) 2NF
(C) 3NF
(D) BCNF

**Answer and Reasoning**:
- Finding candidate keys:
  - (A)+ = {A, B, C, D, E}, so A is a candidate key.
  - (E)+ = {E, A, B, C, D}, so E is a candidate key.
  - (CD)+ = {C, D, E, A, B}, so CD is a candidate key.
  - (BC)+ = {B, C, D, E, A}, so BC is a candidate key.
- Prime attributes: {A, B, C, D, E} (all attributes are prime!).
- Check normal forms:
  - Since all attributes are prime, no non-prime attribute exists, so partial dependency cannot occur -> relation is in 2NF.
  - In 3NF condition for X -> Y: either X is superkey or Y is prime attribute. Since every attribute in R is prime, every FD trivially satisfies the second condition of 3NF.
  - For BCNF: in B -> D, B is not a superkey. Thus R is not in BCNF.
- Highest normal form is 3NF.
Correct Answer: (C) 3NF

## Question 3: Algorithms Heap Construction [GATE 2021 || 1 Mark]
**Topic**: Algorithms
**Subtopic**: Heaps and Sorting
**Question**:
What is the worst-case time complexity of constructing a binary max-heap of n elements from an unsorted array of n elements?
Options:
(A) O(n log n)
(B) O(n)
(C) O(log n)
(D) O(n^2)

**Answer and Reasoning**:
- Bottom-up heap construction invokes Max-Heapify from index floor(n/2) down to 1.
- Number of nodes at height h is at most ceil(n / 2^(h+1)).
- Time complexity is bounded by: Sum_{h=0}^{floor(log n)} (n / 2^(h+1)) * O(h) = O(n * Sum_{h=0}^{inf} h / 2^h) = O(n).
Correct Answer: (B) O(n)

## Question 4: Operating Systems Memory Management [GATE 2024 || 2 Marks]
**Topic**: Operating Systems
**Subtopic**: Virtual Memory and TLB
**Question**:
A system uses a 2-level page table. Main memory access time is 100 ns and TLB access time is 20 ns. If the TLB hit ratio is 90%, what is the Effective Memory Access Time (EMAT)?
Options:
(A) 120 ns
(B) 140 ns
(C) 150 ns
(D) 130 ns

**Answer and Reasoning**:
- In a 2-level paging system:
  - On TLB hit: access TLB + 1 main memory access for actual data = 20 + 100 = 120 ns.
  - On TLB miss: access TLB + 2 memory accesses for 2-level page tables + 1 memory access for data = 20 + 3 * 100 = 320 ns.
- EMAT = 0.90 * (20 + 100) + 0.10 * (20 + 300) = 0.90 * 120 + 0.10 * 320 = 108 + 32 = 140 ns.
Correct Answer: (B) 140 ns

## Question 5: Database Transactions [GATE 2023 || 1 Mark]
**Topic**: Database Management Systems
**Subtopic**: Concurrency Control
**Question**:
Which of the following concurrency control protocols guarantees conflict serializability and eliminates cascading aborts?
Options:
(A) Basic 2-Phase Locking (2PL)
(B) Strict 2-Phase Locking (Strict 2PL)
(C) Conservative 2-Phase Locking
(D) Timestamp Ordering Protocol

**Answer and Reasoning**:
- Basic 2PL ensures conflict serializability but may suffer from cascading rollbacks if a transaction reads uncommitted data written by a failed transaction.
- Strict 2PL requires that all exclusive (write) locks held by a transaction be released only after the transaction commits or aborts.
- This prevents dirty reads and cascading rollbacks while guaranteeing conflict serializability.
Correct Answer: (B) Strict 2-Phase Locking (Strict 2PL)
