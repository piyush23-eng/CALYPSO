# GATE Computer Science - Operating Systems Reference Notes

## Process Management and CPU Scheduling

### Process States and Transitions
A process in an operating system moves through several states during its execution lifecycle:
- **New**: The process is being created.
- **Ready**: The process is loaded into main memory and is waiting to be assigned to a CPU.
- **Running**: Instructions are currently being executed by the CPU.
- **Waiting (Blocked)**: The process is waiting for an I/O completion or event signal.
- **Terminated**: The process has finished execution and operating system resources are reclaimed.

### CPU Scheduling Algorithms
CPU scheduling decides which ready process gets CPU time:
1. **First-Come, First-Served (FCFS)**: Non-preemptive. Suffers from the *convoy effect* where short processes wait behind a long CPU burst process.
2. **Shortest Job First (SJF)**: Minimizes average waiting time for a given set of stationary processes. Non-preemptive.
3. **Shortest Remaining Time First (SRTF)**: Preemptive version of SJF. Provably optimal in terms of minimizing average waiting time.
4. **Round Robin (RR)**: Preemptive scheduling designed for time-sharing systems using a fixed *time quantum*. If time quantum is very large, RR degenerates to FCFS. If too small, context switch overhead dominates.
5. **Priority Scheduling**: Can be preemptive or non-preemptive. Suffers from *starvation* (indefinite blocking), which is mitigated by *aging*.

## Memory Management and Virtual Memory

### Paging and Page Tables
Paging is a memory management scheme that eliminates the need for contiguous physical memory allocation:
- **Logical Address**: Divided into Page Number ($p$) and Page Offset ($d$).
- **Physical Address**: Divided into Frame Number ($f$) and Frame Offset ($d$).
- **Page Table**: Translates logical page numbers to physical frame numbers. The page table itself is stored in main memory.
- **Translation Lookaside Buffer (TLB)**: High-speed associative cache holding recent page translations.
- **Effective Memory Access Time (EMAT)**:
  $$EMAT = h \cdot (t_{TLB} + t_m) + (1 - h) \cdot (t_{TLB} + 2 \cdot t_m)$$
  where $h$ is the TLB hit ratio, $t_{TLB}$ is TLB access time, and $t_m$ is main memory access time.

### Page Replacement Algorithms
When a page fault occurs and no free frames exist:
- **FIFO (First-In, First-Out)**: Replaces oldest page. Subject to *Belady's Anomaly* (more frames can lead to more page faults).
- **Optimal (OPT / Belady's Optimal)**: Replaces the page that will not be used for the longest period of time. Benchmark algorithm; cannot be implemented in general purpose OS due to lack of future knowledge.
- **Least Recently Used (LRU)**: Replaces the page that has not been referenced for the longest time. Stack algorithm; immune to Belady's Anomaly.

## Concurrency, Semaphores and Deadlocks

### Critical Section Problem
A solution must satisfy three requirements:
1. **Mutual Exclusion**: If process $P_i$ is executing in its critical section, no other processes can execute in their critical sections.
2. **Progress**: If no process is in critical section and some wish to enter, selection cannot be postponed indefinitely.
3. **Bounded Waiting**: There must be a bound on the number of times other processes are allowed to enter their critical sections after a process has made a request.

### Deadlock Characterization
Four Coffman conditions must hold simultaneously for a deadlock:
1. Mutual Exclusion
2. Hold and Wait
3. No Preemption
4. Circular Wait

### Banker's Algorithm
Used for deadlock avoidance by verifying whether granting a resource request leaves the system in a *Safe State*.
- Need Matrix calculation: $Need[i][j] = Max[i][j] - Allocation[i][j]$.
