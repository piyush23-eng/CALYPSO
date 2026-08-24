# GATE Computer Science - Algorithms and Data Structures

## Asymptotic Analysis and Recurrence Relations

### Master Theorem
For recurrences of form $T(n) = a T(n/b) + f(n)$ where $a \ge 1, b > 1$:
Let $c = \log_b a$:
1. If $f(n) = O(n^{c - \epsilon})$ for some $\epsilon > 0$, then $T(n) = \Theta(n^{\log_b a})$.
2. If $f(n) = \Theta(n^c \log^k n)$ for $k \ge 0$, then $T(n) = \Theta(n^{\log_b a} \log^{k+1} n)$.
3. If $f(n) = \Omega(n^{c + \epsilon})$ for some $\epsilon > 0$ and regularity condition $a f(n/b) \le d f(n)$ holds for $d < 1$, then $T(n) = \Theta(f(n))$.

## Graph Algorithms and Dynamic Programming

### Shortest Path Algorithms
1. **Dijkstra's Algorithm**: Single-source shortest path for non-negative edge weights. Time complexity with binary heap is $O((V + E) \log V)$, with Fibonacci heap $O(E + V \log V)$.
2. **Bellman-Ford Algorithm**: Single-source shortest path handling negative edge weights and detects negative weight cycles. Time complexity $O(V \cdot E)$.
3. **Floyd-Warshall Algorithm**: All-pairs shortest paths using dynamic programming. Recurrence:
   $$D_{i,j}^{(k)} = \min(D_{i,j}^{(k-1)}, D_{i,k}^{(k-1)} + D_{k,j}^{(k-1)})$$
   Time complexity is $\Theta(V^3)$ and space is $\Theta(V^2)$.

### Minimum Spanning Trees (MST)
- **Kruskal's Algorithm**: Greedy edge-based using Disjoint Set Union (Union-Find with path compression and rank). Time complexity $O(E \log E) = O(E \log V)$.
- **Prim's Algorithm**: Greedy vertex-growth algorithm. Time complexity $O(E \log V)$ with min-heap.

### Heap Data Structures
- **Building a Max-Heap**: Bottom-up heap construction using Floyd's algorithm takes $O(n)$ time due to summation $\sum_{h=0}^{\lfloor \log n \rfloor} \frac{n}{2^{h+1}} O(h) = O(n)$.
- **Heapify**: Takes $O(\log n)$ time per element.
- **HeapSort**: Takes $O(n \log n)$ time in best, worst, and average cases with $O(1)$ auxiliary space.
