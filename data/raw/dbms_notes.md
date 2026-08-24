# GATE Computer Science - Database Management Systems (DBMS)

## Relational Model and Normalization

### Functional Dependencies and Keys
- A functional dependency $X \to Y$ holds on relation $R$ if tuples agreeing on attribute set $X$ must also agree on attribute set $Y$.
- **Candidate Key**: A minimal superkey. A set of attributes $K$ is a candidate key if $K^+ = R$ and no proper subset of $K$ has closure equal to $R$.
- **Superkey**: Any superset of a candidate key.

### Normal Forms
1. **First Normal Form (1NF)**: All attribute domains are atomic (no multi-valued or composite attributes).
2. **Second Normal Form (2NF)**: In 1NF and no non-prime attribute is partially dependent on any candidate key (no partial dependency).
3. **Third Normal Form (3NF)**: For every non-trivial functional dependency $X \to Y$, either $X$ is a superkey or $Y$ is a prime attribute (part of some candidate key). 3NF preserves dependencies and achieves lossless join.
4. **Boyce-Codd Normal Form (BCNF)**: For every non-trivial functional dependency $X \to Y$, $X$ must be a superkey. Strictly stronger than 3NF, but may not preserve functional dependencies.

## Transactions and Concurrency Control

### ACID Properties
- **Atomicity**: All-or-nothing execution guaranteed by transaction recovery manager (write-ahead logging).
- **Consistency**: Execution preserves database integrity constraints.
- **Isolation**: Concurrent transactions execute as if in isolation (handled by concurrency control manager).
- **Durability**: Committed updates persist across system crashes.

### Serializability and Schedules
- **Conflict Serializability**: A schedule is conflict serializable if it is conflict equivalent to a serial schedule. Checked via precedence (serialization) graph cycle detection.
- **View Serializability**: Weaker condition than conflict serializability; NP-complete in general. Blind writes allow view serializability without conflict serializability.
- **Two-Phase Locking (2PL)**:
  - Growing phase (locks acquired, none released) and Shrinking phase (locks released, none acquired).
  - Guarantees conflict serializability.
  - **Strict 2PL**: Holds all exclusive locks until transaction commit/abort (avoids cascading aborts).
  - **Rigorous 2PL**: Holds all shared and exclusive locks until commit/abort (produces strict serializable schedules).

## Indexing and B/B+ Trees

### B+ Tree Properties
A B+ tree of order $m$ satisfies:
- All data records/pointers reside strictly in the leaf nodes.
- Leaf nodes are linked as a doubly-linked list for efficient range scans.
- Root node has at least 2 children (unless it is a leaf).
- Internal nodes hold between $\lceil m/2 \rceil$ and $m$ child pointers.
- Maximum number of keys in internal node: $m - 1$. Minimum keys: $\lceil m/2 \rceil - 1$.
