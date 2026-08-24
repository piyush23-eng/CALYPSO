# GATE Computer Science - Theory of Computation and Compiler Design

## Theory of Computation

### Chomsky Hierarchy and Automata
1. **Regular Languages (Type 3)**:
   - Recognized by Deterministic/Nondeterministic Finite Automata (DFA/NFA).
   - Closed under union, intersection, complementation, concatenation, Kleene star.
   - Pumping Lemma for Regular Languages: if $L$ is regular, $\exists p$ such that $\forall s \in L$ with $|s| \ge p$, $s = xyz$ with $|y| > 0, |xy| \le p, \forall i \ge 0, xy^i z \in L$.
2. **Context-Free Languages (Type 2)**:
   - Recognized by Non-deterministic Pushdown Automata (NPDA).
   - Deterministic CFLs (DCFL) recognized by DPDA.
   - Closed under union, concatenation, Kleene star. NOT closed under intersection or complementation.
3. **Context-Sensitive Languages (Type 1)**:
   - Recognized by Linear Bounded Automata (LBA).
4. **Recursively Enumerable Languages (Type 0)**:
   - Recognized by standard Turing Machines. Decidable (Recursive) vs Undecidable (RE but not Recursive).
   - Halting problem of Turing machine is undecidable (RE but not recursive).

## Compiler Design

### Lexical and Syntax Analysis
- **Lexical Analyzer (Scanner)**: Tokenizes input string using Regular Expressions.
- **LL(1) Parsers**: Top-down predictive parsers. A grammar is LL(1) iff for all $A \to \alpha \mid \beta$:
  1. $FIRST(\alpha) \cap FIRST(\beta) = \emptyset$.
  2. If $\epsilon \in FIRST(\beta)$, then $FIRST(\alpha) \cap FOLLOW(A) = \emptyset$.
- **LR Parsers**: Bottom-up shift-reduce parsers.
  - Power: $LR(0) \subset SLR(1) \subset LALR(1) \subset CLR(1)$.

### Code Optimization and Data Flow Analysis
- **Common Subexpression Elimination**: Eliminates recomputations of identical expressions.
- **Dead Code Elimination**: Removes instructions whose computed values are never used.
- **Loop Invariant Code Motion (Hoisting)**: Moves computations whose operand values do not change inside the loop outside before loop entry.
- **Liveness Analysis**: A variable $x$ is live at point $p$ if there is a path from $p$ to a use of $x$ along which $x$ is not redefined. Backward data flow analysis problem.
