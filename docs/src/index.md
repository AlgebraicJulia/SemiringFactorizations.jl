# SemiringFactorizations.jl

```@meta
CurrentModule = SemiringFactorizations
```

## Semirings

A [semiring](https://en.wikipedia.org/wiki/Semiring) is an algebraic structure that
generalizes rings, dropping the requirement that each element must have an additive
inverse. Examples include

- ``(\mathbb{R}, +, \times, 0, 1)``
- ``(\mathbb{Z}, +, \times, 0, 1)``
- ``(\mathbb{R}, \mathrm{max}, +, -\infty, 0)``
- ``(\mathbb{R}, \mathrm{max}, \mathrm{min}, -\infty, +\infty)``

Several semirings are implemented in the Julia library [TropicalNumbers.jl](https://github.com/TensorBFS/TropicalNumbers.jl/).

## Fixed-Point Equations

Let $A \in \mathbb{S}^{n \times n}$ and $B \in \mathbb{S}^{n \times m}$ be matrices over a semiring $\mathbb{S}$.
The linear fixed-point equation

```math
AX + B = X
```

is solved by the matrix $X = A^*B$, where $A^* \in \mathbb{S}^{n \times n}$ is a
matrix called the *quasi-inverse* of $A$. With SemiringFactorizations.jl, we can solve linear fixed-point
equations with the functions `sinv(A)`, `sldiv(A, B)`, and `srdiv(B, A)`, which respectively compute

- ``A^*``
- ``A^*B``
- ``BA^*``

All three functions work by computing an LU factorization of ``A``. A factorization can also be computed
directly with the function `slu` and then reused. **Both dense and sparse matrices are supported**.

## Examples

### Linear System of Equations

Any linear system of equations

```math
AX = B
```

can be reformulated as a linear fixed-point problem

```math
(I - A)X + B = X.
```

This problem can be solved using the function `sldiv`.

```julia-repl
julia> using LinearAlgebra, SemiringFactorizations

julia> A = Float64[
           1 1 1 1 1 1
           1 2 1 1 1 1
           1 1 2 1 1 1
           1 1 1 2 1 1
           1 1 1 1 2 1
           1 1 1 1 1 2
       ];

julia> b = Float64[
           1
           2
           3
           4
           5
           6 
       ];

julia> sldiv(I - A, b)
6-element Vector{Float64}:
 -14.0
   1.0
   2.0
   3.0
   4.0
   5.0
```

### Path Counting

Directed graphs can be represented by 1-0 matrices
called adjacency matrices.

![](assets/counting.svg)

The quasi-inverse of this matrix contains, for every
pair of vertices, the number paths from one to the
other.

```julia-repl
julia> using SemiringFactorizations

julia> A = Int[
           0 1 1 0 0 0
           0 0 0 0 0 1
           0 0 0 1 1 0
           0 0 0 0 1 0
           0 0 0 0 0 1
           0 0 0 0 0 0
       ];

julia> sinv(A)
6×6 Matrix{Int64}:
 1  1  1  1  2  3
 0  1  0  0  0  1
 0  0  1  1  2  2
 0  0  0  1  1  1
 0  0  0  0  1  1
 0  0  0  0  0  1
```

### Shortest Paths

A weighted graph is a directed graph with a weight
assigned to each edge. Weighted graphs can be
represented by adjacency matrices with entries
in the *min-plus* semiring.

![](assets/shortest.svg)

The quasi-inverse of this matrix contains, for every
pair of vertices, the weight of the shortest path from
one to the other.

```julia-repl
julia> using SemiringFactorizations, TropicalNumbers

julia> A = TropicalMinPlusF64[
           Inf 9.0 1.0 Inf Inf Inf
           Inf Inf Inf Inf Inf 4.0
           Inf Inf Inf 2.0 6.0 Inf
           Inf Inf Inf Inf 3.0 Inf
           Inf Inf Inf Inf Inf 5.0
           Inf Inf Inf Inf Inf Inf
       ];

julia> sinv(A)
6×6 Matrix{TropicalMinPlusF64}:
 0.0ₛ  9.0ₛ  1.0ₛ  3.0ₛ  6.0ₛ  11.0ₛ
 Infₛ  0.0ₛ  Infₛ  Infₛ  Infₛ   4.0ₛ
 Infₛ  Infₛ  0.0ₛ  2.0ₛ  5.0ₛ  10.0ₛ
 Infₛ  Infₛ  Infₛ  0.0ₛ  3.0ₛ   8.0ₛ
 Infₛ  Infₛ  Infₛ  Infₛ  0.0ₛ   5.0ₛ
 Infₛ  Infₛ  Infₛ  Infₛ  Infₛ   0.0ₛ
```

### Finite Automata

A finite automaton is a graph with a string assigned
to each edge. Finite automata can be represented
by adjacency matrices with entries in the semiring
of regular expressions.

![](assets/regular.svg)

The quasi-inverse of this matrix contains, for every
pair of states, the regular expression that brings
the automaton from one to the other.

```julia-repl
julia> using SemiringFactorizations

julia> A = RE[
           "a^" "a"  "a"  "a^" "a^" "a^"
           "a^" "a^" "a^" "a^" "a^" "c"
           "a^" "a^" "a^" "c"  "b"  "a^"
           "a^" "a^" "a^" "a^" "b"  "a^"
           "a^" "a^" "a^" "a^" "a^" "a"
           "a^" "a^" "a^" "a^" "a^" "a^"
       ];

julia> sinv(A)
6×6 Matrix{RE}:
     a   a   ac  ab|acb  ac|(?:ab|acb)a
 a^      a^  a^  a^      c
 a^  a^      c   b|cb    (?:b|cb)a
 a^  a^  a^      b       ba
 a^  a^  a^  a^          a
 a^  a^  a^  a^  a^      
```

Note that regular expression `a^` matches nothing.
