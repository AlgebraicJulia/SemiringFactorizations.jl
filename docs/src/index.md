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

is solved by the matrix $A^*X \in \mathbb{S}^{n \times m}$, where $A^*$ is a matrix called the *quasi-inverse*
of $A$. With SemiringFactorizations.jl, we can solve linear fixed-point equations with the functions `sinv(A)`,
`sldiv(A, B)`, and `srdiv(B, A)`, which respectively compute

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

julia> A = [
           2.0 1.0 1.0
           1.0 2.0 0.0
           1.0 0.0 2.0
       ];

julia> b = [
           1.0
           2.0
           3.0
       ];

julia> sldiv(I - A, b)
3-element Vector{Float64}:
 -1.4999999999999998
  1.75
  2.2499999999999996
```

### All-Pairs Shortest Paths

Let $G$ be a directed weighted graph with
adjacency matrix $A$. The all-pairs shortest path
problem can be reformulated as a linear fixed-point point
problem over the min-plus semiring.

```math
AX + I = X.
```

This problem can be solved using the function `sinv`.

```julia-repl
julia> using SemiringFactorizations, TropicalNumbers

julia> A = TropicalMinPlusF64[
           Inf 9.0 8.0 Inf
           Inf Inf 6.0 Inf
           Inf Inf Inf 7.0
           5.0 Inf Inf Inf
       ];

julia> sinv(A)
4×4 Matrix{TropicalMinPlusF64}:
  0.0ₛ   9.0ₛ   8.0ₛ  15.0ₛ
 18.0ₛ   0.0ₛ   6.0ₛ  13.0ₛ
 12.0ₛ  21.0ₛ   0.0ₛ   7.0ₛ
  5.0ₛ  14.0ₛ  13.0ₛ   0.0ₛ
```
