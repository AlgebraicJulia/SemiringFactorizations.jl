# SemiringFactorizations.jl

SemiringFactorizations.jl is a high-performance Julia library for solving
affine fixed-point equations

```math
AX + B = X
```

where A and B are matrices valued in a [semiring](https://en.wikipedia.org/wiki/Semiring).
The library supports all the tropical number types in
[TropicalNumbers.jl](https://github.com/TensorBFS/TropicalNumbers.jl) as well as the
built-in Julia number types (`Float64`, `Int64`, etc.). The matrix $A$ can be either
dense or sparse.

# Examples

## Linear System of Equations

Any linear system of equations

```math
AX = B
```

can be reforulated as a fixed-point problem

```math
(I - A)X + B = X.
```

This problem can be solved using the function `sinv!`.

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

julia> sldiv!(I - A, b)
3-element Vector{Float64}:
 -1.4999999999999998
  1.75
  2.2499999999999996
```

## All-Pairs Shortest Paths

Let $G$ be a directed weighted graph with
adjacency matrix $A$. The all-pairs shortest path
problem can be formulated as a fixed-point point
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
 Infₛ  9.0ₛ  8.0ₛ  Infₛ
 Infₛ  Infₛ  6.0ₛ  Infₛ
 Infₛ  Infₛ  Infₛ  7.0ₛ
 5.0ₛ  Infₛ  Infₛ  Infₛ
```
