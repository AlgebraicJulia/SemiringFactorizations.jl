# SemiringFactorizations.jl

```@meta
CurrentModule = SemiringFactorizations
```

## Semirings

A [semiring](https://en.wikipedia.org/wiki/Semiring) is an algebraic structure that
generalizes rings, omitting the requirement that each element have an additive
inverse. Examples include

- ``(\mathbb{R}, +, \times, 0, 1)``
- ``(\mathbb{R}, \mathrm{max}, +, -\infty, 0)``
- ``(\mathbb{R}, \mathrm{max}, \mathrm{min}, -\infty, +\infty)``

## Kleene Star

Let ``A \in \mathbb{S}^{n \times n}`` be a square matrix with elements in a semiring ``\mathbb{S}``.
The [Kleene star](https://en.wikipedia.org/wiki/Kleene_algebra) of ``A`` is the infinite sum
```math
    I + A + A^2 + A^3 + \cdots
```

With SemiringFactorizations.jl, we can compute the Kleene star of a matrix with the function `star`.

## Examples

### Shortest Paths

A weighted graph is a directed graph with a weight
assigned to each edge. Weighted graphs can be
represented by adjacency matrices with entries
in the *min-plus* semiring.

![](assets/shortest.svg)

The Kleene star of this matrix contains the weight of
the shortest path between every pair of vertices.

```julia-repl
julia> using SemiringFactorizations

julia> A = MinPlus{Float64}[
           Inf 9.0 1.0 Inf Inf Inf
           Inf Inf Inf Inf Inf 4.0
           Inf Inf Inf 2.0 6.0 Inf
           Inf Inf Inf Inf 3.0 Inf
           Inf Inf Inf Inf Inf 5.0
           Inf Inf Inf Inf Inf Inf
       ];

julia> star(A)
6×6 Matrix{MinPlus{Float64}}:
 0.0  9.0  1.0  3.0  6.0  11.0
 Inf  0.0  Inf  Inf  Inf   4.0
 Inf  Inf  0.0  2.0  5.0  10.0
 Inf  Inf  Inf  0.0  3.0   8.0
 Inf  Inf  Inf  Inf  0.0   5.0
 Inf  Inf  Inf  Inf  Inf   0.0
```
