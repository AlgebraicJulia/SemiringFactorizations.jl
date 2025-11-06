using Graphs
using SemiringFactorizations
using TropicalNumbers

### Max Completion

# duration
d = [3.0, 2.0, 4.0, 3.0, 5.0, 2.0, 6.0, 4.0]
n = length(d)

# initial A
A = fill(TropicalMaxPlusF64(-Inf), n, n)

# constraints (finish→start) with lags ℓ_ij
# A[j,i] = d[i] + ℓ_ij if i → j, else -Inf
constraints = [
    (1, 3, 1.0),
    (2, 3, 0.0),
    (2, 4, 2.0),
    (3, 5, 1.0),
    (4, 5, 0.0),
    (5, 6, 2.0),
    (6, 7, 1.0),
    (7, 8, 0.0),
    (3, 7, 3.0),
    (1, 6, 2.0)
]

for (i, j, ℓ) in constraints
    A[j, i] = TropicalMaxPlusF64(d[i] + ℓ)
end

# release times
b = TropicalMaxPlusF64[0.0, 1.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0]

# Quasi-inverse
Astar = sinv(A)

# fixed point solution for x satisfies x = A ⊗ x ⊕ b
x = sldiv(A, b)

# finish times y = x + d
y = TropicalMaxPlusF64[x[i] + TropicalMaxPlusF64(d[i]) for i in 1:n]

# max completion
Cmax = maximum([yi.n for yi in y])

println("Max-plus matrix A")
display(A)

println("\nQuasi-inverse A*")
display(Astar)

println("\nEarliest start times x")
display(x)

println("\nFinish times y = x + d")
display(y)

println("\n Completion time Cmax = ", Cmax)

### Now Min cost

d = [3.0, 2.0, 4.0, 3.0, 5.0, 2.0, 6.0, 4.0]
n = length(d)

A = fill(TropicalMinPlusF64(Inf), n, n)

# constraints: (i, j, cost)
constraints = [
    (1, 3, 4.0),
    (2, 3, 2.0),
    (2, 4, 5.0),
    (3, 5, 3.0),
    (4, 5, 4.0),
    (5, 6, 2.0),
    (6, 7, 1.0),
    (7, 8, 6.0),
    (3, 7, 7.0),
    (1, 6, 5.0)
]

for (i, j, c) in constraints
    A[j, i] = TropicalMinPlusF64(c)
end

# start costs
b = TropicalMinPlusF64[0.0, 1.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0]

Astar = sinv(A)
x = sldiv(A, b)

# finish costs = x + d
y = TropicalMinPlusF64[x[i] + TropicalMinPlusF64(d[i]) for i in 1:n]

# min cost
Cmin = maximum([yi.n for yi in y])

println("Min-plus matrix A")
display(A)

println("\nQuasi-inverse A*")
display(Astar)

println("\nMinimum cumulative costs x")
display(x)

println("\nFinish costs y = x + d")
display(y)

println("\nMinimum total cost Cmin = ", Cmin)
