# # Examples: Scheduling with SemiringFactorizations.jl
# This literate example demonstrates how to use **SemiringFactorizations.jl**
# together with **TropicalNumbers.jl** to solve scheduling problems.
#
# We demonstrate both max-plus and min-plus formulations.

using SemiringFactorizations
using TropicalNumbers

# ## Max Completion

# This system represents a list of tasks that need to be completed and we want to know how long the longest sequence takes to
# complete. The solution to this system gives us the amount of time needed to complete all tasks. We can encode sequential tasks,
# delayed starts, and lag between tasks.
#
# For this example we have 8 tasks. Task 1 starts at time = 0 since it has no predecessors, Task 2 starts at time = 1 (its release time),
# Task 3 starts at time = 4 (after Task 1 finishes at time = 3 + 1 lag, Task 2 finishes at time = 1 + 2), and so on.
#
# We first set duration values for each task. For example we have set task 3 to take 4 units of time.

d = [3.0, 2.0, 4.0, 3.0, 5.0, 2.0, 6.0, 4.0]
n = length(d)

# Our initial matrix A will be a Max Plus matrix with float values and -Inf. We set this matrix to be n x n.
A = fill(TropicalMaxPlusF64(-Inf), n, n)

# We now fill the initial Max Plus matrix with the constraints, valid ways to move throught the tasks and the lag for specific paths.
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

# Release Times: these are the time it takes to start a given task meaning that for example task 1 can start at time = 0
# or no startup cost, while task 2 can start at time = 1 or 1 unit of startup cost. If a release value is set to -Inf then
# we are not able to start from that task, so in this case we can only start from tasks 1 or 2.
b = TropicalMaxPlusF64[0.0, 1.0, -Inf, -Inf, -Inf, -Inf, -Inf, -Inf]

# Quasi-inverse
Astar = star(A)

# Fixed point solution for x satisfies x = A ⊗ x ⊕ b
x = slmul(A, b)

# Finish times y = x + d
y = TropicalMaxPlusF64[x[i].n + d[i] for i in 1:n]

# Max completion
# This represents the maximum cost it takes to complete the last task, which is calculated just by finding the longest path from
# any avaliable starting task to the final task
# In this case the true solution is the path 1 -> 3 -> 5 -> 6 -> 7 -> 8, 
# total cost 3(1) + 1(1->3) + 4(3) + 1(3->5) + 5(5) + 2(5->6) + 2(6) + 1(6->7) + 6(7) + 0(7->8) + 4(8) = 29
# It is worth noting that this vector gives us the maximum cost to complete any task given the starting tasks so in cases where 
# the tasks are not semi sequential we would still get the highest cost for completion. 
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

# ## Now Min cost 
#
# We now consider a min cost problem where we want to have the lowest cost to reach and complete the final task.
# We can constrain our starting location and ways to traverse through the task list to reach the solution.
# This is a version of the shortest path problem.
#
# We first set duration values for each task. For example we have set task 3 to take 4 units of time.

d = [3.0, 2.0, 4.0, 3.0, 5.0, 2.0, 6.0, 4.0]
n = length(d)

# Our initial matrix A will be a Min Plus matrix with float values and Inf. We set this matrix to be n x n.
A = fill(TropicalMinPlusF64(Inf), n, n)

# We now fill the initial Min Plus matrix with the constraints, valid ways to move throught the tasks and the lag for specific paths.
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
    A[j, i] = TropicalMinPlusF64(d[i] + ℓ)
end

# Release Times: these are the time it takes to start a given task meaning that for example task 1 can start at time = 0
# or no startup cost, while task 2 can start at time = 1 or 1 unit of startup cost. If a release value is set to Inf then
# we are not able to start from that task, so in this case we can only start from tasks 1 or 2.
b = TropicalMinPlusF64[0.0, 1.0, Inf, Inf, Inf, Inf, Inf, Inf]

# Quasi-inverse
Astar = star(A)

# Fixed point solution for x satisfies x = A ⊗ x ⊕ b
x = slmul(A, b)

# Finish times y = x + d
y = TropicalMinPlusF64[x[i].n + d[i] for i in 1:n]

# Min completion
# This represents the minimum cost it takes to complete task 8 starting at either task 1 or 2.
# In this case the true solution is the path 1 -> 6 -> 7 -> 8, total cost 3(1) + 2(1->6) + 2(6) + 1(6->7) + 6(7) + 0(7->8) + 4(8) = 18
# It is worth noting that this vector gives us the minimum cost to complete any task given the starting tasks so in cases where 
# the tasks are not semi sequential we would still get the least cost for completion. 
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