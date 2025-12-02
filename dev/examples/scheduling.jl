# # Examples: Scheduling with SemiringFactorizations.jl
# This literate example demonstrates how to use **SemiringFactorizations.jl**
# to solve scheduling problems.
#
# We demonstrate both max-plus and min-plus formulations.

using SemiringFactorizations, SparseArrays
const Max64 = MaxPlus{Float64}
const Min64 = MinPlus{Float64}

# ## Max Completion

# This system represents a list of tasks that need to be completed and we want to know how long the longest sequence takes to
# complete. The solution to this system gives us the amount of time needed to complete all tasks. We can encode sequential tasks,
# delayed starts, and lag between tasks.
#
# For this example we have 8 tasks. Task 1 starts at time = 0 since it has no predecessors, Task 2 starts at time = 1 (its release time),
# Task 3 starts at time = 4 (after Task 1 finishes at time = 3 + 1 lag, Task 2 finishes at time = 1 + 2), and so on.
# 
# This scenario is graphed below
#
# ![Task graph](assets/scheduling.svg)
#
# We first set duration values for each task. For example we have set task 3 to take 4 units of time.

d = Max64[3, 2, 4, 3, 5, 2, 6, 4]

# Our initial matrix A will be a Max Plus matrix with float values and -Inf. We set this matrix to be n x n.
# We now fill the initial Max Plus matrix with the constraints, valid ways to move throught the tasks and the lag for specific paths.
# constraints (finish→start) with lags ℓ_ij
# A[j,i] = d[i] + ℓ_ij if i → j, else -Inf

l = Max64[1, 0, 2, 1, 0, 2, 1, 0, 3, 2]
I = [3, 3, 4, 5, 5, 6, 7, 8, 7, 6]
J = [1, 2, 2, 3, 4, 5, 6, 7, 3, 1]
A = sparse(I, J, d[J] + l, 8, 8)

# Release Times: these are the time it takes to start a given task meaning that for example task 1 can start at time = 0
# or no startup cost, while task 2 can start at time = 1 or 1 unit of startup cost. If a release value is set to -Inf then
# we are not able to start from that task, so in this case we can only start from tasks 1 or 2.
b = Max64[0, 1, -Inf, -Inf, -Inf, -Inf, -Inf, -Inf]

# Kleene star
S = slu(A)

# Fixed point solution for x satisfies x = A ⊗ x ⊕ b
x = S * b

# Finish times y = x + d
y = x + d

# Max completion
# This represents the maximum cost it takes to complete the last task, which is calculated just by finding the longest path from
# any avaliable starting task to the final task
# In this case the true solution is the path 1 -> 3 -> 5 -> 6 -> 7 -> 8, 
# total cost 3(1) + 1(1->3) + 4(3) + 1(3->5) + 5(5) + 2(5->6) + 2(6) + 1(6->7) + 6(7) + 0(7->8) + 4(8) = 29
# It is worth noting that this vector gives us the maximum cost to complete any task given the starting tasks so in cases where 
# the tasks are not semi sequential we would still get the highest cost for completion. 

# Max-plus matrix A
A

# Kleene star A*
Matrix(S)

# Earliest start times x
x

# Finish times y = x + d
y

# Completion time
Cmax = sum(y)

# ## Now Min cost 
#
# We now consider a min cost problem where we want to have the lowest cost to reach and complete the final task.
# We can constrain our starting location and ways to traverse through the task list to reach the solution.
# This is a version of the shortest path problem.
#
# We first set duration values for each task. For example we have set task 3 to take 4 units of time.

d = Min64[3, 2, 4, 3, 5, 2, 6, 4]

# Our initial matrix A will be a Min Plus matrix with float values and Inf. We set this matrix to be n x n.
# We now fill the initial Min Plus matrix with the constraints, valid ways to move throught the tasks and the lag for specific paths.
# constraints (finish→start) with lags ℓ_ij
# A[j,i] = d[i] + ℓ_ij if i → j, else -Inf

l = Min64[1, 0, 2, 1, 0, 2, 1, 0, 3, 2]
I = [3, 3, 4, 5, 5, 6, 7, 8, 7, 6]
J = [1, 2, 2, 3, 4, 5, 6, 7, 3, 1]
A = sparse(I, J, d[J] + l, 8, 8)

# Release Times: these are the time it takes to start a given task meaning that for example task 1 can start at time = 0
# or no startup cost, while task 2 can start at time = 1 or 1 unit of startup cost. If a release value is set to Inf then
# we are not able to start from that task, so in this case we can only start from tasks 1 or 2.
b = Min64[0, 1, Inf, Inf, Inf, Inf, Inf, Inf]

# Kleene star
S = slu(A)

# Fixed point solution for x satisfies x = A ⊗ x ⊕ b
x = S * b

# Finish times y = x + d
y = x + d

# Min completion
# This represents the minimum cost it takes to complete task 8 starting at either task 1 or 2.
# In this case the true solution is the path 1 -> 6 -> 7 -> 8, total cost 3(1) + 2(1->6) + 2(6) + 1(6->7) + 6(7) + 0(7->8) + 4(8) = 18
# It is worth noting that this vector gives us the minimum cost to complete any task given the starting tasks so in cases where 
# the tasks are not semi sequential we would still get the least cost for completion. 

# Min-plus matrix A
A

# Kleene star A*
Matrix(S)

# Minimum cumulative costs x"
x

# Finish costs y = x + d
y

# Minimum Cost
Cmin = sum(y)
