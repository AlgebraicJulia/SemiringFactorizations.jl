include("max_min.jl")
include("min_max.jl")

const BottleneckLattice = Union{MaxMinLattice, MinMaxLattice}

#
#   -a
#
function MaxMin(a::MinMax)
    return MaxMin(-parent(a))
end

#
#   -a
#
function MinMax(a::MaxMin)
    return MinMax(-parent(a))    
end

#
#   { 1 if a ≤ b
#   { b if a > b
#
function mul_impl(::Type{A}, a::T, b::T, ta::Val{:C}, tb::Val{:N}, dual::Val{:C}) where {A <: BottleneckLattice, T}
    return ifelse(le_impl(A, a, b), one_impl(A, T, Val(:N)), b)
end
