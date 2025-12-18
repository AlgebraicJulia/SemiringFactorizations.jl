struct MinMaxLattice <: AbstractLattice end

"""
    MinMax{T} <: Number

The semiring (ℝ ∪ {-∞, +∞}, ∧, ∨, +∞, -∞).

  - elements are extended real numbers: a ∈ ℝ ∪ {-∞, +∞}
  - addition is minimum: a ∧ b
  - multiplication is maximum: a ∨ b

It is sometimes called the bottleneck semiring.
"""
const MinMax = SemiringNumber{MinMaxLattice}

#
#   +∞
#
function zero_impl(::Type{MinMaxLattice}, ::Type{T}, dual::Val{:N}) where {T}
    return typemax(T)
end

#
#   -∞
#
function zero_impl(::Type{MinMaxLattice}, ::Type{T}, dual::Val{:C}) where {T}
    return typemin(T)
end

#
#   a ∧ b
#
function add_impl(::Type{MinMaxLattice}, a::T, b::T, dual::Val{:N}) where {T}
    return min(a, b)
end

#
#   a ∨ b
#
function add_impl(::Type{MinMaxLattice}, a::T, b::T, dual::Val{:C}) where {T}
    return max(a, b)
end

#
#    a ≥ b
#
function le_impl(::Type{MinMaxLattice}, a::T, b::T) where {T}
    return a >= b
end

#
#   a > b
#
function lt_impl(::Type{MinMaxLattice}, a::T, b::T) where {T}
    return a > b
end
