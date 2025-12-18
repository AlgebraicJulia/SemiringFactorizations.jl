struct MaxMinLattice <: AbstractLattice end

"""
    MaxMin{T} <: Number

The semiring (ℝ ∪ {-∞, +∞}, ∨, ∧, -∞, +∞).

  - elements are extended real numbers: a ∈ ℝ ∪ {-∞, +∞}
  - addition is maximum: a ∨ b
  - multiplication is minimum: a ∧ b

It is sometimes called the bottleneck semiring.
"""
const MaxMin = SemiringNumber{MaxMinLattice}

#
#   -∞
#
function zero_impl(::Type{MaxMinLattice}, ::Type{T}, dual::Val{:N}) where {T}
    return typemin(T)
end

#
#   +∞
#
function zero_impl(::Type{MaxMinLattice}, ::Type{T}, dual::Val{:C}) where {T}
    return typemax(T)
end

#
#   a ∨ b
#
function add_impl(::Type{MaxMinLattice}, a::T, b::T, dual::Val{:N}) where {T}
    return max(a, b)
end

#
#   a ∧ b
#
function add_impl(::Type{MaxMinLattice}, a::T, b::T, dual::Val{:C}) where {T}
    return min(a, b)
end

#
#   a ≤ b
#
function le_impl(::Type{MaxMinLattice}, a::T, b::T) where {T}
    return a <= b
end

#
#   a < b
#
function lt_impl(::Type{MaxMinLattice}, a::T, b::T) where {T}
    return a < b
end
