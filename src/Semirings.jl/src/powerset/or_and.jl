struct OrAndLattice <: AbstractLattice end

"""
    OrAnd{T} <: Number

The semiring (2ⁿ, ∪, ∩, ∅, 2ⁿ).

  - elements are subsets: a ∈ 2ⁿ
  - addition is union: a ∪ b
  - multiplication is intersection: a ∩ b

"""
const OrAnd = SemiringNumber{OrAndLattice}

#
#   0
#
function zero_impl(::Type{OrAndLattice}, ::Type{T}) where {T}
    return zero(T)
end

#
#   2ⁿ
#
function one_impl(::Type{OrAndLattice}, ::Type{T}) where {T}
    return typemax(T)
end

#
#   a ∪ b
#
function add_impl(::Type{OrAndLattice}, a::T, b::T) where {T}
    return a | b
end

#
#   a ∩ b
#
function mul_impl(::Type{OrAndLattice}, a::T, b::T) where {T}
    return a & b
end

#
#   a → b
#
function ldiv_impl(::Type{OrAndLattice}, a::T, b::T) where {T}
    return b | ~a
end
