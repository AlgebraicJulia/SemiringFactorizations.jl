struct AndOrLattice <: AbstractLattice end

"""
    AndOr{T} <: Number

The semiring (2ⁿ, ∩, ∪, 2ⁿ, ∅).

  - elements are subsets: a ∈ 2ⁿ
  - addition is intersection: a ∩ b
  - multiplication is union: a ∪ b

"""
const AndOr = SemiringNumber{AndOrLattice}

#
#   2ⁿ
#
function zero_impl(::Type{AndOrLattice}, ::Type{T}) where {T}
    return typemax(T)
end

#
#   0
#
function one_impl(::Type{AndOrLattice}, ::Type{T}) where {T}
    return zero(T)
end

#
#   a ∩ b
#
function add_impl(::Type{AndOrLattice}, a::T, b::T) where {T}
    return a & b
end

#
#   a ∪ b
#
function mul_impl(::Type{AndOrLattice}, a::T, b::T) where {T}
    return a | b
end

#
#   b - a
#
function ldiv_impl(::Type{AndOrLattice}, a::T, b::T) where {T}
    return b & ~a
end
