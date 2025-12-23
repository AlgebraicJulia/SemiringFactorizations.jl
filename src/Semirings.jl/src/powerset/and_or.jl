struct AndOrLattice <: AbstractLattice end

"""
    AndOr{T} <: Number

The *-autonomous quantale ``(2^n, \\supseteq, \\cup, \\emptyset)``.

  - elements are subsets: ``a \\in 2^n``
  - the ordering is subset exclusion: ``a \\supseteq b``
  - multiplication is union: ``a \\cup b``
  - the multiplicative identity is ``\\emptyset``

This quantale is sometimes called the powerset semiring.
"""
const AndOr = SemiringNumber{AndOrLattice}

#
#   2ⁿ
#
function zero_impl(::Type{AndOrLattice}, ::Type{T}, dual::Val{:N}) where {T}
    return typemax(T)
end

#
#   0
#
function zero_impl(::Type{AndOrLattice}, ::Type{T}, dual::Val{:C}) where {T}
    return zero(T)
end

#
#   a ∩ b
#
function add_impl(::Type{AndOrLattice}, a::T, b::T, dual::Val{:N}) where {T}
    return a & b
end

#
#   a ∪ b
#
function add_impl(::Type{AndOrLattice}, a::T, b::T, dual::Val{:C}) where {T}
    return a | b
end

#
#   b - a
#
function mul_impl(::Type{AndOrLattice}, a::T, b::T, ta::Val{:C}, tb::Val{:N}, dual::Val{:C}) where {T}
    return b & ~a
end
