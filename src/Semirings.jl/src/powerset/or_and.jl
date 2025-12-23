struct OrAndLattice <: AbstractLattice end

"""
    OrAnd{T} <: AbstractSemiringNumber{T}

The *-autonomous quantale ``(2^n, \\subseteq, \\cap, 2^n)``.

  - elements are subsets: ``a \\in 2^n``
  - the ordering is subset inclusion: ``a \\subseteq b``
  - multiplication is intersection: ``a \\cap b``
  - the multiplicative identity is ``2^n``

This quantale is sometimes called the powerset semiring.
"""
const OrAnd = SemiringNumber{OrAndLattice}

#
#   0
#
function zero_impl(::Type{OrAndLattice}, ::Type{T}, dual::Val{:N}) where {T}
    return zero(T)
end

#
#   2ⁿ
#
function zero_impl(::Type{OrAndLattice}, ::Type{T}, dual::Val{:C}) where {T}
    return typemax(T)
end

#
#   a ∪ b
#
function add_impl(::Type{OrAndLattice}, a::T, b::T, dual::Val{:N}) where {T}
    return a | b
end

#
#   a ∩ b
#
function add_impl(::Type{OrAndLattice}, a::T, b::T, dual::Val{:C}) where {T}
    return a & b
end

#
#   a → b
#
function mul_impl(::Type{OrAndLattice}, a::T, b::T, ta::Val{:C}, tb::Val{:N}, dual::Val{:C}) where {T}
    return b | ~a
end
