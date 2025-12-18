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
