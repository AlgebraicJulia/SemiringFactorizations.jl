struct MinPlusSemiring <: AbstractCommutativeSemiring end

"""
    MinPlus{T} <: AbstractSemiringNumber{T}

The *-autonomous quantale ``([-\\infty, +\\infty], \\geq, +, 0)``.

  - elements are extended real numbers: a ∈ ``[-\\infty, +\\infty]``
  - the ordering is backwards: ``a \\geq b``
  - multiplication is addition: ``a + b``
  - the multiplicative identity is ``0``

This quantale is sometimes called the tropical semiring.
"""
const MinPlus = SemiringNumber{MinPlusSemiring}

#
#   +∞
#
function zero_impl(::Type{MinPlusSemiring}, ::Type{T}, dual::Val{:N}) where {T}
    return typemax(T)
end

#
#   -∞
#
function zero_impl(::Type{MinPlusSemiring}, ::Type{T}, dual::Val{:C}) where {T}
    return typemin(T)
end

#
#   0
#
function one_impl(::Type{MinPlusSemiring}, ::Type{T}, dual::Val) where {T}
    return zero(T)
end

#
#   a ∧ b
#
function add_impl(::Type{MinPlusSemiring}, a::T, b::T, dual::Val{:N}) where {T}
    return min(a, b)
end

#
#   a ∨ b
#
function add_impl(::Type{MinPlusSemiring}, a::T, b::T, dual::Val{:C}) where {T}
    return max(a, b)
end

#
#        -∞   b   +∞
#      + ----------- +
#   -∞ | -∞  -∞   +∞ |
#    a | -∞ a + b +∞ |
#   +∞ | +∞  +∞   +∞ |
#      + ----------- +
# 
function mul_impl(::Type{MinPlusSemiring}, a::T, b::T, ta::Val{:N}, tb::Val{:N}, dual::Val{:N}) where {T}
    return a + b
end

function mul_impl(::Type{MinPlusSemiring}, a::T, b::T, ta::Val{:N}, tb::Val{:N}, dual::Val{:N}) where {T <: Rational}
    ⊤ = typemax(T)
    return (a == ⊤) || (b == ⊤) ? ⊤ : a + b
end

#
#        -∞   b   +∞
#      + ----------- +
#   -∞ | -∞  +∞   +∞ |
#    a | -∞ b - a +∞ |
#   +∞ | -∞  -∞   -∞ |
#      + ----------- +
# 
function mul_impl(::Type{MinPlusSemiring}, a::T, b::T, ta::Val{:C}, tb::Val{:N}, dual::Val{:C}) where {T}
    return b - a
end

function mul_impl(::Type{MinPlusSemiring}, a::T, b::T, ta::Val{:C}, tb::Val{:N}, dual::Val{:C}) where {T <: Rational}
    ⊤ = typemax(T)
    ⊥ = typemin(T)
    return (a == ⊤) || (b == ⊥) ? ⊥ : b - a
end

#
#   a ≥ b
#
function le_impl(::Type{MinPlusSemiring}, a::T, b::T) where {T}
    return a >= b
end

#
#   a > b
#
function lt_impl(::Type{MinPlusSemiring}, a::T, b::T) where {T}
    return a > b
end
