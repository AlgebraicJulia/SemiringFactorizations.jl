struct MinPlusPosSemiring <: AbstractIntegralSemiring end

"""
    MinPlusPos{T} <: Number

The quantale ``([0, +\\infty], \\geq, +, 0)``.

  - elements are nonneegative extended real numbers: a ∈ ``[0, +\\infty]``
  - the ordering backwards: ``a \\geq b``
  - multiplication is addition: ``a + b``
  - the multiplicative identity is ``0``

This quantale is sometimes called the Lawvere quantale.
"""
const MinPlusPos{T} = SemiringNumber{MinPlusPosSemiring, T}

#
#   +∞
#
function zero_impl(::Type{MinPlusPosSemiring}, ::Type{T}, dual::Val{:N}) where {T}
    return typemax(T)
end

#
#   0
#
function zero_impl(::Type{MinPlusPosSemiring}, ::Type{T}, dual::Val{:C}) where {T}
    return zero(T)
end

#
#   a ∧ b
#
function add_impl(::Type{MinPlusPosSemiring}, a::T, b::T, dual::Val{:N}) where {T}
    return min(a, b)
end

#
#   a ∨ b
#
function add_impl(::Type{MinPlusPosSemiring}, a::T, b::T, dual::Val{:C}) where {T}
    return max(a, b)
end

#
#   a + b
#
function mul_impl(::Type{MinPlusPosSemiring}, a::T, b::T, ta::Val{:N}, tb::Val{:N}, dual::Val{:N}) where {T}
    return a + b
end

#
#   [b - a]₊
#
function mul_impl(::Type{MinPlusPosSemiring}, a::T, b::T, ta::Val{:C}, tb::Val{:N}, dual::Val{:C}) where {T}
    return max(b - a, zero(T))
end

function mul_impl(::Type{MinPlusPosSemiring}, a::T, b::T, ta::Val{:C}, tb::Val{:N}, dual::Val{:C}) where {T <: Rational}
    return isinf(a) ? zero(T) : max(b - a, zero(T))
end

#
#   a ≥ b
#
function le_impl(::Type{MinPlusPosSemiring}, a::T, b::T) where {T}
    return a >= b
end

#
#   a > b
#
function lt_impl(::Type{MinPlusPosSemiring}, a::T, b::T) where {T}
    return a > b
end
