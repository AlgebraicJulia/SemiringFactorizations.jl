struct MinPlusPosSemiring{P} <: AbstractIntegralSemiring end

"""
    MinPlusPos{P, T} <: Number

The semiring (ℝ⁺ ∪ {+∞}, ∧, +ᴾ, +∞, 0).

  - elements are extended nonnegative real numbers: a ∈ ℝ⁺ ∪ {+∞}
  - addition is minimum: a ∧ b
  - multiplication is P-addition: ᴾ√ (aᴾ + bᴾ)

When P = 1, it is called the Lawvere quantale.
"""
const MinPlusPos{P, T} = SemiringNumber{MinPlusPosSemiring{P}, T}

#
#   +∞
#
function zero_impl(::Type{<:MinPlusPosSemiring}, ::Type{T}, dual::Val{:N}) where {T}
    return typemax(T)
end

#
#   0
#
function zero_impl(::Type{<:MinPlusPosSemiring}, ::Type{T}, dual::Val{:C}) where {T}
    return zero(T)
end

#
#   a ∧ b
#
function add_impl(::Type{<:MinPlusPosSemiring}, a::T, b::T, dual::Val{:N}) where {T}
    return min(a, b)
end

#
#   a ∨ b
#
function add_impl(::Type{<:MinPlusPosSemiring}, a::T, b::T, dual::Val{:C}) where {T}
    return max(a, b)
end

#
#   √ (a + b)
#
function mul_impl(::Type{MinPlusPosSemiring{1}}, a::T, b::T, ta::Val{:N}, tb::Val{:N}, dual::Val{:N}) where {T}
    return a + b
end

#
#   ²√ (a² + b²)
#
function mul_impl(::Type{MinPlusPosSemiring{2}}, a::T, b::T, ta::Val{:N}, tb::Val{:N}, dual::Val{:N}) where {T}
    return sqrt(a^2 + b^2)
end

#
#   ³√ (a³ + b³)
#
function mul_impl(::Type{MinPlusPosSemiring{3}}, a::T, b::T, ta::Val{:N}, tb::Val{:N}, dual::Val{:N}) where {T}
    return cbrt(a^3 + b^3)
end

#
#   ⁴√ (a⁴ + b⁴)
#
function mul_impl(::Type{MinPlusPosSemiring{4}}, a::T, b::T, ta::Val{:N}, tb::Val{:N}, dual::Val{:N}) where {T}
    return fourthroot(a^4 + b^4)
end

#
#   ᴾ√ (aᴾ + bᴾ)
#
function mul_impl(::Type{MinPlusPosSemiring{P}}, a::T, b::T, ta::Val{:N}, tb::Val{:N}, dual::Val{:N}) where {P, T}
    return (a^P + b^P)^inv(P)
end

#
#   √ [b - a]₊
#
function mul_impl(::Type{MinPlusPosSemiring{1}}, a::T, b::T, ta::Val{:C}, tb::Val{:N}, dual::Val{:C}) where {T}
    return max(b - a, zero(T))
end

function mul_impl(::Type{MinPlusPosSemiring{1}}, a::T, b::T, ta::Val{:C}, tb::Val{:N}, dual::Val{:C}) where {T <: Rational}
    return isinf(a) ? zero(T) : max(b - a, zero(T))
end

#
#   ²√ [b² - a²]₊
#
function mul_impl(::Type{MinPlusPosSemiring{2}}, a::T, b::T, ta::Val{:C}, tb::Val{:N}, dual::Val{:C}) where {T}
    return sqrt(max(b^2 - a^2, zero(T)))
end

#
#   ³√ [b³ - a³]₊
#
function mul_impl(::Type{MinPlusPosSemiring{3}}, a::T, b::T, ta::Val{:C}, tb::Val{:N}, dual::Val{:C}) where {T}
    return cbrt(max(b^3 - a^3, zero(T)))
end

#
#   ⁴√ [b⁴ - a⁴]₊
#
function mul_impl(::Type{MinPlusPosSemiring{4}}, a::T, b::T, ta::Val{:C}, tb::Val{:N}, dual::Val{:C}) where {T}
    return fourthroot(max(b^4 - a^4, zero(T)))
end

#
#   ᴾ√ [bᴾ - aᴾ]₊
#
function mul_impl(::Type{MinPlusPosSemiring{P}}, a::T, b::T, ta::Val{:C}, tb::Val{:N}, dual::Val{:C}) where {P, T}
    return max(b^P - a^P, zero(T))^inv(P)
end

#
#   a ≥ b
#
function le_impl(::Type{<:MinPlusPosSemiring}, a::T, b::T) where {T}
    return a >= b
end

#
#   a > b
#
function lt_impl(::Type{<:MinPlusPosSemiring}, a::T, b::T) where {T}
    return a > b
end
