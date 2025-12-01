struct MinPlusPosQuantale{P} <: AbstractCommutativeQuantale end

"""
    MinPlusPos{P, T} <: Number

The semiring (ℝ⁺ ∪ {+∞}, ∧, +ᴾ, +∞, 0).

  - elements are extended nonnegative real numbers: a ∈ ℝ⁺ ∪ {+∞}
  - addition is minimum: a ∧ b
  - multiplication is P-addition: ᴾ√ (aᴾ + bᴾ)

When P = 1, it is called the Lawvere quantale.
"""
const MinPlusPos{P, T} = SemiringNumber{MinPlusPosQuantale{P}, T}

#
#   +∞
#
function zero_impl(::Type{<:MinPlusPosQuantale}, ::Type{T}) where {T}
    return typemax(T)
end

#
#   0
#
function one_impl(::Type{<:MinPlusPosQuantale}, ::Type{T}) where {T}
    return zero(T)
end

#
#   a ∧ b
#
function add_impl(::Type{<:MinPlusPosQuantale}, a::T, b::T) where {T}
    return min(a, b)
end

#
#   a ∨ b
#
function inf_impl(::Type{<:MinPlusPosQuantale}, a::T, b::T) where {T}
    return max(a, b)
end

#
#   √ (a + b)
#
function mul_impl(::Type{MinPlusPosQuantale{1}}, a::T, b::T) where {T}
    return a + b
end

#
#   ²√ (a² + b²)
#
function mul_impl(::Type{MinPlusPosQuantale{2}}, a::T, b::T) where {T}
    return sqrt(a^2 + b^2)
end

#
#   ³√ (a³ + b³)
#
function mul_impl(::Type{MinPlusPosQuantale{3}}, a::T, b::T) where {T}
    return cbrt(a^3 + b^3)
end

#
#   ⁴√ (a⁴ + b⁴)
#
function mul_impl(::Type{MinPlusPosQuantale{4}}, a::T, b::T) where {T}
    return fourthroot(a^4 + b^4)
end

#
#   ᴾ√ (aᴾ + bᴾ)
#
function mul_impl(::Type{MinPlusPosQuantale{P}}, a::T, b::T) where {P, T}
    return (a^P + b^P)^inv(P)
end

#
#   √ [b - a]₊
#
function ldiv_impl(::Type{MinPlusPosQuantale{1}}, a::T, b::T) where {T}
    return max(b - a, zero(T))
end

function ldiv_impl(::Type{MinPlusPosQuantale{1}}, a::T, b::T) where {T <: Rational}
    return isinf(a) ? zero(T) : max(b - a, zero(T))
end

#
#   ²√ [b² - a²]₊
#
function ldiv_impl(::Type{MinPlusPosQuantale{2}}, a::T, b::T) where {T}
    return sqrt(max(b^2 - a^2, zero(T)))
end

#
#   ³√ [b³ - a³]₊
#
function ldiv_impl(::Type{MinPlusPosQuantale{3}}, a::T, b::T) where {T}
    return cbrt(max(b^3 - a^3, zero(T)))
end

#
#   ⁴√ [b⁴ - a⁴]₊
#
function ldiv_impl(::Type{MinPlusPosQuantale{4}}, a::T, b::T) where {T}
    return fourthroot(max(b^4 - a^4, zero(T)))
end

#
#   ᴾ√ [bᴾ - aᴾ]₊
#
function ldiv_impl(::Type{MinPlusPosQuantale{P}}, a::T, b::T) where {P, T}
    return max(b^P - a^P, zero(T))^inv(P)
end

#
#   a ≥ b
#
function le_impl(::Type{<:MinPlusPosQuantale}, a::T, b::T) where {T}
    return a >= b
end

#
#   a > b
#
function lt_impl(::Type{<:MinPlusPosQuantale}, a::T, b::T) where {T}
    return a > b
end
