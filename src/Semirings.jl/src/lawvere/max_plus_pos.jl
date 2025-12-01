struct MaxPlusPosQuantale{P} <: AbstractCommutativeQuantale end

"""
    MaxPlusPos{P, T} <: Number

The semiring (ℝ⁺ ∪ {+∞}, ∨, +⁻ᴾ, 0, +∞).

  - elements are extended nonnegative real numbers: a ∈ ℝ⁺ ∪ {+∞}
  - addition is maximum: a ∨ b
  - multiplication is inverse P-addition: ⁻ᴾ√ (a⁻ᴾ + b⁻ᴾ)

"""
const MaxPlusPos{P, T} = SemiringNumber{MaxPlusPosQuantale{P}, T}

#
#   0
#
function zero_impl(::Type{<:MaxPlusPosQuantale}, ::Type{T}) where {T}
    return zero(T)
end

#
#   +∞
#
function one_impl(::Type{<:MaxPlusPosQuantale}, ::Type{T}) where {T}
    return typemax(T)
end

#
#   a ∨ b
#
function add_impl(::Type{<:MaxPlusPosQuantale}, a::T, b::T) where {T}
    return max(a, b)
end

#
#   a ∧ b
#
function inf_impl(::Type{<:MaxPlusPosQuantale}, a::T, b::T) where {T}
    return min(a, b)
end

#
#   ⁻¹√ (a⁻¹ + b⁻¹)
#
function mul_impl(::Type{MaxPlusPosQuantale{1}}, a::T, b::T) where {T}
    return inv(inv(a) + inv(b))
end

function mul_impl(::Type{MaxPlusPosQuantale{1}}, a::T, b::T) where {T <: Rational}
    return iszero(a) ? a : inv(inv(a) + inv(b))
end

#
#   ⁻²√ (a⁻² + b⁻²)
#
function mul_impl(::Type{MaxPlusPosQuantale{2}}, a::T, b::T) where {T}
    return sqrt(inv(inv(a)^2 + inv(b)^2))
end

#
#   ⁻³√ (a⁻³ + b⁻³)
#
function mul_impl(::Type{MaxPlusPosQuantale{3}}, a::T, b::T) where {T}
    return cbrt(inv(inv(a)^3 + inv(b)^3))
end

#
#   ⁻⁴√ (a⁻⁴ + b⁻⁴)
#
function mul_impl(::Type{MaxPlusPosQuantale{4}}, a::T, b::T) where {T}
    return fourthroot(inv(inv(a)^4 + inv(b)^4))
end

#
#   ⁻ᴾ√ (a⁻ᴾ + b⁻ᴾ)
#
function mul_impl(::Type{MaxPlusPosQuantale{P}}, a::T, b::T) where {P, T}
    return inv(inv(a)^P + inv(b)^P)^inv(P)
end

#
#   ⁻¹√ [b⁻¹ - a⁻¹]₊
#
function ldiv_impl(::Type{MaxPlusPosQuantale{1}}, a::T, b::T) where {T}
    return inv(max(inv(b) - inv(a), zero(T)))
end

function ldiv_impl(::Type{MaxPlusPosQuantale{1}}, a::T, b::T) where {T <: Rational}
    return iszero(a) ? typemax(T) : inv(max(inv(b) - inv(a), zero(T)))
end

#
#   ⁻²√ [b⁻² - a⁻²]₊
#
function ldiv_impl(::Type{MaxPlusPosQuantale{2}}, a::T, b::T) where {T}
    return sqrt(inv(max(inv(b)^2 - inv(a)^2, zero(T))))
end

#
#   ⁻³√ [b⁻³ - a⁻³]₊
#
function ldiv_impl(::Type{MaxPlusPosQuantale{3}}, a::T, b::T) where {T}
    return cbrt(inv(max(inv(b)^3 - inv(a)^3, zero(T))))
end

#
#   ⁻⁴√ [b⁻⁴ - a⁻⁴]₊
#
function ldiv_impl(::Type{MaxPlusPosQuantale{4}}, a::T, b::T) where {T}
    return fourthroot(inv(max(inv(b)^4 - inv(a)^4, zero(T))))
end

#
#   ⁻ᴾ√ [b⁻ᴾ - a⁻ᴾ]₊
#
function ldiv_impl(::Type{MaxPlusPosQuantale{P}}, a::T, b::T) where {P, T}
    return inv(max(inv(b)^P - inv(a)^P, zero(T)))^inv(P)
end

#
#   a ≤ b
#
function le_impl(::Type{<:MaxPlusPosQuantale}, a::T, b::T) where {T}
    return a <= b
end

#
#   a < b
#
function lt_impl(::Type{<:MaxPlusPosQuantale}, a::T, b::T) where {T}
    return a < b
end
