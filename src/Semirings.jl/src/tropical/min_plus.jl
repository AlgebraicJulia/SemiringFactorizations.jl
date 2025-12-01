struct MinPlusQuantale <: AbstractCommutativeQuantale end

"""
    MinPlus{T} <: Number

The semiring (ℝ ∪ {-∞, +∞}, ∧, +, +∞, 0).

  - elements are extended real numbers: a ∈ ℝ ∪ {-∞, +∞}
  - addition is minimum: a ∧ b
  - multiplication is addition: a + b

It is sometimes called the tropical semiring.
"""
const MinPlus = SemiringNumber{MinPlusQuantale}

#
#   +∞
#
function zero_impl(::Type{MinPlusQuantale}, ::Type{T}) where {T}
    return typemax(T)
end

#
#   0
#
function one_impl(::Type{MinPlusQuantale}, ::Type{T}) where {T}
    return zero(T)
end

#
#   -∞
#
function typemax_impl(::Type{MinPlusQuantale}, ::Type{T}) where {T}
    return typemin(T)
end

#
#   a ∧ b
#
function add_impl(::Type{MinPlusQuantale}, a::T, b::T) where {T}
    return min(a, b)
end

#
#   a ∨ b
#
function inf_impl(::Type{MinPlusQuantale}, a::T, b::T) where {T}
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
function mul_impl(::Type{MinPlusQuantale}, a::T, b::T) where {T}
    return a + b
end

function mul_impl(::Type{MinPlusQuantale}, a::T, b::T) where {T <: Rational}
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
function ldiv_impl(::Type{MinPlusQuantale}, a::T, b::T) where {T}
    return b - a
end

function ldiv_impl(::Type{MinPlusQuantale}, a::T, b::T) where {T <: Rational}
    ⊤ = typemax(T)
    ⊥ = typemin(T)
    return (a == ⊤) || (b == ⊥) ? ⊥ : b - a
end

#
#   a ≥ b
#
function le_impl(::Type{MinPlusQuantale}, a::T, b::T) where {T}
    return a >= b
end

#
#   a > b
#
function lt_impl(::Type{MinPlusQuantale}, a::T, b::T) where {T}
    return a > b
end
