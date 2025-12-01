struct MinMulQuantale <: AbstractCommutativeQuantale end

"""
    MinMul{T} <: Number

The semiring (ℝ⁺ ∪ {+∞}, ∧, ×, +∞, 1).

  - elements are nonnegative extended real numbers: a ∈ ℝ⁺ ∪ {+∞}
  - addition is minimum: a ∧ b
  - multiplication is standard: a × b

"""
const MinMul = SemiringNumber{MinMulQuantale}

#
#   +∞
#
function zero_impl(::Type{MinMulQuantale}, ::Type{T}) where {T}
    return typemax(T)
end

#
#   1
#
function one_impl(::Type{MinMulQuantale}, ::Type{T}) where {T}
    return one(T)
end

#
#   0
#
function typemax_impl(::Type{MinMulQuantale}, ::Type{T}) where {T}
    return zero(T)
end

#
#   a ∧ b
#
function add_impl(::Type{MinMulQuantale}, a::T, b::T) where {T}
    return min(a, b)
end

#
#   a ∨ b
#
function inf_impl(::Type{MinMulQuantale}, a::T, b::T) where {T}
    return max(a, b)
end

#
#         0   b   +∞
#      + ----------- +
#    0 |  0   0   +∞ |
#    a |  0 a × b +∞ |
#   +∞ | +∞  +∞   +∞ |
#      + ----------- +
# 
function mul_impl(::Type{MinMulQuantale}, a::T, b::T) where {T}
    return a * b
end

function mul_impl(::Type{MinMulQuantale}, a::T, b::T) where {T <: Rational}
    ⊤ = typemax(T)
    return (a == ⊤) || (b == ⊤) ? ⊤ : a * b
end

#
#         0   b   +∞
#      + ----------- +
#    0 |  0  +∞   +∞ |
#    a |  0 b / a +∞ |
#   +∞ |  0   0    0 |
#      + ----------- +
# 
function ldiv_impl(::Type{MinMulQuantale}, a::T, b::T) where {T}
    return b / a
end

function ldiv_impl(::Type{MinMulQuantale}, a::T, b::T) where {T <: Rational}
    ⊤ = typemax(T)
    ⊥ = zero(T)
    return (a == ⊥) || (b == ⊤) ? ⊥ : b / a
end

#
#   a ≥ b
#
function le_impl(::Type{MinMulQuantale}, a::T, b::T) where {T}
    return a >= b
end

#
#   a > b
#
function lt_impl(::Type{MinMulQuantale}, a::T, b::T) where {T}
    return a > b
end
