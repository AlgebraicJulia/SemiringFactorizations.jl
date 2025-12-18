struct MinMulSemiring <: AbstractCommutativeSemiring end

"""
    MinMul{T} <: Number

The semiring (ℝ⁺ ∪ {+∞}, ∧, ×, +∞, 1).

  - elements are nonnegative extended real numbers: a ∈ ℝ⁺ ∪ {+∞}
  - addition is minimum: a ∧ b
  - multiplication is standard: a × b

"""
const MinMul = SemiringNumber{MinMulSemiring}

#
#   +∞
#
function zero_impl(::Type{MinMulSemiring}, ::Type{T}, dual::Val{:N}) where {T}
    return typemax(T)
end

#
#   0
#
function zero_impl(::Type{MinMulSemiring}, ::Type{T}, dual::Val{:C}) where {T}
    return zero(T)
end

#
#   1
#
function one_impl(::Type{MinMulSemiring}, ::Type{T}, dual::Val) where {T}
    return one(T)
end

#
#   a ∧ b
#
function add_impl(::Type{MinMulSemiring}, a::T, b::T, dual::Val{:N}) where {T}
    return min(a, b)
end

#
#   a ∨ b
#
function add_impl(::Type{MinMulSemiring}, a::T, b::T, dual::Val{:C}) where {T}
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
function mul_impl(::Type{MinMulSemiring}, a::T, b::T, ta::Val{:N}, tb::Val{:N}, dual::Val{:N}) where {T}
    return a * b
end

function mul_impl(::Type{MinMulSemiring}, a::T, b::T, ta::Val{:N}, tb::Val{:N}, dual::Val{:N}) where {T <: Rational}
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
function mul_impl(::Type{MinMulSemiring}, a::T, b::T, ta::Val{:C}, tb::Val{:N}, dual::Val{:C}) where {T}
    return b / a
end

function mul_impl(::Type{MinMulSemiring}, a::T, b::T, ta::Val{:C}, tb::Val{:N}, dual::Val{:C}) where {T <: Rational}
    ⊤ = typemax(T)
    ⊥ = zero(T)
    return (a == ⊥) || (b == ⊤) ? ⊥ : b / a
end

#
#   a ≥ b
#
function le_impl(::Type{MinMulSemiring}, a::T, b::T) where {T}
    return a >= b
end

#
#   a > b
#
function lt_impl(::Type{MinMulSemiring}, a::T, b::T) where {T}
    return a > b
end
