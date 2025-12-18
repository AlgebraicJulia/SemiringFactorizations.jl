struct MaxPlusSemiring <: AbstractCommutativeSemiring end

"""
    MaxPlus{T} <: Number

The semiring (ℝ ∪ {-∞, +∞}, ∨, +, -∞, 0).

  - elements are extended real numbers: a ∈ ℝ ∪ {-∞, +∞}
  - addition is maximum: a ∨ b
  - multiplication is addition: a + b

It is sometimes called the arctic semiring.
"""
const MaxPlus = SemiringNumber{MaxPlusSemiring}

#
#   -∞
#
function zero_impl(::Type{MaxPlusSemiring}, ::Type{T}, dual::Val{:N}) where {T}
    return typemin(T)
end

#
#   +∞
#
function zero_impl(::Type{MaxPlusSemiring}, ::Type{T}, dual::Val{:C}) where {T}
    return typemax(T)
end

#
#   0
#
function one_impl(::Type{MaxPlusSemiring}, ::Type{T}, dual::Val) where {T}
    return zero(T)
end

#
#   a ∨ b
#
function add_impl(::Type{MaxPlusSemiring}, a::T, b::T, dual::Val{:N}) where {T}
    return max(a, b)
end

#
#   a ∧ b
#
function add_impl(::Type{MaxPlusSemiring}, a::T, b::T, dual::Val{:C}) where {T}
    return min(a, b)
end

#
#        -∞   b   +∞
#      + ----------- +
#   -∞ | -∞  -∞   -∞ |
#    a | -∞ a + b +∞ |
#   +∞ | -∞  +∞   +∞ |
#      + ----------- +
# 
function mul_impl(::Type{MaxPlusSemiring}, a::T, b::T, ta::Val{:N}, tb::Val{:N}, dual::Val{:N}) where {T}
    return a + b
end

function mul_impl(::Type{MaxPlusSemiring}, a::T, b::T, ta::Val{:N}, tb::Val{:N}, dual::Val{:N}) where {T <: Rational}
    ⊥ = typemin(T)
    return (a == ⊥) || (b == ⊥) ? ⊥ : a + b
end

#
#        -∞   b   +∞
#      + ----------- +
#   -∞ | +∞  +∞   +∞ |
#    a | -∞ b - a +∞ |
#   +∞ | -∞  -∞   +∞ |
#      + ----------- +
# 
function mul_impl(::Type{MaxPlusSemiring}, a::T, b::T, ta::Val{:C}, tb::Val{:N}, dual::Val{:C}) where {T}
    return b - a
end

function mul_impl(::Type{MaxPlusSemiring}, a::T, b::T, ta::Val{:C}, tb::Val{:N}, dual::Val{:C}) where {T <: Rational}
    ⊤ = typemax(T)
    ⊥ = typemin(T)
    return (a == ⊥) || (b == ⊤) ? ⊤ : b - a
end

#
#   a ≤ b
#
function le_impl(::Type{MaxPlusSemiring}, a::T, b::T) where {T}
    return a <= b
end

#
#   a < b
#
function lt_impl(::Type{MaxPlusSemiring}, a::T, b::T) where {T}
    return a < b
end
