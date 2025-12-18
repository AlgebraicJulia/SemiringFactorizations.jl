struct MinPlusSemiring <: AbstractCommutativeSemiring end

"""
    MinPlus{T} <: Number

The semiring (ℝ ∪ {-∞, +∞}, ∧, +, +∞, 0).

  - elements are extended real numbers: a ∈ ℝ ∪ {-∞, +∞}
  - addition is minimum: a ∧ b
  - multiplication is addition: a + b

It is sometimes called the tropical semiring.
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
#   -a
#
function id_impl(::Type{MinPlusSemiring}, a, dual::Val{:C})
    return -a
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

#=
#
#        -∞   b   +∞
#      + ----------- +
#   -∞ | -∞  -∞   -∞ |
#    a | -∞ a + b +∞ |
#   +∞ | -∞  +∞   +∞ |
#      + ----------- +
# 
function mul_impl(::Type{MinPlusSemiring}, a::T, b::T, ta::Val{:N}, tb::Val{:N}, dual::Val{:C}) where {T}
    return a + b
end

function mul_impl(::Type{MinPlusSemiring}, a::T, b::T, ta::Val{:N}, tb::Val{:N}, dual::Val{:C}) where {T <: Rational}
    ⊥ = typemin(T)
    return (a == ⊥) || (b == ⊥) ? ⊥ : a + b
end
=#

#=
#
#        -∞   b   +∞
#      + ----------- +
#   -∞ | +∞  +∞   +∞ |
#    a | -∞ b - a +∞ |
#   +∞ | -∞  -∞   +∞ |
#      + ----------- +
# 
function mul_impl(::Type{MinPlusSemiring}, a::T, b::T, ta::Val{:C}, tb::Val{:N}, dual::Val{:N}) where {T}
    return b - a
end

function mul_impl(::Type{MinPlusSemiring}, a::T, b::T, ta::Val{:C}, tb::Val{:N}, dual::Val{:N}) where {T <: Rational}
    ⊤ = typemax(T)
    ⊥ = typemin(T)
    return (a == ⊥) || (b == ⊤) ? ⊤ : b - a
end
=#

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
