struct MinLSESemiring <: AbstractIntegralSemiring end

"""
    MinLSE{T} <: Number

The semiring (ℝ ∪ {-∞, +∞}, ∧, LSE, +∞, 0).

  - elements are extended real numbers: a ∈ ℝ ∪ {-∞, +∞}
  - addition is minimum: a ∧ b
  - multiplication is LSE: log (eᵃ + eᵇ)

It is sometimes called the logarithmic semiring.
"""
const MinLSE{T} = SemiringNumber{MinLSESemiring, T}

#
#   +∞
#
function zero_impl(::Type{MinLSESemiring}, ::Type{T}, dual::Val{:N}) where {T}
    return typemax(T)
end

#
#   -∞
#
function zero_impl(::Type{MinLSESemiring}, ::Type{T}, dual::Val{:C}) where {T}
    return typemin(T)
end

#
#   a ∧ b
#
function add_impl(::Type{MinLSESemiring}, a::T, b::T, dual::Val{:N}) where {T}
    return min(a, b)
end

#
#   a ∨ b
#
function add_impl(::Type{MinLSESemiring}, a::T, b::T, dual::Val{:C}) where {T}
    return max(a, b)
end

#
#   log (eᵃ + eᵇ)
#
function mul_impl(::Type{MinLSESemiring}, a::T, b::T, ta::Val{:N}, tb::Val{:N}, dual::Val{:N}) where {T}
    return log(exp(a) + exp(b))
end

#
#   log [eᵇ - eᵃ]₊ 
#
function mul_impl(::Type{MinLSESemiring}, a::T, b::T, ta::Val{:C}, tb::Val{:N}, dual::Val{:C}) where {T}
    return log(max(exp(b) - exp(a), zero(T)))
end

#
#   a ≥ b
#
function le_impl(::Type{MinLSESemiring}, a::T, b::T) where {T}
    return a >= b
end

#
#   a > b
#
function lt_impl(::Type{MinLSESemiring}, a::T, b::T) where {T}
    return a > b
end
