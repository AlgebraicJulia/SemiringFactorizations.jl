struct MinLSESemiring{P} <: AbstractIntegralSemiring end

"""
    MinLSE{P, T} <: Number

The semiring (ℝ ∪ {-∞, +∞}, ∧, P-LSE, +∞, 0).

  - elements are extended real numbers: a ∈ ℝ ∪ {-∞, +∞}
  - addition is minimum: a ∧ b
  - multiplication is P-LSE: 1/P log (eᴾᵃ + eᴾᵇ)

It is sometimes called the logarithmic semiring.
"""
const MinLSE{P, T} = SemiringNumber{MinLSESemiring{P}, T}

#
#   +∞
#
function zero_impl(::Type{<:MinLSESemiring}, ::Type{T}, dual::Val{:N}) where {T}
    return typemax(T)
end

#
#   -∞
#
function zero_impl(::Type{<:MinLSESemiring}, ::Type{T}, dual::Val{:C}) where {T}
    return typemin(T)
end

#
#   a ∧ b
#
function add_impl(::Type{<:MinLSESemiring}, a::T, b::T, dual::Val{:N}) where {T}
    return min(a, b)
end

#
#   a ∨ b
#
function add_impl(::Type{<:MinLSESemiring}, a::T, b::T, dual::Val{:C}) where {T}
    return max(a, b)
end

#
#   1/P log (eᴾᵃ + eᴾᵇ)
#
function mul_impl(::Type{MinLSESemiring{P}}, a::T, b::T, ta::Val{:N}, tb::Val{:N}, dual::Val{:N}) where {P, T}
    return log(exp(P * a) + exp(P * b)) / P
end

#
#   1/P log [eᴾᵇ - eᴾᵃ]₊ 
#
function mul_impl(::Type{MinLSESemiring{P}}, a::T, b::T, ta::Val{:C}, tb::Val{:N}, dual::Val{:C}) where {P, T}
    return log(max(exp(P * b) - exp(P * a), zero(T))) / P
end

#
#   a ≥ b
#
function le_impl(::Type{<:MinLSESemiring}, a::T, b::T) where {T}
    return a >= b
end

#
#   a > b
#
function lt_impl(::Type{<:MinLSESemiring}, a::T, b::T) where {T}
    return a > b
end
