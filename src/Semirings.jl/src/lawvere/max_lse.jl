struct MaxLSESemiring{P} <: AbstractIntegralSemiring end

"""
    MaxLSE{P, T} <: Number

The semiring (ℝ ∪ {-∞, +∞}, ∨, -P-LSE, +∞, 0).

  - elements are extended real numbers: a ∈ ℝ ∪ {-∞, +∞}
  - addition is maximum: a ∨ b
  - multiplication is -P-LSE: -1/P log (e⁻ᴾᵃ + e⁻ᴾᵇ)

It is sometimes called the logarithmic semiring.
"""
const MaxLSE{P, T} = SemiringNumber{MaxLSESemiring{P}, T}

#
#   -∞
#
function zero_impl(::Type{<:MaxLSESemiring}, ::Type{T}, dual::Val{:N}) where {T}
    return typemin(T)
end

#
#   +∞
#
function zero_impl(::Type{<:MaxLSESemiring}, ::Type{T}, dual::Val{:C}) where {T}
    return typemax(T)
end

#
#   a ∨ b
#
function add_impl(::Type{<:MaxLSESemiring}, a::T, b::T, dual::Val{:N}) where {T}
    return max(a, b)
end

#
#   a ∧ b
#
function add_impl(::Type{<:MaxLSESemiring}, a::T, b::T, dual::Val{:C}) where {T}
    return min(a, b)
end

#
#   -1/P log (e⁻ᴾᵃ + e⁻ᵇ)
#
function mul_impl(::Type{MaxLSESemiring{P}}, a::T, b::T, ta::Val{:N}, tb::Val{:N}, dual::Val{:N}) where {P, T}
    return -log(exp(P * -a) + exp(P * -b)) / P
end

#
#   -1/P log [e⁻ᴾᵇ - e⁻ᴾᵃ]₊ 
#
function mul_impl(::Type{MaxLSESemiring{P}}, a::T, b::T, ta::Val{:C}, tb::Val{:N}, dual::Val{:C}) where {P, T}
    return -log(max(exp(P * -b) - exp(P * -a), zero(T))) / P
end

#
#   a ≤ b
#
function le_impl(::Type{<:MaxLSESemiring}, a::T, b::T) where {T}
    return a <= b
end

#
#   a < b
#
function lt_impl(::Type{<:MaxLSESemiring}, a::T, b::T) where {T}
    return a < b
end
