struct MaxLSESemiring <: AbstractIntegralSemiring end

"""
    MaxLSE{T} <: Number

The semiring (ℝ ∪ {-∞, +∞}, ∨, NLSE, +∞, 0).

  - elements are extended real numbers: a ∈ ℝ ∪ {-∞, +∞}
  - addition is maximum: a ∨ b
  - multiplication is NLSE: -log (e⁻ᵃ + e⁻ᵇ)

It is sometimes called the logarithmic semiring.
"""
const MaxLSE{T} = SemiringNumber{MaxLSESemiring, T}

#
#   -∞
#
function zero_impl(::Type{MaxLSESemiring}, ::Type{T}, dual::Val{:N}) where {T}
    return typemin(T)
end

#
#   +∞
#
function zero_impl(::Type{MaxLSESemiring}, ::Type{T}, dual::Val{:C}) where {T}
    return typemax(T)
end

#
#   a ∨ b
#
function add_impl(::Type{MaxLSESemiring}, a::T, b::T, dual::Val{:N}) where {T}
    return max(a, b)
end

#
#   a ∧ b
#
function add_impl(::Type{MaxLSESemiring}, a::T, b::T, dual::Val{:C}) where {T}
    return min(a, b)
end

#
#   -log (e⁻ᵃ + e⁻ᵇ)
#
function mul_impl(::Type{MaxLSESemiring}, a::T, b::T, ta::Val{:N}, tb::Val{:N}, dual::Val{:N}) where {T}
    return -log(exp(-a) + exp(-b))
end

#
#   -log [e⁻ᵇ - e⁻ᵃ]₊
#
function mul_impl(::Type{MaxLSESemiring}, a::T, b::T, ta::Val{:C}, tb::Val{:N}, dual::Val{:C}) where {T}
    return -log(max(exp(-b) - exp(-a), zero(T)))
end

#
#   a ≤ b
#
function le_impl(::Type{MaxLSESemiring}, a::T, b::T) where {T}
    return a <= b
end

#
#   a < b
#
function lt_impl(::Type{MaxLSESemiring}, a::T, b::T) where {T}
    return a < b
end
