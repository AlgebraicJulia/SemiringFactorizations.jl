struct MaxLSEQuantale{P} <: AbstractCommutativeQuantale end

"""
    MaxLSE{P, T} <: Number

The semiring (ℝ ∪ {-∞, +∞}, ∨, -P-LSE, +∞, 0).

  - elements are extended real numbers: a ∈ ℝ ∪ {-∞, +∞}
  - addition is maximum: a ∨ b
  - multiplication is -P-LSE: -1/P log (e⁻ᴾᵃ + e⁻ᴾᵇ)

It is sometimes called the logarithmic semiring.
"""
const MaxLSE{P, T} = SemiringNumber{MaxLSEQuantale{P}, T}

#
#   -∞
#
function zero_impl(::Type{<:MaxLSEQuantale}, ::Type{T}) where {T}
    return typemin(T)
end

#
#   +∞
#
function one_impl(::Type{<:MaxLSEQuantale}, ::Type{T}) where {T}
    return typemax(T)
end

#
#   a ∨ b
#
function add_impl(::Type{<:MaxLSEQuantale}, a::T, b::T) where {T}
    return max(a, b)
end

#
#   a ∧ b
#
function inf_impl(::Type{<:MaxLSEQuantale}, a::T, b::T) where {T}
    return min(a, b)
end

#
#   -1/P log (e⁻ᴾᵃ + e⁻ᵇ)
#
function mul_impl(::Type{MaxLSEQuantale{P}}, a::T, b::T) where {P, T}
    return -log(exp(P * -a) + exp(P * -b)) / P
end

#
#   -1/P log [e⁻ᴾᵇ - e⁻ᴾᵃ]₊ 
#
function ldiv_impl(::Type{MaxLSEQuantale{P}}, a::T, b::T) where {P, T}
    return -log(max(exp(P * -b) - exp(P * -a), zero(T))) / P
end

#
#   a ≤ b
#
function le_impl(::Type{<:MaxLSEQuantale}, a::T, b::T) where {T}
    return a <= b
end

#
#   a < b
#
function lt_impl(::Type{<:MaxLSEQuantale}, a::T, b::T) where {T}
    return a < b
end
