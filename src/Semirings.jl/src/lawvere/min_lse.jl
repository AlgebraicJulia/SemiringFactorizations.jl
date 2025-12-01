struct MinLSEQuantale{P} <: AbstractCommutativeQuantale end

"""
    MinLSE{P, T} <: Number

The semiring (ℝ ∪ {-∞, +∞}, ∧, P-LSE, +∞, 0).

  - elements are extended real numbers: a ∈ ℝ ∪ {-∞, +∞}
  - addition is minimum: a ∧ b
  - multiplication is P-LSE: 1/P log (eᴾᵃ + eᴾᵇ)

It is sometimes called the logarithmic semiring.
"""
const MinLSE{P, T} = SemiringNumber{MinLSEQuantale{P}, T}

#
#   +∞
#
function zero_impl(::Type{<:MinLSEQuantale}, ::Type{T}) where {T}
    return typemax(T)
end

#
#   -∞
#
function one_impl(::Type{<:MinLSEQuantale}, ::Type{T}) where {T}
    return typemin(T)
end

#
#   a ∧ b
#
function add_impl(::Type{<:MinLSEQuantale}, a::T, b::T) where {T}
    return min(a, b)
end

#
#   a ∨ b
#
function inf_impl(::Type{<:MinLSEQuantale}, a::T, b::T) where {T}
    return max(a, b)
end

#
#   1/P log (eᴾᵃ + eᴾᵇ)
#
function mul_impl(::Type{MinLSEQuantale{P}}, a::T, b::T) where {P, T}
    return log(exp(P * a) + exp(P * b)) / P
end

#
#   1/P log [eᴾᵇ - eᴾᵃ]₊ 
#
function ldiv_impl(::Type{MinLSEQuantale{P}}, a::T, b::T) where {P, T}
    return log(max(exp(P * b) - exp(P * a), zero(T))) / P
end

#
#   a ≥ b
#
function le_impl(::Type{<:MinLSEQuantale}, a::T, b::T) where {T}
    return a >= b
end

#
#   a > b
#
function lt_impl(::Type{<:MinLSEQuantale}, a::T, b::T) where {T}
    return a > b
end
