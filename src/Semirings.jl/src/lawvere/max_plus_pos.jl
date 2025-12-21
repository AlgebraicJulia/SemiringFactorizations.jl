struct MaxPlusPosSemiring <: AbstractIntegralSemiring end

"""
    MaxPlusPos{T} <: Number

The semiring (ℝ⁺ ∪ {+∞}, ∨, +⁻¹, 0, +∞).

  - elements are extended nonnegative real numbers: a ∈ ℝ⁺ ∪ {+∞}
  - addition is maximum: a ∨ b
  - multiplication is inverse addition: (a⁻¹ + b⁻¹)⁻¹

"""
const MaxPlusPos{T} = SemiringNumber{MaxPlusPosSemiring, T}

#
#   0
#
function zero_impl(::Type{<:MaxPlusPosSemiring}, ::Type{T}, dual::Val{:N}) where {T}
    return zero(T)
end

#
#   +∞
#
function zero_impl(::Type{<:MaxPlusPosSemiring}, ::Type{T}, dual::Val{:C}) where {T}
    return typemax(T)
end

#
#   a ∨ b
#
function add_impl(::Type{<:MaxPlusPosSemiring}, a::T, b::T, dual::Val{:N}) where {T}
    return max(a, b)
end

#
#   a ∧ b
#
function add_impl(::Type{<:MaxPlusPosSemiring}, a::T, b::T, dual::Val{:C}) where {T}
    return min(a, b)
end

#
#   (a⁻¹ + b⁻¹)⁻¹
#
function mul_impl(::Type{MaxPlusPosSemiring}, a::T, b::T, ta::Val{:N}, tb::Val{:N}, dual::Val{:N}) where {T}
    return inv(inv(a) + inv(b))
end

function mul_impl(::Type{MaxPlusPosSemiring}, a::T, b::T, ta::Val{:N}, tb::Val{:N}, dual::Val{:N}) where {T <: Rational}
    return iszero(a) ? a : inv(inv(a) + inv(b))
end

#
#   [b⁻¹ - a⁻¹]₊⁻¹
#
function mul_impl(::Type{MaxPlusPosSemiring}, a::T, b::T, ta::Val{:C}, tb::Val{:N}, dual::Val{:C}) where {T}
    return inv(max(inv(b) - inv(a), zero(T)))
end

function mul_impl(::Type{MaxPlusPosSemiring}, a::T, b::T, ta::Val{:C}, tb::Val{:N}, dual::Val{:C}) where {T <: Rational}
    return iszero(a) ? typemax(T) : inv(max(inv(b) - inv(a), zero(T)))
end

#
#   a ≤ b
#
function le_impl(::Type{MaxPlusPosSemiring}, a::T, b::T) where {T}
    return a <= b
end

#
#   a < b
#
function lt_impl(::Type{MaxPlusPosSemiring}, a::T, b::T) where {T}
    return a < b
end
