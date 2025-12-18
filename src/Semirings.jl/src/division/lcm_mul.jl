struct LCMMulSemiring <: AbstractCommutativeSemiring end

"""
    LCMMul{T} <: Number

The semiring (ℚ⁺ ∪ {+∞}, lcm, ×, +∞, 1).

  - elements are nonnegative extended rational numbers: a ∈ ℚ⁺ ∪ {+∞}
  - addition is least common multiple: lcm(a, b)
  - multiplication is standard: a × b

"""
const LCMMul = SemiringNumber{LCMMulSemiring}

#
#   +∞
#
function zero_impl(::Type{LCMMulSemiring}, ::Type{T}, dual::Val{:N}) where {T}
    return typemax(T)
end

#
#   0
#
function zero_impl(::Type{LCMMulSemiring}, ::Type{T}, dual::Val{:C}) where {T}
    return zero(T)
end

#
#   1
#
function one_impl(::Type{LCMMulSemiring}, ::Type{T}, dual::Val) where {T}
    return one(T)
end

#
#   { 1 a⁻¹ ∈ ℕ
#   { 0 a⁻¹ ∉ ℕ
#
function star_impl(::Type{LCMMulSemiring}, a::T) where {T}
    return isinteger(inv(a)) ? one(T) : zero(T)
end

#
#   lcm(a, b)
#
function add_impl(::Type{LCMMulSemiring}, a::T, b::T, dual::Val{:N}) where {T}
    return lcm(a, b)
end

#
#   gcd(a, b)
#
function add_impl(::Type{LCMMulSemiring}, a::T, b::T, dual::Val{:C}) where {T}
    return gcd(a, b)
end

#
#   a × b
#
function mul_impl(::Type{LCMMulSemiring}, a::T, b::T, ta::Val{:N}, tb::Val{:N}, dual::Val{:N}) where {T}
    return isinf(a) || isinf(b) ? typemax(T) : a * b
end

#
#   b / a
#
function mul_impl(::Type{LCMMulSemiring}, a::T, b::T, ta::Val{:C}, tb::Val{:N}, dual::Val{:C}) where {T}
    return isinf(a) || iszero(b) ? zero(T) : b / a
end
