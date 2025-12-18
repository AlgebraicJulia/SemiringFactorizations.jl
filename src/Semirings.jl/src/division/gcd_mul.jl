struct GCDMulSemiring <: AbstractCommutativeSemiring end

"""
    GCDMul{T} <: Number

The semiring (ℚ⁺ ∪ {+∞}, gcd, ×, 0, 1).

  - elements are nonnegative extended rational numbers: a ∈ ℚ⁺ ∪ {+∞}
  - addition is greatest common divisor: gcd(a, b)
  - multiplication is standard: a × b

"""
const GCDMul = SemiringNumber{GCDMulSemiring}

#
#   0
#
function zero_impl(::Type{GCDMulSemiring}, ::Type{T}, dual::Val{:N}) where {T}
    return zero(T)
end

#
#   +∞
#
function zero_impl(::Type{GCDMulSemiring}, ::Type{T}, dual::Val{:C}) where {T}
    return typemax(T)
end

#
#   1
#
function one_impl(::Type{GCDMulSemiring}, ::Type{T}, dual::Val) where {T}
    return one(T)
end

#
#   { 0  a ∈ ℕ
#   { +∞ a ∉ ℕ
#
function star_impl(::Type{GCDMulSemiring}, a::T) where {T}
    return isinteger(a) ? one(T) : typemax(T)
end

#
#   gcd(a, b)
#
function add_impl(::Type{GCDMulSemiring}, a::T, b::T, dual::Val{:N}) where {T}
    return gcd(a, b)
end

#
#   lcm(a, b)
#
function add_impl(::Type{GCDMulSemiring}, a::T, b::T, dual::Val{:C}) where {T}
    return lcm(a, b)
end

#
#   a × b
#
function mul_impl(::Type{GCDMulSemiring}, a::T, b::T, ta::Val{:N}, tb::Val{:N}, dual::Val{:N}) where {T}
    return iszero(a) || iszero(b) ? zero(T) : a * b
end

#
#   b / a
#
function mul_impl(::Type{GCDMulSemiring}, a::T, b::T, ta::Val{:C}, tb::Val{:N}, dual::Val{:C}) where {T}
    return iszero(a) || isinf(b) ? typemax(T) : b / a
end
