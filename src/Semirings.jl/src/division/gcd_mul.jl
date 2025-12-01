struct GCDMulQuantale <: AbstractCommutativeQuantale end

"""
    GCDMul{T} <: Number

The semiring (ℚ⁺ ∪ {+∞}, gcd, ×, 0, 1).

  - elements are nonnegative extended rational numbers: a ∈ ℚ⁺ ∪ {+∞}
  - addition is greatest common divisor: gcd(a, b)
  - multiplication is standard: a × b

"""
const GCDMul = SemiringNumber{GCDMulQuantale}

#
#   0
#
function zero_impl(::Type{GCDMulQuantale}, ::Type{T}) where {T}
    return zero(T)
end

#
#   1
#
function one_impl(::Type{GCDMulQuantale}, ::Type{T}) where {T}
    return one(T)
end

#
#   +∞
#
function typemax_impl(::Type{GCDMulQuantale}, ::Type{T}) where {T}
    return typemax(T)
end

#
#   { 0  a ∈ ℕ
#   { +∞ a ∉ ℕ
#
function star_impl(::Type{GCDMulQuantale}, a::T) where {T}
    return isinteger(a) ? one(T) : typemax(T)
end

#
#   gcd(a, b)
#
function add_impl(::Type{GCDMulQuantale}, a::T, b::T) where {T}
    return gcd(a, b)
end

#
#   lcm(a, b)
#
function inf_impl(::Type{GCDMulQuantale}, a::T, b::T) where {T}
    return lcm(a, b)
end

#
#   a × b
#
function mul_impl(::Type{GCDMulQuantale}, a::T, b::T) where {T}
    return iszero(a) || iszero(b) ? zero(T) : a * b
end

#
#   b / a
#
function ldiv_impl(::Type{GCDMulQuantale}, a::T, b::T) where {T}
    return iszero(a) || isinf(b) ? typemax(T) : b / a
end
