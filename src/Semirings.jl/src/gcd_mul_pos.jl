struct GCDMulPosQuantale <: AbstractCommutativeQuantale end

"""
    GCDMulPos{T} <: Number

The semiring (ℕ, gcd, ×, 0, 1).

  - elements are nonnegative integers: a ∈ ℕ
  - addition is greatest common divisor: gcd(a, b)
  - multiplication is standard: a × b

"""
const GCDMulPos = SemiringNumber{GCDMulPosQuantale}

#
#   0
#
function zero_impl(::Type{GCDMulPosQuantale}, ::Type{T}) where {T}
    return zero(T)
end

#
#   1
#
function one_impl(::Type{GCDMulPosQuantale}, ::Type{T}) where {T}
    return one(T)
end

#
#   1
#
function typemax_impl(::Type{GCDMulPosQuantale}, ::Type{T}) where {T}
    return one(T)
end

#
#   1
#
function star_impl(::Type{GCDMulPosQuantale}, a::T) where {T}
    return one(T)
end

#
#   gcd(a, b)
#
function add_impl(::Type{GCDMulPosQuantale}, a::T, b::T) where {T}
    return gcd(a, b)
end

#
#   lcm(a, b)
#
function inf_impl(::Type{GCDMulPosQuantale}, a::T, b::T) where {T}
    return lcm(a, b)
end

#
#   a × b
#
function mul_impl(::Type{GCDMulPosQuantale}, a::T, b::T) where {T}
    return a * b
end

#
#   b ÷ gcd(a, b)
#
function ldiv_impl(::Type{GCDMulPosQuantale}, a::T, b::T) where {T}
    return iszero(a) ? one(T) : b ÷ gcd(a, b)
end
