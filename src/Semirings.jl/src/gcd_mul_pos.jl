struct GCDMulPosSemiring <: AbstractIntegralSemiring end

"""
    GCDMulPos{T} <: Number

The semiring (ℕ, gcd, ×, 0, 1).

  - elements are nonnegative integers: a ∈ ℕ
  - addition is greatest common divisor: gcd(a, b)
  - multiplication is standard: a × b

"""
const GCDMulPos = SemiringNumber{GCDMulPosSemiring}

#
#   0
#
function zero_impl(::Type{GCDMulPosSemiring}, ::Type{T}, dual::Val{:N}) where {T}
    return zero(T)
end

#
#   1
#
function zero_impl(::Type{GCDMulPosSemiring}, ::Type{T}, dual::Val{:C}) where {T}
    return one(T)
end

#
#   gcd(a, b)
#
function add_impl(::Type{GCDMulPosSemiring}, a::T, b::T, dual::Val{:N}) where {T}
    return gcd(a, b)
end

#
#   lcm(a, b)
#
function add_impl(::Type{GCDMulPosSemiring}, a::T, b::T, dual::Val{:C}) where {T}
    return lcm(a, b)
end

#
#   a × b
#
function mul_impl(::Type{GCDMulPosSemiring}, a::T, b::T, ta::Val{:N}, tb::Val{:N}, dual::Val{:N}) where {T}
    return a * b
end

#
#   b ÷ gcd(a, b)
#
function mul_impl(::Type{GCDMulPosSemiring}, a::T, b::T, ta::Val{:C}, tb::Val{:N}, dual::Val{:C}) where {T}
    return iszero(a) ? one(T) : b ÷ gcd(a, b)
end
