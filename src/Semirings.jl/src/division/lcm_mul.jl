struct LCMMulQuantale <: AbstractCommutativeQuantale end

"""
    LCMMul{T} <: Number

The semiring (ℚ⁺ ∪ {+∞}, lcm, ×, +∞, 1).

  - elements are nonnegative extended rational numbers: a ∈ ℚ⁺ ∪ {+∞}
  - addition is least common multiple: lcm(a, b)
  - multiplication is standard: a × b

"""
const LCMMul = SemiringNumber{LCMMulQuantale}

#
#   +∞
#
function zero_impl(::Type{LCMMulQuantale}, ::Type{T}) where {T}
    return typemax(T)
end

#
#   1
#
function one_impl(::Type{LCMMulQuantale}, ::Type{T}) where {T}
    return one(T)
end

#
#   0
#
function typemax_impl(::Type{LCMMulQuantale}, ::Type{T}) where {T}
    return zero(T)
end

#
#   { 1 a⁻¹ ∈ ℕ
#   { 0 a⁻¹ ∉ ℕ
#
function star_impl(::Type{LCMMulQuantale}, a::T) where {T}
    return isinteger(inv(a)) ? one(T) : zero(T)
end

#
#   lcm(a, b)
#
function add_impl(::Type{LCMMulQuantale}, a::T, b::T) where {T}
    return lcm(a, b)
end

#
#   gcd(a, b)
#
function inf_impl(::Type{LCMMulQuantale}, a::T, b::T) where {T}
    return gcd(a, b)
end

#
#   a × b
#
function mul_impl(::Type{LCMMulQuantale}, a::T, b::T) where {T}
    return isinf(a) || isinf(b) ? typemax(T) : a * b
end

#
#   b / a
#
function ldiv_impl(::Type{LCMMulQuantale}, a::T, b::T) where {T}
    return isinf(a) || iszero(b) ? zero(T) : b / a
end
