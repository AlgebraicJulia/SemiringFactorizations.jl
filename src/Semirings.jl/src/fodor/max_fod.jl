struct MaxFodQuantale <: AbstractCommutativeQuantale end

"""
    MaxFod{T} <: Number

The semiring ([0, 1], ∨, ⊤, 0, 1).

  - elements are real numbers in the unit interval: a ∈ [0, 1]
  - addition is maximum: a ∨ b
  - multiplication is nilpotent conjunction

"""
const MaxFod{T} = SemiringNumber{MaxFodQuantale, T}

#
#   0
#
function zero_impl(::Type{MaxFodQuantale}, ::Type{T}) where {T}
    return zero(T)
end

#
#   1
#
function one_impl(::Type{MaxFodQuantale}, ::Type{T}) where {T}
    return one(T)
end

#
#   a ∨ b
#
function add_impl(::Type{MaxFodQuantale}, a::T, b::T) where {T}
    return max(a, b)
end

#
#   a ∧ b
#
function inf_impl(::Type{MaxFodQuantale}, a::T, b::T) where {T}
    return min(a, b)
end

#
#   { 0     if a + b ≤ 1
#   { a ∧ b if a + b > 1
#
function mul_impl(::Type{MaxFodQuantale}, a::T, b::T) where {T}
    return ifelse(a + b <= one(T), zero(T), min(a, b))
end

#
#   { 1           if a ≤ b
#   { (1 - a) ∨ b if a > b
#
function ldiv_impl(::Type{MaxFodQuantale}, a::T, b::T) where {T}
    return ifelse(a <= b, one(T), max(one(T) - a, b))
end

#
#   a ≤ b
#
function le_impl(::Type{MaxFodQuantale}, a::T, b::T) where {T}
    return a <= b
end

#
#   a < b
#
function lt_impl(::Type{MaxFodQuantale}, a::T, b::T) where {T}
    return a < b
end
