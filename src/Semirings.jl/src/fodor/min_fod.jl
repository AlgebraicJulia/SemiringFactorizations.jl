struct MinFodQuantale <: AbstractCommutativeQuantale end

"""
    MinFod{T} <: Number

The semiring ([0, 1], ∧, ⊥, 0, 1).

  - elements are real numbers in the unit interval: a ∈ [0, 1]
  - addition is minimum: a ∧ b
  - multiplication is nilpotent disjunction

"""
const MinFod{T} = SemiringNumber{MinFodQuantale, T}

#
#   1
#
function zero_impl(::Type{MinFodQuantale}, ::Type{T}) where {T}
    return one(T)
end

#
#   0
#
function one_impl(::Type{MinFodQuantale}, ::Type{T}) where {T}
    return zero(T)
end

#
#
#   a ∧ b
#
function add_impl(::Type{MinFodQuantale}, a::T, b::T) where {T}
    return min(a, b)
end

#
#   a ∨ b
#
function inf_impl(::Type{MinFodQuantale}, a::T, b::T) where {T}
    return max(a, b)
end

#
#   { 1         if a + b ≥ 1
#   { max(a, b) if a + b < 1
#
function mul_impl(::Type{MinFodQuantale}, a::T, b::T) where {T}
    return ifelse(a + b >= one(T), one(T), max(a, b))
end
#
#   { 0           if a ≥ b
#   { (1 - a) ∧ b if a < b
#
function ldiv_impl(::Type{MinFodQuantale}, a::T, b::T) where {T}
    return ifelse(a >= b, zero(T), min(one(T) - a, b))
end

#
#   a ≥ b
#
function le_impl(::Type{MinFodQuantale}, a::T, b::T) where {T}
    return a >= b
end

#
#   a > b
#
function lt_impl(::Type{MinFodQuantale}, a::T, b::T) where {T}
    return a > b
end
