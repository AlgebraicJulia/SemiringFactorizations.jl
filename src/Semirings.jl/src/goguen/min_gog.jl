struct MinGogQuantale <: AbstractCommutativeQuantale end

"""
    MinGog{T} <: Number

The semiring ([0, 1], ∧, ⊥, 0, 1).

  - elements are real numbers in the unit interval: a ∈ [0, 1]
  - addition is minimum: a ∧ b
  - multiplication is probabilistic sum: a + b - a × b

"""
const MinGog{T} = SemiringNumber{MinGogQuantale, T}

#
#   1
#
function zero_impl(::Type{MinGogQuantale}, ::Type{T}) where {T}
    return one(T)
end

#
#   0
#
function one_impl(::Type{MinGogQuantale}, ::Type{T}) where {T}
    return zero(T)
end

#
#
#   a ∧ b
#
function add_impl(::Type{MinGogQuantale}, a::T, b::T) where {T}
    return min(a, b)
end

#
#   a ∨ b
#
function inf_impl(::Type{MinGogQuantale}, a::T, b::T) where {T}
    return max(a, b)
end

#
#   a + b - a × b
#
function mul_impl(::Type{MinGogQuantale}, a::T, b::T) where {T}
    return a + b - a * b
end

#
#   { 0               if a ≥ b
#   { (b - a)/(1 - a) if a < b
#
function ldiv_impl(::Type{MinGogQuantale}, a::T, b::T) where {T}
    return ifelse(a >= b, zero(T), (b - a) / (one(T) - a))
end

#
#   a ≥ b
#
function le_impl(::Type{MinGogQuantale}, a::T, b::T) where {T}
    return a >= b
end

#
#   a > b
#
function lt_impl(::Type{MinGogQuantale}, a::T, b::T) where {T}
    return a > b
end
