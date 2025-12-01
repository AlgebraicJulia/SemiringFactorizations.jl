struct MaxGogQuantale <: AbstractCommutativeQuantale end

"""
    MaxGog{T} <: Number

The semiring ([0, 1], ∨, ⊤, 0, 1).

  - elements are real numbers in the unit interval: a ∈ [0, 1]
  - addition is maximum: a ∨ b
  - multiplication is standard: a × b

"""
const MaxGog{T} = SemiringNumber{MaxGogQuantale, T}

#
#   0
#
function zero_impl(::Type{MaxGogQuantale}, ::Type{T}) where {T}
    return zero(T)
end

#
#   1
#
function one_impl(::Type{MaxGogQuantale}, ::Type{T}) where {T}
    return one(T)
end

#
#   a ∨ b
#
function add_impl(::Type{MaxGogQuantale}, a::T, b::T) where {T}
    return max(a, b)
end

#
#   a ∧ b
#
function inf_impl(::Type{MaxGogQuantale}, a::T, b::T) where {T}
    return min(a, b)
end

#
#   a × b
#
function mul_impl(::Type{MaxGogQuantale}, a::T, b::T) where {T}
    return a * b
end

#
#   { 1   if a ≤ b
#   { b/a if a > b
#
function ldiv_impl(::Type{MaxGogQuantale}, a::T, b::T) where {T}
    return ifelse(a <= b, one(T), b / a)
end

#
#   a ≤ b
#
function le_impl(::Type{MaxGogQuantale}, a::T, b::T) where {T}
    return a <= b
end

#
#   a < b
#
function lt_impl(::Type{MaxGogQuantale}, a::T, b::T) where {T}
    return a < b
end
