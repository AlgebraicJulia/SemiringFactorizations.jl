struct MaxLukQuantale <: AbstractCommutativeQuantale end

"""
    MaxLuk{T} <: Number

The semiring ([0, 1], ∨, ⊤, 0, 1).

  - elements are real numbers in the unit interval: a ∈ [0, 1]
  - addition is maximum: a ∨ b
  - multiplication is Lukasiewicz conjunction: (a + b - 1) ∨ 0

"""
const MaxLuk{T} = SemiringNumber{MaxLukQuantale, T}

#
#   0
#
function zero_impl(::Type{MaxLukQuantale}, ::Type{T}) where {T}
    return zero(T)
end

#
#   1
#
function one_impl(::Type{MaxLukQuantale}, ::Type{T}) where {T}
    return one(T)
end

#
#   a ∨ b
#
function add_impl(::Type{MaxLukQuantale}, a::T, b::T) where {T}
    return max(a, b)
end

#
#   a ∧ b
#
function inf_impl(::Type{MaxLukQuantale}, a::T, b::T) where {T}
    return min(a, b)
end

#
#   (a + b - 1) ∨ 0
#
function mul_impl(::Type{MaxLukQuantale}, a::T, b::T) where {T}
    return max(a + b - one(T), zero(T))
end

#
#   (1 - a + b) ∧ 1
#
function ldiv_impl(::Type{MaxLukQuantale}, a::T, b::T) where {T}
    return min(one(T) - a + b, one(T))
end

#
#   a ≤ b
#
function le_impl(::Type{MaxLukQuantale}, a::T, b::T) where {T}
    return a <= b
end

#
#   a < b
#
function lt_impl(::Type{MaxLukQuantale}, a::T, b::T) where {T}
    return a < b
end
