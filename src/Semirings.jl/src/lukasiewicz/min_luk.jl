struct MinLukQuantale <: AbstractCommutativeQuantale end

"""
    MinLuk{T} <: Number

The semiring ([0, 1], ∧, ⊥, 0, 1).

  - elements are real numbers in the unit interval: a ∈ [0, 1]
  - addition is minimum: a ∧ b
  - multiplication is Lukasiewicz disjunction: (a + b) ∧ 1

"""
const MinLuk{T} = SemiringNumber{MinLukQuantale, T}

#
#   1
#
function zero_impl(::Type{MinLukQuantale}, ::Type{T}) where {T}
    return one(T)
end

#
#   0
#
function one_impl(::Type{MinLukQuantale}, ::Type{T}) where {T}
    return zero(T)
end

#
#
#   a ∧ b
#
function add_impl(::Type{MinLukQuantale}, a::T, b::T) where {T}
    return min(a, b)
end

#
#   a ∨ b
#
function inf_impl(::Type{MinLukQuantale}, a::T, b::T) where {T}
    return max(a, b)
end

#
#   (a + b) ∧ 1
#
function mul_impl(::Type{MinLukQuantale}, a::T, b::T) where {T}
    return min(a + b, one(T))
end

#
#   (b - a) ∨ 0
#
function ldiv_impl(::Type{MinLukQuantale}, a::T, b::T) where {T}
    return max(b - a, zero(T))
end

#
#   a ≥ b
#
function le_impl(::Type{MinLukQuantale}, a::T, b::T) where {T}
    return a >= b
end

#
#   a > b
#
function lt_impl(::Type{MinLukQuantale}, a::T, b::T) where {T}
    return a > b
end
