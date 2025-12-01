struct MinGodQuantale <: AbstractLattice end

"""
    MinGod{T} <: Number

The semiring ([0, 1], ∧, ⊥, 0, 1).

  - elements are real numbers in the unit interval: a ∈ [0, 1]
  - addition is minimum: a ∧ b
  - multiplication is maximum: a ∨ b

"""
const MinGod{T} = SemiringNumber{MinGodQuantale, T}

#
#   1
#
function zero_impl(::Type{MinGodQuantale}, ::Type{T}) where {T}
    return one(T)
end

#
#   0
#
function one_impl(::Type{MinGodQuantale}, ::Type{T}) where {T}
    return zero(T)
end

#
#   a ∧ b
#
function add_impl(::Type{MinGodQuantale}, a::T, b::T) where {T}
    return min(a, b)
end

#
#   a ∨ b
#
function mul_impl(::Type{MinGodQuantale}, a::T, b::T) where {T}
    return max(a, b)
end

#
#   { 0 if a ≥ b
#   { b if a < b
#
function ldiv_impl(::Type{MinGodQuantale}, a::T, b::T) where {T}
    return ifelse(a >= b, zero(T), b)
end

#
#   a ≥ b
#
function le_impl(::Type{MinGodQuantale}, a::T, b::T) where {T}
    return a >= b
end

#
#   a > b
#
function lt_impl(::Type{MinGodQuantale}, a::T, b::T) where {T}
    return a > b
end
