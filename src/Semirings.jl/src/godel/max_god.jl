struct MaxGodQuantale <: AbstractLattice end

"""
    MaxGod{T} <: Number

The semiring ([0, 1], ∨, ⊤, 0, 1).

  - elements are real numbers in the unit interval: a ∈ [0, 1]
  - addition is maximum: a ∨ b
  - multiplication is minimum: a ∧ b

"""
const MaxGod{T} = SemiringNumber{MaxGodQuantale, T}

#
#   0
#
function zero_impl(::Type{MaxGodQuantale}, ::Type{T}) where {T}
    return zero(T)
end

#
#   1
#
function one_impl(::Type{MaxGodQuantale}, ::Type{T}) where {T}
    return one(T)
end

#
#   a ∨ b
#
function add_impl(::Type{MaxGodQuantale}, a::T, b::T) where {T}
    return max(a, b)
end

#
#   a ∧ b
#
function mul_impl(::Type{MaxGodQuantale}, a::T, b::T) where {T}
    return min(a, b)
end

#
#   { 1 if a ≤ b
#   { b if a > b
#
function ldiv_impl(::Type{MaxGodQuantale}, a::T, b::T) where {T}
    return ifelse(a <= b, one(T), b)
end

#
#   a ≤ b
#
function le_impl(::Type{MaxGodQuantale}, a::T, b::T) where {T}
    return a <= b
end

#
#   a < b
#
function lt_impl(::Type{MaxGodQuantale}, a::T, b::T) where {T}
    return a < b
end
