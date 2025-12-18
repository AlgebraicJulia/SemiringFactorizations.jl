struct MinFodSemiring <: AbstractTriConorm end

"""
    MinFod{T} <: Number

The semiring ([0, 1], ∧, ⊥, 0, 1).

  - elements are real numbers in the unit interval: a ∈ [0, 1]
  - addition is minimum: a ∧ b
  - multiplication is nilpotent disjunction

"""
const MinFod{T} = SemiringNumber{MinFodSemiring, T}

#
#   { 1         if a + b ≥ 1
#   { max(a, b) if a + b < 1
#
function mul_impl(::Type{MinFodSemiring}, a::T, b::T, ta::Val{:N}, tb::Val{:N}, dual::Val{:N}) where {T}
    return ifelse(a + b >= one(T), one(T), max(a, b))
end
#
#   { 0           if a ≥ b
#   { (1 - a) ∧ b if a < b
#
function mul_impl(::Type{MinFodSemiring}, a::T, b::T, ta::Val{:C}, tb::Val{:N}, dual::Val{:C}) where {T}
    return ifelse(a >= b, zero(T), min(one(T) - a, b))
end
