struct MinGodSemiring <: AbstractTriConorm end

"""
    MinGod{T} <: Number

The semiring ([0, 1], ∧, ⊥, 0, 1).

  - elements are real numbers in the unit interval: a ∈ [0, 1]
  - addition is minimum: a ∧ b
  - multiplication is maximum: a ∨ b

"""
const MinGod{T} = SemiringNumber{MinGodSemiring, T}

#
#   a ∨ b
#
function mul_impl(::Type{MinGodSemiring}, a::T, b::T, ta::Val{:N}, tb::Val{:N}, dual::Val{:N}) where {T}
    return max(a, b)
end

#
#   { 0 if a ≥ b
#   { b if a < b
#
function mul_impl(::Type{MinGodSemiring}, a::T, b::T, ta::Val{:C}, tb::Val{:N}, dual::Val{:C}) where {T}
    return ifelse(a >= b, zero(T), b)
end
