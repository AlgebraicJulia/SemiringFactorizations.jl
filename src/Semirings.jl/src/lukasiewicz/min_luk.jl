struct MinLukSemiring <: AbstractTriConorm end

"""
    MinLuk{T} <: Number

The semiring ([0, 1], ∧, ⊥, 0, 1).

  - elements are real numbers in the unit interval: a ∈ [0, 1]
  - addition is minimum: a ∧ b
  - multiplication is Lukasiewicz disjunction: (a + b) ∧ 1

"""
const MinLuk{T} = SemiringNumber{MinLukSemiring, T}

#
#   (a + b) ∧ 1
#
function mul_impl(::Type{MinLukSemiring}, a::T, b::T, ta::Val{:N}, tb::Val{:N}, dual::Val{:N}) where {T}
    return min(a + b, one(T))
end

#
#   (b - a) ∨ 0
#
function mul_impl(::Type{MinLukSemiring}, a::T, b::T, ta::Val{:C}, tb::Val{:N}, dual::Val{:C}) where {T}
    return max(b - a, zero(T))
end
