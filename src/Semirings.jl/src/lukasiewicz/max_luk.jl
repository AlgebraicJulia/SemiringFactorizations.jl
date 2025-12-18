struct MaxLukSemiring <: AbstractTriNorm end

"""
    MaxLuk{T} <: Number

The semiring ([0, 1], ∨, ⊤, 0, 1).

  - elements are real numbers in the unit interval: a ∈ [0, 1]
  - addition is maximum: a ∨ b
  - multiplication is Lukasiewicz conjunction: (a + b - 1) ∨ 0

"""
const MaxLuk{T} = SemiringNumber{MaxLukSemiring, T}

#
#   (a + b - 1) ∨ 0
#
function mul_impl(::Type{MaxLukSemiring}, a::T, b::T, ta::Val{:N}, tb::Val{:N}, dual::Val{:N}) where {T}
    return max(a + b - one(T), zero(T))
end

#
#   (1 - a + b) ∧ 1
#
function mul_impl(::Type{MaxLukSemiring}, a::T, b::T, ta::Val{:C}, tb::Val{:N}, dual::Val{:C}) where {T}
    return min(one(T) - a + b, one(T))
end
