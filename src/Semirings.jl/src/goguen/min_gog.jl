struct MinGogSemiring <: AbstractTriConorm end

"""
    MinGog{T} <: Number

The semiring ([0, 1], ∧, ⊥, 0, 1).

  - elements are real numbers in the unit interval: a ∈ [0, 1]
  - addition is minimum: a ∧ b
  - multiplication is probabilistic sum: a + b - a × b

"""
const MinGog{T} = SemiringNumber{MinGogSemiring, T}

#
#   a + b - a × b
#
function mul_impl(::Type{MinGogSemiring}, a::T, b::T, ta::Val{:N}, tb::Val{:N}, dual::Val{:N}) where {T}
    return a + b - a * b
end

#
#   { 0               if a ≥ b
#   { (b - a)/(1 - a) if a < b
#
function mul_impl(::Type{MinGogSemiring}, a::T, b::T, ta::Val{:C}, tb::Val{:N}, dual::Val{:C}) where {T}
    return ifelse(a >= b, zero(T), (b - a) / (one(T) - a))
end
