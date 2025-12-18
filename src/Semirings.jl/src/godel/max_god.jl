struct MaxGodSemiring <: AbstractTriNorm end

"""
    MaxGod{T} <: Number

The semiring ([0, 1], ∨, ⊤, 0, 1).

  - elements are real numbers in the unit interval: a ∈ [0, 1]
  - addition is maximum: a ∨ b
  - multiplication is minimum: a ∧ b

"""
const MaxGod{T} = SemiringNumber{MaxGodSemiring, T}

#
#   a ∧ b
#
function mul_impl(::Type{MaxGodSemiring}, a::T, b::T, ta::Val{:N}, tb::Val{:N}, dual::Val{:N}) where {T}
    return min(a, b)
end

#
#   { 1 if a ≤ b
#   { b if a > b
#
function mul_impl(::Type{MaxGodSemiring}, a::T, b::T, ta::Val{:C}, tb::Val{:N}, dual::Val{:C}) where {T}
    return ifelse(a <= b, one(T), b)
end
