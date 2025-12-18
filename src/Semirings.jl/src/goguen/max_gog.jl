struct MaxGogSemiring <: AbstractTriNorm end

"""
    MaxGog{T} <: Number

The semiring ([0, 1], ∨, ⊤, 0, 1).

  - elements are real numbers in the unit interval: a ∈ [0, 1]
  - addition is maximum: a ∨ b
  - multiplication is standard: a × b

"""
const MaxGog{T} = SemiringNumber{MaxGogSemiring, T}

#
#   a × b
#
function mul_impl(::Type{MaxGogSemiring}, a::T, b::T, ta::Val{:N}, tb::Val{:N}, dual::Val{:N}) where {T}
    return a * b
end

#
#   { 1   if a ≤ b
#   { b/a if a > b
#
function mul_impl(::Type{MaxGogSemiring}, a::T, b::T, ta::Val{:C}, tb::Val{:N}, dual::Val{:C}) where {T}
    return ifelse(a <= b, one(T), b / a)
end
