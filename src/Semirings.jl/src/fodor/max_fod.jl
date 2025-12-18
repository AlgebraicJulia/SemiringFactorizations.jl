struct MaxFodSemiring <: AbstractTriNorm end

"""
    MaxFod{T} <: Number

The semiring ([0, 1], ∨, ⊤, 0, 1).

  - elements are real numbers in the unit interval: a ∈ [0, 1]
  - addition is maximum: a ∨ b
  - multiplication is nilpotent conjunction

"""
const MaxFod{T} = SemiringNumber{MaxFodSemiring, T}

#
#   { 0     if a + b ≤ 1
#   { a ∧ b if a + b > 1
#
function mul_impl(::Type{MaxFodSemiring}, a::T, b::T, ta::Val{:N}, tb::Val{:N}, dual::Val{:N}) where {T}
    return ifelse(a + b <= one(T), zero(T), min(a, b))
end

#
#   { 1           if a ≤ b
#   { (1 - a) ∨ b if a > b
#
function mul_impl(::Type{MaxFodSemiring}, a::T, b::T, ta::Val{:C}, tb::Val{:N}, dual::Val{:C}) where {T}
    return ifelse(a <= b, one(T), max(one(T) - a, b))
end
