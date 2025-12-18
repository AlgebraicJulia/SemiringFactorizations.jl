struct OrAndRelSemiring <: AbstractSemiring end

"""
    OrAndRel <: Number

The semiring (2⁶⁴, ∪, ×, ∅, I).

   - elements are 8x8 binary relations
   - addition is union: a ∪ b
   - multiplication is relative product: a × b

"""
const OrAndRel = SemiringNumber{OrAndRelSemiring, UInt64}

#
#   ∅
#
function zero_impl(::Type{OrAndRelSemiring}, ::Type{T}, dual::Val{:N}) where {T}
    return zero(T)
end

#
#   2⁶⁴
#
function zero_impl(::Type{OrAndRelSemiring}, ::Type{T}, dual::Val{:C}) where {T}
    return typemax(T)
end

#
#   I
#
function one_impl(::Type{OrAndRelSemiring}, ::Type{UInt64}, dual::Val{:N})
    return 0x8040201008040201
end

#
#   a*
#
function star_impl(::Type{OrAndRelSemiring}, a)
    return bst(a)
end

#
#   a ∪ b
#
function add_impl(::Type{OrAndRelSemiring}, a::T, b::T, dual::Val{:N}) where {T}
    return a | b
end

#
#   a ∩ b
#
function add_impl(::Type{OrAndRelSemiring}, a::T, b::T, dual::Val{:C}) where {T}
    return a & b
end

#
#   a × b
#
function mul_impl(::Type{OrAndRelSemiring}, a::T, b::T, ta::Val{:N}, tb::Val{:N}, dual::Val{:N}) where {T}
    return bml(a, b)
end

#
#   (aᵀ × bᶜ)ᶜ
#
function mul_impl(::Type{OrAndRelSemiring}, a::T, b::T, ta::Val{:C}, tb::Val{:N}, dual::Val{:C}) where {T}
    return ~bml(btr(a), ~b)
end

#
#   (aᶜ × bᵀ)ᶜ
#
function mul_impl(::Type{OrAndRelSemiring}, a::T, b::T, ta::Val{:N}, tb::Val{:C}, dual::Val{:C}) where {T}
    return ~bml(~a, btr(b))
end
