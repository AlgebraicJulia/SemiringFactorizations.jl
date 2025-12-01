struct OrAndRelQuantale <: AbstractQuantale end

"""
    OrAndRel <: Number

The semiring (2⁶⁴, ∪, ×, ∅, I).

   - elements are 8x8 binary relations
   - addition is union: a ∪ b
   - multiplication is relative product: a × b

"""
const OrAndRel = SemiringNumber{OrAndRelQuantale, UInt64}

#
#   ∅
#
function zero_impl(::Type{OrAndRelQuantale}, ::Type{T}) where {T}
    return zero(T)
end

#
#   I
#
function one_impl(::Type{OrAndRelQuantale}, ::Type{UInt64})
    return 0x8040201008040201
end

#
#   2⁶⁴
#
function typemax_impl(::Type{OrAndRelQuantale}, ::Type{T}) where {T}
    return typemax(T)
end

#
#   a*
#
function star_impl(::Type{OrAndRelQuantale}, a)
    return bst(a)
end

#
#   a ∪ b
#
function add_impl(::Type{OrAndRelQuantale}, a::T, b::T) where {T}
    return a | b
end

#
#   a ∩ b
#
function inf_impl(::Type{OrAndRelQuantale}, a::T, b::T) where {T}
    return a & b
end

#
#   a × b
#
function mul_impl(::Type{OrAndRelQuantale}, a::T, b::T) where {T}
    return bml(a, b)
end

#
#   (aᵀ × bᶜ)ᶜ
#
function ldiv_impl(::Type{OrAndRelQuantale}, a::T, b::T) where {T}
    return ~bml(btr(a), ~b)
end

#
#   (bᶜ × aᵀ)ᶜ
#
function rdiv_impl(::Type{OrAndRelQuantale}, b::T, a::T) where {T}
    return ~bml(~b, btr(a))
end
