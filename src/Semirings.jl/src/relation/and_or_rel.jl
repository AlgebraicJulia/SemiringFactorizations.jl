struct AndOrRelQuantale <: AbstractQuantale end

"""
    AndOrRel <: Number

The semiring (2⁶⁴, ∩, †, 2⁶⁴, Iᶜ).

   - elements are 8x8 binary relations
   - addition is intersection: a ∩ b
   - multiplication is relative sum: a † b

"""
const AndOrRel = SemiringNumber{AndOrRelQuantale, UInt64}

#
#   2⁶⁴
#
function zero_impl(::Type{AndOrRelQuantale}, ::Type{T}) where {T}
    return typemax(T)
end

#
#   Iᶜ
#
function one_impl(::Type{AndOrRelQuantale}, ::Type{UInt64})
    return 0x7FBFDFEFF7FBFDFE
end

#
#   ∅
#
function typemax_impl(::Type{AndOrRelQuantale}, ::Type{T}) where {T}
    return zero(T)
end

#
#   ((aᶜ)*)ᶜ
#
function star_impl(::Type{AndOrRelQuantale}, a)
    return ~bst(~a)
end

#
#   a ∩ b
#
function add_impl(::Type{AndOrRelQuantale}, a::T, b::T) where {T}
    return a & b
end

#
#   a ∪ b
#
function inf_impl(::Type{AndOrRelQuantale}, a::T, b::T) where {T}
    return a | b
end

#
#   (aᶜ × bᶜ)ᶜ
#
function mul_impl(::Type{AndOrRelQuantale}, a::T, b::T) where {T}
    return ~bml(~a, ~b)
end

#
#   (aᶜ)ᵀ × b
#
function ldiv_impl(::Type{AndOrRelQuantale}, a::T, b::T) where {T}
    return bml(btr(~a), b)
end

#
#   b × (aᶜ)ᵀ
#
function rdiv_impl(::Type{AndOrRelQuantale}, b::T, a::T) where {T}
    return bml(b, btr(~a))
end
