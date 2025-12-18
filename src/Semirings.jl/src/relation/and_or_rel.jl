struct AndOrRelSemiring <: AbstractSemiring end

"""
    AndOrRel <: Number

The semiring (2⁶⁴, ∩, †, 2⁶⁴, Iᶜ).

   - elements are 8x8 binary relations
   - addition is intersection: a ∩ b
   - multiplication is relative sum: a † b

"""
const AndOrRel = SemiringNumber{AndOrRelSemiring, UInt64}

#
#   2⁶⁴
#
function zero_impl(::Type{AndOrRelSemiring}, ::Type{T}, dual::Val{:N}) where {T}
    return typemax(T)
end

#
#   ∅
#
function zero_impl(::Type{AndOrRelSemiring}, ::Type{T}, dual::Val{:C}) where {T}
    return zero(T)
end

#
#   Iᶜ
#
function one_impl(::Type{AndOrRelSemiring}, ::Type{UInt64}, dual::Val{:N})
    return 0x7FBFDFEFF7FBFDFE
end

#
#   ((aᶜ)*)ᶜ
#
function star_impl(::Type{AndOrRelSemiring}, a)
    return ~bst(~a)
end

#
#   a ∩ b
#
function add_impl(::Type{AndOrRelSemiring}, a::T, b::T, dual::Val{:N}) where {T}
    return a & b
end

#
#   a ∪ b
#
function add_impl(::Type{AndOrRelSemiring}, a::T, b::T, dual::Val{:C}) where {T}
    return a | b
end

#
#   (aᶜ × bᶜ)ᶜ
#
function mul_impl(::Type{AndOrRelSemiring}, a::T, b::T, ta::Val{:N}, tb::Val{:N}, dual::Val{:N}) where {T}
    return ~bml(~a, ~b)
end

#
#   (aᶜ)ᵀ × b
#
function mul_impl(::Type{AndOrRelSemiring}, a::T, b::T, ta::Val{:C}, tb::Val{:N}, dual::Val{:C}) where {T}
    return bml(btr(~a), b)
end

#
#   b × (aᶜ)ᵀ
#
function mul_impl(::Type{AndOrRelSemiring}, a::T, b::T, ta::Val{:N}, tb::Val{:C}, dual::Val{:C}) where {T}
    return bml(a, btr(~b))
end
