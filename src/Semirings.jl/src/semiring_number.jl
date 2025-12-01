abstract type AbstractSemiringNumber{T} <: Number end

struct SemiringNumber{A <: AbstractSemiring, T} <: AbstractSemiringNumber{T}
    num::T
end

const QuantaleNumber = SemiringNumber{<:AbstractQuantale}

function SemiringNumber{A}(num::T) where {A <: AbstractSemiring, T}
    return SemiringNumber{A, T}(num)
end

function SemiringNumber{A}(a::SemiringNumber{A}) where {A <: AbstractSemiring}
    return SemiringNumber{A}(parent(a))
end

function SemiringNumber{A, T}(a::SemiringNumber) where {A <: AbstractSemiring, T}
    return SemiringNumber{A, T}(SemiringNumber{A}(a))
end

function SemiringNumber{A, T}(a::SemiringNumber{A}) where {A <: AbstractSemiring, T}
    return SemiringNumber{A, T}(parent(a))
end

function Base.parent(a::SemiringNumber)
    return a.num
end

function Base.show(io::IO, a::SemiringNumber)
    print(io, parent(a))
    return
end

function Base.isapprox(a::SemiringNumber{A}, b::SemiringNumber{A}; kw...) where {A <: AbstractSemiring}
    return isapprox(parent(a), parent(b); kw...)
end

function Base.isapprox(a::SemiringNumber{A, <:Tuple}, b::SemiringNumber{A, <:Tuple}; kw...) where {A <: AbstractSemiring}
    return all(isapprox.(parent(a), parent(b); kw...))
end

function Base.isapprox(a::AbstractArray{<:SemiringNumber{A}}, b::AbstractArray{<:SemiringNumber{A}}; kw...) where {A <: AbstractSemiring}
    return all(isapprox.(a, b; kw...))
end

function Base.promote_rule(::Type{SemiringNumber{A, T}}, ::Type{SemiringNumber{A, U}}) where {A <: AbstractSemiring, T, U}
    V = promote_rule(T, U)
    return SemiringNumber{A, V}
end

# --------- #
# Semirings #
# --------- #

function Base.zero(::Type{SemiringNumber{A, T}}) where {A <: AbstractSemiring, T}
    num = zero_impl(A, T)
    return SemiringNumber{A}(num)
end

function Base.zero(::T) where {T <: SemiringNumber}
    return zero(T)
end

function Base.one(::Type{SemiringNumber{A, T}}) where {A <: AbstractSemiring, T}
    num = one_impl(A, T)
    return SemiringNumber{A}(num)
end

function Base.one(::T) where {T <: SemiringNumber}
    return one(T)
end

"""
    star(a)

Compute the Kleene star a*.
"""
star(a)

function star(a::Missing)
    return missing
end

function star(a::T) where {T <: Number}
    ϵ = one(T)
    return ϵ / (ϵ - a)
end

function star(a::T) where {T <: Rational}
    ϵ = one(T)
    return ifelse(isinf(a), a, ϵ / (ϵ - a))
end

function star(a::SemiringNumber{A}) where {A <: AbstractSemiring}
    num = star_impl(A, parent(a))
    return SemiringNumber{A}(num)
end

function Base.:+(a::SemiringNumber{A, T}, b::SemiringNumber{A, T}) where {A <: AbstractSemiring, T}
    num = add_impl(A, parent(a), parent(b))
    return SemiringNumber{A}(num)
end

function Base.:+(a::StaticInt{0}, b::SemiringNumber)
    return b
end

function Base.:+(a::SemiringNumber, b::StaticInt{0})
    return a
end

function Base.FastMath.add_fast(a::SemiringNumber, b::StaticInt)
    return a + b
end

function Base.FastMath.add_fast(a::StaticInt, b::SemiringNumber)
    return a + b
end

function Base.:*(a::SemiringNumber{A, T}, b::SemiringNumber{A, T}) where {A <: AbstractSemiring, T}
    num = mul_impl(A, parent(a), parent(b))
    return SemiringNumber{A}(num)
end

function Base.:*(a::StaticInt{0}, b::T) where {T <: SemiringNumber}
    return zero(T)
end

function Base.:*(a::T, b::StaticInt{0}) where {T <: SemiringNumber}
    return zero(T)
end

function Base.:*(a::StaticInt{1}, b::SemiringNumber)
    return b
end

function Base.:*(a::SemiringNumber, b::StaticInt{1})
    return a
end

function Base.FastMath.mul_fast(a::StaticInt, b::SemiringNumber)
    return a * b
end

function Base.FastMath.mul_fast(a::SemiringNumber, b::StaticInt)
    return a * b
end

"""
    slmul(a, b)

Compute the product a* × b.
"""
slmul(a, b)

function slmul(a::Number, b::Missing)
    return missing
end

function slmul(a::Number, b::Number)
    return slmul(promote(a, b)...)
end

function slmul(a::T, b::T) where {T <: Number}
    ϵ = one(T)
    return b / (ϵ - a)
end

function slmul(a::T, b::T) where {T <: Rational}
    ϵ = one(T)
    return ifelse(isinf(a) | isinf(b), b, b / (ϵ - a)) 
end

function slmul(a::SemiringNumber{A, T}, b::SemiringNumber{A, T}) where {A, T}
    num = slmul_impl(A, parent(a), parent(b))
    return SemiringNumber{A}(num)    
end

"""
    srmul(b, a)

Compute the product b × a*.
"""
srmul(b, a)

function srmul(b::Missing, a::Number)
    return missing
end

function srmul(b::Number, a::Number)
    return srmul(promote(b, a)...)
end

function srmul(b::T, a::T) where {T <: Number}
    return slmul(a, b)
end

function srmul(b::SemiringNumber{A, T}, a::SemiringNumber{A, T}) where {A, T}
    num = srmul_impl(A, parent(b), parent(a))
    return SemiringNumber{A}(num)    
end

"""
    fma(a, b, c)

Compute the sum (a × b) + c.
"""
fma(a, b, c)

function fma(a::Number, b::Number, c::Number)
    return fma(promote(a, b, c)...)
end

function fma(a::T, b::T, c::T) where {T <: Number}
    return Base.fma(a, b, c)
end

function fma(a::T, b::T, c::T) where {T <: Complex}
    return (a * b) + c
end

function Base.fma(a::SemiringNumber{A, T}, b::SemiringNumber{A, T}, c::SemiringNumber{A, T}) where {A <: AbstractSemiring, T}
    num = mul_add_impl(A, parent(a), parent(b), parent(c))
    return SemiringNumber{A}(num)
end

function Base.fma(a::StaticInt, b::SemiringNumber{A, T}, c::SemiringNumber{A, T}) where {A <: AbstractSemiring, T}
    return (a * b) + c
end

function Base.:(==)(a::SemiringNumber{A}, b::SemiringNumber{A}) where {A <: AbstractSemiring}
    return parent(a) == parent(b)
end

# --------- #
# Quantales #
# --------- #

function Base.typemin(::Type{T}) where {T <: QuantaleNumber}
    return zero(T)
end

function Base.typemin(::T) where {T <: QuantaleNumber}
    return typemin(T)
end

function Base.typemax(::Type{SemiringNumber{A, T}}) where {A <: AbstractQuantale, T}
    num = typemax_impl(A, T)
    return SemiringNumber{A}(num)
end

function Base.typemax(::T) where {T <: QuantaleNumber}
    return typemax(T)
end

"""
    inf(a, b)

Compute the infimum a ∧ b.
"""
inf(a, b)

function inf(a::Number, b::Number)
    return inf(promote(a, b)...) 
end

function inf(a::T, b::T) where {T <: Number}
    return min(a, b)
end

function inf(a::SemiringNumber{A, T}, b::SemiringNumber{A, T}) where {A <: AbstractQuantale, T}
    num = inf_impl(A, parent(a), parent(b))
    return SemiringNumber{A}(num)
end

function Base.:\(a::SemiringNumber{A, T}, b::SemiringNumber{A, T}) where {A <: AbstractQuantale, T}
    num = ldiv_impl(A, parent(a), parent(b))
    return SemiringNumber{A}(num)
end

function Base.:/(b::SemiringNumber{A, T}, a::SemiringNumber{A, T}) where {A <: AbstractQuantale, T}
    num = rdiv_impl(A, parent(b), parent(a))
    return SemiringNumber{A}(num)
end

"""
    sldiv(a, b)

Compute the residual a* \\ b.
"""
sldiv(a, b)

function sldiv(a::Number, b::Number)
    return sldiv(promote(a, b)...)
end

function sldiv(a::T, b::T) where {T <: Number}
    return star(a) \ b
end

function sldiv(a::SemiringNumber{A, T}, b::SemiringNumber{A, T}) where {A <: AbstractQuantale, T}
    num = sldiv_impl(A, parent(a), parent(b))
    return SemiringNumber{A}(num)
end

"""
    srdiv(b, a)

Compute the residual b / a*
"""
srdiv(b, a)

function srdiv(b::Number, a::Number)
    return sldiv(promote(b, a)...)
end

function srdiv(b::T, a::T) where {T <: Number}
    return b / star(a)
end

function srdiv(b::SemiringNumber{A, T}, a::SemiringNumber{A, T}) where {A <: AbstractQuantale, T}
    num = srdiv_impl(A, parent(b), parent(a))
    return SemiringNumber{A}(num)
end

"""
    fli(a, b, c)

Compute (a \\ b) ∧ c.
"""
fli(a, b, c)

function fli(a::Number, b::Number, c::Number)
    return fli(promote(a, b, c)...)
end

function fli(a::T, b::T, c::T) where {T <: Number}
    return (a \ b) ∧ c
end

function fli(a::SemiringNumber{A, T}, b::SemiringNumber{A, T}, c::SemiringNumber{A, T}) where {A <: AbstractQuantale, T}
    num = inf_ldiv_impl(A, parent(a), parent(b), parent(c))
    return SemiringNumber{A}(num)
end

"""
    fri(b, a, c)

Compute (b / a) ∧ c.
"""
fri(b, a, c)

function fri(b::Number, a::Number, c::Number)
    return fri(promote(b, a, c)...)
end

function fri(b::T, a::T, c::T) where {T <: Number}
    return (b / a) ∧ c
end

function fri(b::SemiringNumber{A, T}, a::SemiringNumber{A, T}, c::SemiringNumber{A, T}) where {A <: AbstractQuantale, T}
    num = inf_rdiv_impl(A, parent(b), parent(a), parent(c))
    return SemiringNumber{A}(num)
end

function Base.:<=(a::SemiringNumber{A, T}, b::SemiringNumber{A, T}) where {A <: AbstractQuantale, T}
    return le_impl(A, parent(a), parent(b))
end

function Base.:<(a::SemiringNumber{A, T}, b::SemiringNumber{A, T}) where {A <: AbstractQuantale, T}
    return lt_impl(A, parent(a), parent(b))
end
