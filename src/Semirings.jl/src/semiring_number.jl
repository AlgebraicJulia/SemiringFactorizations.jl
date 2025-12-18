abstract type AbstractSemiringNumber{T} <: Number end

struct SemiringNumber{A <: AbstractSemiring, T} <: AbstractSemiringNumber{T}
    num::T
end

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

function Base.promote_rule(::Type{SemiringNumber{A, T}}, ::Type{SemiringNumber{A, U}}) where {A <: AbstractSemiring, T, U}
    V = promote_rule(T, U)
    return SemiringNumber{A, V}
end

#
#
#

function zero_impl(::Type{T}, dual::Val) where {T <: Number}
    return zero(T)
end

function zero_impl(::Type{SemiringNumber{A, T}}, dual::Val) where {A <: AbstractSemiring, T}
    num = zero_impl(A, T, dual)
    return SemiringNumber{A}(num)
end

function one_impl(::Type{T}, dual::Val) where {T <: Number}
    return one(T)
end

function one_impl(::Type{SemiringNumber{A, T}}, dual::Val) where {A <: AbstractSemiring, T}
    num = one_impl(A, T, dual)
    return SemiringNumber{A}(num)
end

function star_impl(a::T) where {T <: Number}
    return inv(one(T) - a)
end

function star_impl(a::SemiringNumber{A}) where {A <: AbstractSemiring}
    num = star_impl(A, parent(a))
    return SemiringNumber{A}(num)
end

function id_impl(a::SemiringNumber{A}, dual::Val) where {A <: AbstractSemiring}
    num = id_impl(A, parent(a), dual)
    return SemiringNumber{A}(num)
end

function add_impl(a::Number, b::Number, dual::Val)
    return add_impl(promote(a, b)..., dual)
end

function add_impl(a::T, b::T, dual::Val) where {T <: Number}
    return a + b
end

function add_impl(a::T, b::T, dual::Val) where {A <: AbstractSemiring, T <: SemiringNumber{A}}
    num = add_impl(A, parent(a), parent(b), dual)
    return SemiringNumber{A}(num)
end

function add_impl(a::StaticInt{0}, b::SemiringNumber, dual::Val{:N})
    return b
end

function add_impl(a::SemiringNumber, b::StaticInt{0}, dual::Val{:N})
    return a
end

function mul_impl(a::Number, b::Number, ta::Val, tb::Val, dual::Val)
    return mul_impl(promote(a, b)..., ta, tb, dual)
end

function mul_impl(a::T, b::T, ta::Val, tb::Val, dual::Val) where {T <: Number}
    return a * b
end

function mul_impl(a::T, b::T, ta::Val, tb::Val, dual::Val) where {A <: AbstractSemiring, T <: SemiringNumber{A}}
    num = mul_impl(A, parent(a), parent(b), ta, tb, dual)
    return SemiringNumber{A}(num)
end

function mul_impl(a::StaticInt{0}, b::T, ta::Val{R}, tb::Val, dual::Val{R}) where {T <: SemiringNumber, R}
    return zero_impl(T, ta)
end

function mul_impl(a::T, b::StaticInt{0}, ta::Val, tb::Val{R}, dual::Val{R}) where {T <: SemiringNumber, R}
    return zero_impl(T, tb)
end

function mul_impl(a::StaticInt{1}, b::SemiringNumber, ta::Val{R}, tb::Val, dual::Val{R}) where {R}
    return id_impl(b, tb)
end

function mul_impl(a::SemiringNumber, b::StaticInt{1}, ta::Val, tb::Val{R}, dual::Val{R}) where {R}
    return id_impl(a, ta)
end

function mul_add_impl(a::Number, b::Number, c::Number, ta::Val, tb::Val, dual::Val)
    return mul_add_impl(promote(a, b, c)..., ta, tb, dual)
end

function mul_add_impl(a::Union{StaticInt, T}, b::T, c::T, ta::Val, tb::Val, dual::Val) where {T <: Number}
    return add_impl(mul_impl(a, b, ta, tb, dual), dual)
end

function mul_add_impl(a::T, b::T, c::T, ta::Val, tb::Val, dual::Val) where {T <: Union{Float16, Float32, Float64, BigFloat, Integer, Rational}}
    return Base.fma(a, b, c)
end

function mul_add_impl(a::T, b::T, c::T, ta::Val, tb::Val, dual::Val) where {A <: AbstractSemiring, T <: SemiringNumber{A}}
    num = mul_add_impl(A, parent(a), parent(b), parent(c), ta, tb, dual)
    return SemiringNumber{A}(num)
end

function smul_impl(a::Number, b::Number, ta::Val, tb::Val, side::Val)
    return smul_impl(promote(a, b)..., ta, tb, side)
end

function smul_impl(a::T, b::T, ta::Val, tb::Val, side::Val) where {T <: Number}
    return b / (one(T) - a)
end

function smul_impl(a::T, b::T, ta::Val, tb::Val, side::Val) where {A <: AbstractSemiring, T <: SemiringNumber{A}}
    num = smul_impl(A, parent(a), parent(b), ta, tb, side)
    return SemiringNumber{A}(num)
end

#
#
#

function Base.zero(::Type{T}) where {T <: SemiringNumber}
    return zero_impl(T, Val(:N))
end

function Base.zero(::T) where {T <: SemiringNumber}
    return zero(T)
end

function Base.one(::Type{T}) where {T <: SemiringNumber}
    return one_impl(T, Val(:N))
end

function Base.one(::T) where {T <: SemiringNumber}
    return one(T)
end

"""
    star(a)

Compute the Kleene star ``a^*``, equal to the
infinite sum

```math
    a^* = 1 + a + a^2 + \\cdots
```

``a^*`` is the least solution to the fixed-point
equation

```math
    x = a x + 1.
```
"""
function star(a)
    return star_impl(a)
end

function Base.conj(a::SemiringNumber)
    return id_impl(a, Val(:C))
end

function Base.:+(a::SemiringNumber, b::SemiringNumber)
    return add_impl(a, b, Val(:N))
end

function Base.:+(a::StaticInt, b::SemiringNumber)
    return add_impl(a, b, Val(:N))
end

function Base.:+(a::SemiringNumber, b::StaticInt)
    return add_impl(a, b, Val(:N))
end

function Base.FastMath.add_fast(a::SemiringNumber, b::StaticInt)
    return a + b
end

function Base.FastMath.add_fast(a::StaticInt, b::SemiringNumber)
    return a + b
end

function Base.:*(a::SemiringNumber, b::SemiringNumber)
    return mul_impl(a, b, Val(:N), Val(:N), Val(:N))
end

function Base.:*(a::StaticInt, b::SemiringNumber)
    return mul_impl(a, b, Val(:N), Val(:N), Val(:N))
end

function Base.:*(a::SemiringNumber, b::StaticInt)
    return mul_impl(a, b, Val(:N), Val(:N), Val(:N))
end

function Base.FastMath.mul_fast(a::StaticInt, b::SemiringNumber)
    return a * b
end

function Base.FastMath.mul_fast(a::SemiringNumber, b::StaticInt)
    return a * b
end

function Base.fma(a::Union{StaticInt, SemiringNumber}, b::SemiringNumber, c::SemiringNumber)
    return mul_add_impl(a, b, c, Val(:N), Val(:N), Val(:N))
end

function Base.:(==)(a::SemiringNumber{A}, b::SemiringNumber{A}) where {A <: AbstractSemiring}
    return parent(a) == parent(b)
end

function Base.isapprox(a::SemiringNumber{A}, b::SemiringNumber{A}; kw...) where {A <: AbstractSemiring}
    return isapprox(parent(a), parent(b); kw...)
end

function Base.typemin(::Type{T}) where {T <: SemiringNumber}
    return zero(T)
end

function Base.typemin(::T) where {T <: SemiringNumber}
    return typemin(T)
end

function Base.typemax(::Type{T}) where {T <: SemiringNumber}
    return zero_impl(T, Val(:C))
end

function Base.typemax(::T) where {T <: SemiringNumber}
    return typemax(T)
end

"""
    &(a, b)

Compute the infimum ``a \\wedge b``, i.e.
the greatest solution ``x`` to the equation

```math
    x \\leq a \\text{and} x \\leq b.
```
"""
function Base.:&(a::Union{SemiringNumber, AbstractArray{<:SemiringNumber}}, b::Union{SemiringNumber, AbstractArray{<:SemiringNumber}})
    return add_impl(a, b, Val(:C))
end

function ⅋(a, b)
    return mul_impl(a, b, Val(:N), Val(:N), Val(:C))
end

"""
    \\(a, b)

Compute the residual ``a \\ b``, i.e.
the greatest solution ``x`` to the equation

```math
    ax \\leq b.
```
"""
function Base.:\(a::SemiringNumber, b::SemiringNumber)
    return mul_impl(a, b, Val(:C), Val(:N), Val(:C))
end

function Base.:\(a::AbstractMatrix{<:SemiringNumber}, b::AbstractVecOrMat{<:SemiringNumber})
    return mul_impl(a, b, Val(:C), Val(:N), Val(:C))
end

function Base.:\(a::AbstractVector{<:SemiringNumber}, b::AbstractVecOrMat{<:SemiringNumber})
    return mul_impl(a, b, Val(:C), Val(:N), Val(:C))
end

function Base.:\(a::SparseMatrixCSC{<:SemiringNumber}, b::AbstractMatrix{<:SemiringNumber})
    return mul_impl(a, b, Val(:C), Val(:N), Val(:C))
end

function Base.:\(a::SparseMatrixCSC{<:SemiringNumber}, b::AbstractVector{<:SemiringNumber})
    return mul_impl(a, b, Val(:C), Val(:N), Val(:C))
end

"""
    /(b, a)

Compute the residual ``b / a``, i.e.
the greatest solution ``x`` to the equation

```math
    xa \\leq b.
```
"""
function Base.:/(b::SemiringNumber, a::SemiringNumber)
    return mul_impl(b, a, Val(:N), Val(:C), Val(:C))
end

function Base.:/(b::AbstractVecOrMat{<:SemiringNumber}, a::AbstractVecOrMat{<:SemiringNumber})
    return mul_impl(b, a, Val(:N), Val(:C), Val(:C))
end

function ⋉(a, b)
    return mul_impl(a, b, Val(:C), Val(:N), Val(:N))
end

function ⋊(b, a)
    return mul_impl(b, a, Val(:N), Val(:C), Val(:N))
end

function Base.:<=(a::SemiringNumber{A, T}, b::SemiringNumber{A, T}) where {A <: AbstractSemiring, T}
    return le_impl(A, parent(a), parent(b))
end

function Base.:<(a::SemiringNumber{A, T}, b::SemiringNumber{A, T}) where {A <: AbstractSemiring, T}
    return lt_impl(A, parent(a), parent(b))
end
