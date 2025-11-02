const CommutativeSemiring = Union{
    AbstractFloat,
    Integer,
    TropicalAndOr,
    TropicalBitwise,
    TropicalMaxPlus,
    TropicalMinPlus,
    TropicalMaxMul,
    TropicalMaxMin,
}

"""
    sinv(a)

Compute a quasi-inverse of a, i.e. an
object a* satisfying

```math
    a^* = 1 + a a^* = 1 + a^* a.
```

"""
function sinv(a)
    return srdiv(one(a), a)
end

function sinv(A::Union{AbstractMatrix{T}, AbstractSemiringLU{T}}) where {T}
    B = zeros(T, size(A))

    @inbounds for i in diagind(B)
        B[i] = one(T)
    end

    return srdiv!(B, A)
end

"""
    sldiv(a, b)

Solve the linear fixed-point equation

```math
    ax + b = x,
```

where a, b, and x are elements of a
semiring.
"""
sldiv(a, b)

function sldiv(a::T, b::T) where {T <: AbstractFloat}
    if !isone(a) || iszero(b)
        c = b / (one(T) - a)
    else
        c = posinf(T)
    end

    return c
end

function sldiv(a::T, b::T) where {T <: Integer}
    if !ispositive(a) || !ispositive(b)
        c = b
    else
        c = sign(b) * posinf(T)
    end

    return c
end

function sldiv(a::TropicalAndOr, b::TropicalAndOr)
    return b
end

function sldiv(a::TropicalBitwise, b::TropicalBitwise)
    return b
end

function sldiv(a::TropicalMaxPlus{T}, b::TropicalMaxPlus{T}) where {T}
    if !ispositive(a.n) || b.n <= neginf(T)
        n = b.n
    else
        n = posinf(T)
    end

    return TropicalMaxPlus(n)
end

function sldiv(a::TropicalMinPlus{T}, b::TropicalMinPlus{T}) where {T}
    if !isnegative(a.n) || b.n >= posinf(T)
        n = b.n
    else
        n = neginf(T)
    end

    return TropicalMinPlus(n)
end

function sldiv(a::TropicalMaxMul{T}, b::TropicalMaxMul{T}) where {T}
    if a.n <= one(T) || !ispositive(b.n)
        n = b.n
    else
        n = posinf(T)
    end

    return TropicalMaxMul(n)
end

function sldiv(a::TropicalMaxMin{T}, b::TropicalMaxMin{T}) where {T}
    return b
end

function sldiv(A, B::AbstractMatrix)
    return sldiv!(A, Matrix(B))
end

function sldiv(A, B::AbstractVector)
    return sldiv!(A, Vector(B))
end

"""
    srdiv(b, a)

Solve the linear fixed-point equation

```math
    xa + b = x,
```

where a, b, and x are elements of a
semiring.
"""
srdiv(b, a)

function srdiv(b::T, a::T) where {T <: CommutativeSemiring}
    return sldiv(a, b)
end

function srdiv(B::AbstractMatrix, A)
    return srdiv!(Matrix(B), A)
end

function srdiv(B::AbstractVector, A)
    return srdiv!(Vector(B), A)
end
