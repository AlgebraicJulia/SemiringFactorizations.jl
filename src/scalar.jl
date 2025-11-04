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

function sldiv(a::T, b::T) where {T <: Union{TropicalAndOr, TropicalBitwise, TropicalMaxMin}}
    return b
end

function sldiv(a::T, b::T) where {T <: Union{TropicalMaxPlus, TropicalMinPlus, TropicalMaxMul}}
    if a <= one(T) || b <= typemin(T)
        c = b
    else
        c = typemax(T)
    end

    return c
end

function sldiv(a::RE, b::RE)
    if iszero(a) || isone(a)
        c = b
    else
        c = RE(nmg(a.str) * "*") * b
    end

    return c
end

function sldiv(A, B::AbstractArray{T, N}) where {T, N}
    return sldiv!(A, Array{T, N}(B))
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

function srdiv(b::RE, a::RE)
    if iszero(a) || isone(a)
        c = b
    else
        c = b * RE(nmg(a.str) * "*")
    end

    return c
end

function srdiv(B::AbstractArray{T, N}, A) where {T, N}
    return srdiv!(Array{T, N}(B), A)
end
