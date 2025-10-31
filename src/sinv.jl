"""
    sinv(a)

Compute a quasi-inverse of a, i.e. an
object a* satisfying

    a* = 1 + a a*
       = 1 + a* a

"""
sinv(a)

function sinv(a::T) where {T <: AbstractFloat}
    if isone(a)
        b = posinf(T)
    else
        b = inv(one(T) - a)
    end

    return b
end

function sinv(a::T) where {T <: Integer}
    if iszero(a)
        b = one(T)
    else
        b = posinf(T)
    end

    return b
end

function sinv(a::TropicalAndOr)
    return TropicalAndOr(true)
end

function sinv(a::TropicalMaxPlus{T}) where {T}
    if ispositive(a.n)
        n = posinf(T)
    else
        n = zero(T)
    end

    return TropicalMaxPlus(n)
end
    
function sinv(a::TropicalMinPlus{T}) where {T}
    if isnegative(a.n)
        n = neginf(T)
    else
        n = zero(T)
    end

    return TropicalMinPlus(n)
end

function sinv(a::TropicalMaxMul{T}) where {T}
    if a.n > one(T)
        n = posinf(T)
    else
        n = one(T)
    end

    return TropicalMaxMul(n)
end

function sinv(::TropicalMaxMin{T}) where {T}
    n = posinf(T)
    return TropicalMaxMin(n)
end

function sinv(A::Union{AbstractMatrix{T}, AbstractSemiringLU{T}}) where {T}
    B = zeros(T, size(A))

    @inbounds for i in diagind(B)
        B[i] = one(T)
    end

    return srdiv!(B, A)
end
