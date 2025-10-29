"""
    sinv(a)

Compute a quasi-inverse of a, i.e. an
object a* satisfying

    a* = 1 + a a*
       = 1 + a* a

"""
sinv(a)

function sinv(a::TropicalAndOr)
    return TropicalAndOr(true)
end

function sinv(a::T) where {T <: AbstractFloat}
    if isone(a)
        n = posinf(T)
    else
        n = inv(one(T) - a)
    end

    return n
end

function sinv(a::TropicalMaxMul{T}) where {T}
    if a.n > one(T)
        n = posinf(T)
    else
        n = one(T)
    end

    return TropicalMaxMul(n)
end

function sinv(a::TropicalMaxPlus{T}) where {T}
    if a.n > zero(T)
        n = posinf(T)
    else
        n = zero(T)
    end

    return TropicalMaxPlus(n)
end
    
function sinv(a::TropicalMinPlus{T}) where {T}
    if a.n < zero(T)
        n = neginf(T)
    else
        n = zero(T)
    end

    return TropicalMinPlus(n)
end
