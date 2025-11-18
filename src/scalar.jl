const CommutativeSemiring = Union{
    AbstractFloat,
    Complex,
    Rational,
    Integer,
    TropicalAndOr,
    TropicalBitwise,
    TropicalMaxPlus,
    TropicalMinPlus,
    TropicalMaxMul,
    TropicalMaxMin,
    TopN,
}

"""
    lres(a, b)

Compute the residual a \\ b.
"""
function lres(a, b)
    return a \ b
end

"""
    rres(b, a)

Compute the residual b / a.
"""
function rres(b, a)
    return b / a
end

function lresinf(a, b, c)
    return inf(lres(a, b), c)
end

function rresinf(a, b, c)
    return inf(rres(a, b), c)
end

const ⋋ = lres
const ⋌ = rres
const ∧ = inf

"""
    star(a)

Compute the Kleene star a*.
"""
function star(a)
    return srmul(one(a), a)
end

function star(a::RE)
    if a.head == :zero || a.head == :one
        c = one(RE)
    elseif a.head == :top || a.head == :star
        c = a
    else
        c = RE(:star, true, a)
    end

    return c
end

"""
    slmul(a, b)

Compute the product a* b.
"""
function slmul(a, b)
    return star(a) * b
end

function slmul(::Number, ::Missing)
    return missing
end

function slmul(::Missing, ::Number)
    return missing
end

function slmul(::Missing, ::Missing)
    return missing
end

function slmul(a::T, b::T) where {T <: Union{AbstractFloat, Complex}}
    return b / (one(T) - a)
end

function slmul(a::T, b::T) where {T <: Rational}
    if !isone(a) && !isinf(a)
        c = b // (one(T) - a)
    else
        c = typemax(T)
    end

    return c
end

function slmul(a::T, b::T) where {T <: Integer}
    if !ispositive(a) || !ispositive(b)
        c = b
    else
        c = posinf(T)
    end

    return c
end

function slmul(a::T, b::T) where {T <: Union{TropicalAndOr, TropicalBitwise, TropicalMaxMin}}
    return b
end

function slmul(a::T, b::T) where {T <: Union{TropicalMaxPlus, TropicalMinPlus, TropicalMaxMul}}
    if a <= one(T) || b <= typemin(T)
        c = b
    else
        c = typemax(T)
    end

    return c
end

function slmul(a::RE, b::RE)
    return star(a) * b
end


function slmul(a::TopN{N, T}, b::TopN{N, T}) where {N, T}
    if !any(>(typemin(T)), b)
        c = b
    elseif any(>(one(T)), a)
        c = typemax(T)
    else
        tup = ntuple(N) do i
            if isone(i)
                v = one(T)
            else
                v = a[i - 1]
            end

            return v
        end

        return TopN{N, T}(tup) * b
    end
end

"""
    srmul(b, a)

Compute the product b a*.
"""
function srmul(b, a)
    return b * star(a)
end

function srmul(::Number, ::Missing)
    return missing
end

function srmul(::Missing, ::Number)
    return missing
end

function srmul(::Missing, ::Missing)
    return missing
end

function srmul(b::T, a::T) where {T <: CommutativeSemiring}
    return slmul(a, b)
end

"""
    slres(a, b)

Compute the residual a* \\ b
"""
function slres(a, b)
    return lres(star(a), b)
end

function slres(a::T, b::T) where {T <: Union{TropicalAndOr, TropicalBitwise, TropicalMaxMin}}
    return b
end

function slres(a::T, b::T) where {T <: Union{TropicalMaxPlus, TropicalMinPlus, TropicalMaxMul}}
    if a <= one(T) || b >= typemax(T)
        c = b
    else
        c = typemax(T)
    end

    return c
end

"""
    srres(b, a)

Compute the residual b / a*.
"""
function srres(b, a)
    return rres(b, star(a))
end

function srres(b::T, a::T) where {T <: CommutativeSemiring}
    return slres(a, b)
end
