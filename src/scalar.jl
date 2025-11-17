# For our purposes, a closed semiring is a sextuple (S, +, ×, 0, 1, *), where
#
#      +: S × S → S
#      ×: S × S → S
#      *: S × S → S
#
# are "addition", "multiplication", and "closure" operations, and
#
#      0 ∈ S
#      1 ∈ S
#
# are "zero" and "one" elements satisfying the following identities.
#
#   - a + (b + c) = (a + b) + c    addition is associative
#   - a × (b × c) = (a × b) × c    multiplication is associative
#   - a + 0 = a                    0 is a unit for addition
#   - a × 1 = 1 × a = a            1 is a unit for multiplication
#   - a + b = b + a                addition is commutative
#   - a × (b + c) = a × b + a × c  multiplication distritributes over addition
#     (b + c) × a = b × a + c × a
#   - a* = 1 + a × a* = 1 + a* × a
#
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

function lres(a, b)
    return a \ b
end

function rres(b, a)
    return b / a
end

# FIXME
function slres(a, b)
    return lres(star(a), b)
end

# FIXME
function srres(b, a)
    return rres(b, star(a))
end

function lresinf(a, b, c)
    return inf(lres(a, b), c)
end

function rresinf(a, b, c)
    return inf(rres(a, b), c)
end

const ⋋ = lres
const ⋌ = rres

"""
    star(a)

Compute the closure of a.
"""
function star(a)
    return srmul(one(a), a)
end

"""
    slmul(a, b)

Compute ``a^* b``.
"""
slmul(a, b)

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

function slmul(a::RegularExpression, b::RegularExpression)
    if iszero(a) || isone(a)
        c = b
    else
        c = RegularExpression(nmg(a.str) * "*") * b
    end

    return c
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

Compute ``b a^*``.
"""
srmul(b, a)

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

function srmul(b::RegularExpression, a::RegularExpression)
    if iszero(a) || isone(a)
        c = b
    else
        c = b * RegularExpression(nmg(a.str) * "*")
    end

    return c
end

"""
    slres(a, b)
"""
slres(a, b)

function slres(a::T, b::T) where {T <: Union{TropicalMaxPlus, TropicalMinPlus, TropicalMaxMul}}
    if a <= one(T) || b >= typemax(T)
        c = b
    else
        c = typemax(T)
    end

    return c
end

function srres(b::T, a::T) where {T <: CommutativeSemiring}
    return slres(a, b)
end
