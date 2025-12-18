"""
    AbstractStar{T}

An ``AbstractStar`` object represents a matrix ``A^*``.
"""
abstract type AbstractStar{T} end

"""
    AbstractStarLU{T} <: AbstractStar{T}

An `AbstractStarLU` factorization object represents
a matrix ``A^*`` as a pair ``(L, U)``, where

  - ``L`` is strictly lower triangular
  - ``U`` is upper triangular

and ``A^* = U^* L^*``.
"""
abstract type AbstractStarLU{T} <: AbstractStar{T} end

"""
    AbstractStarTriangular{Q, T} <: AbstractStar{T}

An `AbstractStarTriangular` object lazily represents the
Kleene star ``A^*`` of a triangular matrix ``A``. The type
parameter `Q` indicates whether ``A`` is lower or upper
triangular:

  - `:L`: strictly lower triangular
  - `:U`: upper triangular

"""
abstract type AbstractStarTriangular{Q, T} <: AbstractStar{T} end

function Base.Matrix(A::AbstractStar{T}) where {T}
    B = zeros(T, size(A))
    fill!(view(B, diagind(B)), one(T))
    return rmul!(B, A)
end

function Base.show(io::IO, F::T) where {T <: AbstractStar}
    n = size(F, 1)
    print(io, "$(n)×$(n) $T")
    return
end

function Base.getproperty(F::AbstractStarLU, d::Symbol)
    if d == :L
        p = AbstractStarTriangular{:L}(F)
    elseif d == :U
        p = AbstractStarTriangular{:U}(F)
    else
        p = getfield(F, d)
    end

    return p
end

function Base.size(F::AbstractStarLU)
    return size(AbstractStarTriangular{:L}(F))
end

function Base.size(F::AbstractStar, i::Integer)
    return size(F)[i]
end

function Semirings.star(A::AbstractMatrix)
    return Matrix(slu(A))
end

function Semirings.star(A::StrictLowerTriangular)
    return UnitLowerTriangular(Matrix(slu(A)))
end

function Semirings.star(A::UpperTriangular)
    return UpperTriangular(Matrix(slu(A)))
end

function Base.:*(A::AbstractStar{T}, B::AbstractVecOrMat) where {T}
    return lmul!(A, wrapcopy(T, B))
end

function Base.:*(B::AbstractMatrix, A::AbstractStar{T}) where {T}
    return rmul!(wrapcopy(T, B), A)
end

function Base.:\(A::AbstractStar{T}, B::AbstractVecOrMat) where {T}
    return ldiv!(A, wrapcopy(T, B))
end

function Base.:/(B::AbstractMatrix, A::AbstractStar{T}) where {T}
    return rdiv!(wrapcopy(T, B), A)
end

function lmul!(A::AbstractStarLU, B::AbstractVecOrMat)
    L = AbstractStarTriangular{:L}(A)
    U = AbstractStarTriangular{:U}(A)
    return lmul!(U, lmul!(L, B))
end

function rmul!(B::AbstractMatrix, A::AbstractStarLU)
    L = AbstractStarTriangular{:L}(A)
    U = AbstractStarTriangular{:U}(A)
    return rmul!(rmul!(B, U), L)
end

function LinearAlgebra.ldiv!(A::AbstractStarLU, B::AbstractVecOrMat)
    L = AbstractStarTriangular{:L}(A)
    U = AbstractStarTriangular{:U}(A)
    return ldiv!(L, ldiv!(U, B))
end

function LinearAlgebra.rdiv!(B::AbstractMatrix, A::AbstractStarLU)
    L = AbstractStarTriangular{:L}(A)
    U = AbstractStarTriangular{:U}(A)
    return rdiv!(rdiv!(B, L), U)
end
