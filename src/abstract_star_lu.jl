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

struct AdjointStar{T, S <: AbstractStar{T}} <: AbstractStar{T}
    star::S
end

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

function Base.size(F::AdjointStar)
    return size(parent(F))
end

function Base.size(F::AbstractStar, i::Integer)
    return size(F)[i]
end

function Base.parent(F::AdjointStar)
    return F.star
end

function Base.adjoint(F::AbstractStar)
    return AdjointStar(F)
end

function Base.adjoint(F::AdjointStar)
    return parent(F)
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
    return lmul!(A, Array(B))
end

function Base.:*(B::AbstractMatrix, A::AbstractStar{T}) where {T}
    return rmul!(Array(B), A)
end

function Base.:*(B::AbstractRowVector, A::AbstractStar{T}) where {T}
    return transpose(rmul!(Array(transpose(B)), A))
end

function Base.:*(A::AbstractStar, B::Adjoint)
    return (B' / A)'
end

function Base.:*(B::Adjoint, A::AbstractStar)
    return (A \ B')'
end

function Semirings.:⅋(A::AdjointStar, B::AbstractVecOrMat)
    return A' \ B
end

function Semirings.:⅋(B::AbstractMatrix, A::AdjointStar)
    return B / A'
end

function Base.:\(A::AbstractStar{T}, B::AbstractVecOrMat) where {T}
    return ldiv!(A, Array(B))
end

function Base.:/(B::AbstractMatrix, A::AbstractStar{T}) where {T}
    return rdiv!(Array(B), A)
end

function Base.:/(B::AbstractRowVector, A::AbstractStar{T}) where {T}
    return transpose(rdiv!(Array(transpose(B)), A))
end

function LinearAlgebra.lmul!(A::AbstractStarLU, B::AbstractVecOrMat)
    L = AbstractStarTriangular{:L}(A)
    U = AbstractStarTriangular{:U}(A)
    return lmul!(U, lmul!(L, B))
end

function LinearAlgebra.rmul!(B::AbstractVecOrMat, A::AbstractStarLU)
    L = AbstractStarTriangular{:L}(A)
    U = AbstractStarTriangular{:U}(A)
    return rmul!(rmul!(B, U), L)
end

function LinearAlgebra.ldiv!(A::AbstractStarLU, B::AbstractVecOrMat)
    L = AbstractStarTriangular{:L}(A)
    U = AbstractStarTriangular{:U}(A)
    return ldiv!(L, ldiv!(U, B))
end

function LinearAlgebra.rdiv!(B::AbstractVecOrMat, A::AbstractStarLU)
    L = AbstractStarTriangular{:L}(A)
    U = AbstractStarTriangular{:U}(A)
    return rdiv!(rdiv!(B, L), U)
end
