"""
    AbstractStarLU{T}

An `AbstractStarLU` factorization object represents
a matrix ``A`` as a pair ``(L, U)``, where

  - ``L`` is strictly lower triangular
  - ``U`` is upper triangular

and ``A = U^* L^*``.
"""
abstract type AbstractStarLU{T} end

function Base.Matrix(A::AbstractStarLU{T}) where {T}
    B = zeros(T, size(A))
    fill!(view(B, diagind(B)), one(T))
    return rmul!(B, A)
end

"""
    slu(A::AbstractMatrix)

Construct an [`AbstractStarLU`](@ref) factorization
object for the Kleene star ``A^*``.
"""
slu(A::AbstractMatrix)

"""
    lmul!(A::AbstractStarLU, B::AbstractVecOrMat)

Compute the product AB, storing the result in B.
"""
lmul!(A::AbstractStarLU, B::AbstractVecOrMat)

"""
    lmul!(B::AbstractMatrix, A::AbstractStarLU)

Compute the product BA, storing the result in B.
"""
rmul!(B::AbstractMatrix, A::AbstractStarLU)

"""
    slmul!(A::AbstractMatrix, B::AbstractVecOrMat)

Compute the product A*B, storing the result in B.
"""
function slmul!(A::AbstractMatrix, B::AbstractVecOrMat)
    return lmul!(slu(A), B)
end

"""
    srmul!(B::AbstractMatrix, A::AbstractMatrix)

Compute the product BA*, storing the result in B.
"""
function srmul!(B::AbstractMatrix, A::AbstractMatrix)
    return rmul!(B, slu(A))
end

"""
    sldiv!(A::AbstractMatrix, B::AbstractVecOrMat)

Compute the residuum A* \\ B, storing the result in B.
"""
function sldiv!(A::AbstractMatrix, B::AbstractVecOrMat)
    return ldiv!(slu(A), B)
end

"""
    srdiv!(B::AbstractMatrix, A::AbstractMatrix)

Compute the residuum B / A*, storing the result in B.
"""
function srdiv!(B::AbstractMatrix, A::AbstractMatrix)
    return rdiv!(B, slu(A))
end

"""
    fma!(A::AbstractMatrix, B::AbstractVecOrMat, C::AbstractVecOrMat)

Compute the sum AB + C, storing the result in C.
"""
function fma!(A::AbstractMatrix, B::AbstractVecOrMat, C::AbstractVecOrMat)
    mul_impl!(C, A, B, Val(:N), Val(:N))
    return C
end

"""
    fli!(A::AbstractMatrix, B::AbstractVecOrMat, C::AbstractVecOrMat)

Compute the infimum A \\ B ∧ C, storing the result in C.
"""
function fli!(A::AbstractMatrix, B::AbstractVecOrMat, C::AbstractVecOrMat)
    mul_impl!(C, A, B, Val(:C), Val(:N))
    return C
end

"""
    fri!(B::AbstractMatrix, A::AbstractMatrix, C::AbstractMatrix)

Compute the infimum B / A ∧ C, storing the result in C.
"""
function fri!(B::AbstractMatrix, A::AbstractMatrix, C::AbstractMatrix)
    mul_impl!(C, B, A, Val(:N), Val(:C))
    return C
end

function fri!(B::AbstractRowVector, A::AbstractMatrix, C::AbstractRowVector)
    mul_impl!(parent(C), parent(B), A, Val(:N), Val(:C))
    return C
end

function Semirings.star(A::AbstractMatrix)
    return Matrix(slu(A))
end

function Base.:*(A::AbstractStarLU{T}, B::AbstractVecOrMat) where {T}
    return lmul!(A, wrapcopy(T, B))
end

function Base.:*(B::AbstractMatrix, A::AbstractStarLU{T}) where {T}
    return rmul!(wrapcopy(T, B), A)
end

function Base.:\(A::AbstractStarLU{T}, B::AbstractVecOrMat) where {T}
    return ldiv!(A, wrapcopy(T, B))
end

function Base.:\(A::AbstractMatrix{T}, B::AbstractMatrix{T}) where {T <: SemiringNumber}
    return fli!(A, B, fill(typemax(T), size(A, 2), size(B, 2)))
end

function Base.:\(A::SparseMatrixCSC{T}, B::AbstractMatrix{T}) where {T <: SemiringNumber}
    return fli!(A, B, fill(typemax(T), size(A, 2), size(B, 2)))
end

function Base.:\(A::AbstractMatrix{T}, B::AbstractVector{T}) where {T <: SemiringNumber}
    return fli!(A, B, fill(typemax(T), size(A, 2)))
end

function Base.:\(A::SparseMatrixCSC{T}, B::AbstractVector{T}) where {T <: SemiringNumber}
    return fli!(A, B, fill(typemax(T), size(A, 2)))
end

function Base.:\(A::AbstractRowVector{T}, B::AbstractVector{T}) where {T <: SemiringNumber}
    return fli(parent(A), B, typemax(T))
end

function Base.:/(B::AbstractMatrix, A::AbstractStarLU{T}) where {T}
    return rdiv!(wrapcopy(T, B), A)
end

function Base.:/(B::AbstractMatrix{T}, A::AbstractMatrix{T}) where {T <: SemiringNumber}
    return fri!(B, A, fill(typemax(T), size(B, 1), size(A, 1)))
end

function Base.:/(B::AbstractRowVector{T}, A::AbstractMatrix{T}) where {T <: SemiringNumber}
    return fri!(B, A, transpose(fill(typemax(T), size(A, 1))))
end

function Base.:/(B::AbstractRowVector{T}, A::AbstractVector{T}) where {T <: SemiringNumber}
    return fri(parent(B), A, typemax(T))
end

function Semirings.slmul(A::AbstractMatrix{T}, B::AbstractVecOrMat) where {T}
    return slmul!(A, wrapcopy(T, B))
end

function Semirings.srmul(B::AbstractMatrix, A::AbstractMatrix{T}) where {T}
    return srmul!(wrapcopy(T, B), A)
end

function Semirings.sldiv(A::AbstractMatrix{T}, B::AbstractVecOrMat) where {T}
    return sldiv!(A, wrapcopy(T, B))
end

function Semirings.srdiv(B::AbstractMatrix, A::AbstractMatrix{T}) where {T}
    return srdiv!(wrapcopy(T, B), A)
end

function Semirings.fma(A::AbstractMatrix, B::AbstractMatrix, C::AbstractMatrix)
    return fma!(A, B, Matrix(C))
end 

function Semirings.fli(A::AbstractMatrix, B::AbstractMatrix, C::AbstractMatrix)
    return fli!(A, B, Matrix(C))
end 

function Semirings.fri(B::AbstractMatrix, A::AbstractMatrix, C::AbstractMatrix)
    return fri!(B, A, Matrix(C))
end 

# --------------------- #
# Matrix Multiplication #
# --------------------- #

function mul_impl!(C::AbstractMatrix, A::AbstractMatrix, B::AbstractMatrix, tA::Val{R}, tB::Val{:N}) where {R}
    @assert size(C, 2) == size(B, 2)

    if R == :N
        @assert size(C, 1) == size(A, 1)
        @assert size(B, 1) == size(A, 2)
    else
        @assert size(C, 1) == size(A, 2)
        @assert size(B, 1) == size(A, 1)
    end

    @inbounds for j in axes(B, 2)
        Cj = @view C[:, j]
        Bj = @view B[:, j]
        #
        #   Cj ← Cj + A Bj
        #
        mul_impl!(Cj, A, Bj, tA, tB)
    end

    return
end

function mul_impl!(C::AbstractMatrix, A::AbstractMatrix, B::AbstractMatrix, tA::Val{:N}, tB::Val{:C})
    @assert size(C, 1) == size(A, 1)
    @assert size(C, 2) == size(B, 1)
    @assert size(B, 2) == size(A, 2)

    @inbounds for j in axes(A, 2)
        Aj = @view A[:, j]
        Bj = @view B[:, j]
        #
        #   C ← C ∧ Aj / Bj
        #
        mul_impl!(C, Aj, Bj, tA, tB)
    end

    return
end

function mul_impl!(C::AbstractVector, A::AbstractMatrix, B::AbstractVector, tA::Val{:C}, tB::Val{:N})
    @assert length(C) == size(A, 2)
    @assert length(B) == size(A, 1)

    @inbounds for i in eachindex(C)
        Ci = @view C[   i]
        Ai = @view A[:, i]
        #
        #   Ci ← Ci ∧ Ai \ B
        #
        mul_impl!(Ci, Ai, B, tA, tB)
    end

    return
end

function mul_impl!(C::AbstractVector, A::AbstractVector, B::AbstractMatrix, tA::Val{:N}, tB::Val{:N})
    @assert length(A) == size(B, 1)
    @assert length(C) == size(B, 2)

    @inbounds for j in axes(B, 2)
        Cj = @view C[   j]
        Bj = @view B[:, j]
        #
        #   Cj ← Cj + A Bj
        #
        mul_impl!(Cj, A, Bj, tA, tB)
    end

    return
end

function mul_impl!(C::AbstractScalar, A::AbstractVector, B::AbstractVector, tA::Val{:N}, tB::Val{:N})
    @assert length(A) == length(B)
    #
    #   C ← C + AB
    #
    C[] = fma(A, B, C[])
    return
end

function mul_impl!(C::AbstractScalar, A::AbstractVector, B::AbstractVector, tA::Val{:C}, tB::Val{:N})
    @assert length(A) == length(B)
    #
    #   C ← C ∧ A \ B
    #
    C[] = fli(A, B, C[])
    return
end

function mul_impl!(C::AbstractScalar, A::AbstractVector, B::AbstractVector, tA::Val{:N}, tB::Val{:C})
    @assert length(A) == length(B)
    #
    #   C ← C ∧ A / B
    #
    C[] = fri(A, B, C[])
    return
end

function mul_impl!(C::AbstractScalar{T}, A::T, B::T, tA::Val{:C}, tB::Val{:N}) where {T}
    #
    #   C ← C ∧ A \ B
    #
    C[] = fli(A, B, C[])
    return
end

function mul_impl!(C::AbstractScalar{T}, A::T, B::T, tA::Val{:N}, tB::Val{:C}) where {T}
    #
    #   C ← C ∧ A / B
    #
    C[] = fri(A, B, C[])
    return
end

function mul_impl!(C::AbstractScalar{T}, A::T, B::T, tA::Val{:N}, tB::Val{:N}) where {T}
    #
    #   C ← C + A B
    #
    C[] = fma(A, B, C[])
    return
end

# ---------------- #
# Jacobi Algorithm #
# ---------------- #

function jacobi(A::AbstractMatrix{T}, B::AbstractVecOrMat{T}, tA::Val, side::Val{S}; kw...) where {T, S}
    C = Array{T}(undef, size(B))
    D = Array{T}(undef, size(B))
    copyto!(C, B)

    for _ in 1:size(A, 1) + 1
        copyto!(D, B)

        if S == :L
            #
            #   D ← D + A C
            #
            mul_impl!(D, A, C, tA, Val(:N))
        else
            #
            #   D ← D + C A
            #
            mul_impl!(D, C, A, Val(:N), tA)
        end

        if isapprox(C, D; kw...)
            return D
        end

        copyto!(C, D)
    end

    error()
end
