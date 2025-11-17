abstract type AbstractStarLU{T} end

"""
    size(F::AbstractStarLU)

Get the size of a factorized matrix.
"""
Base.size(F::AbstractStarLU)

"""
    slu!(A::AbstractMatrix)

Compute an LU factorization of a semiring-
valued matrix A, over-writing A with the factors.
"""
slu!(A::AbstractMatrix)

"""
    slu(A::AbstractMatrix)

Compute an LU factorization of a semiring-
valued matrix A.
"""
function slu(A::AbstractMatrix)
    return slu!(FMatrix(A))
end

"""
"""
function star(A::Union{AbstractMatrix{T}, AbstractStarLU{T}}) where {T}
    B = zeros(T, size(A))
    fill!(view(B, diagind(B)), one(T))
    return srmul!(B, A)
end

"""
"""
function slmul(A, B::AbstractVecOrMat)
    return slmul!(A, Array(B))
end

"""
    slmul!(A, B::AbstractVecOrMat)

Solve the linear fixed-point equation

```math
    AX + B = X,
```

over-writing B with the solution.
"""
function slmul!(A::AbstractMatrix, B::AbstractVecOrMat)
    return slmul!(slu(A), B)
end

"""
"""
function srmul(B::AbstractVecOrMat, A)
    return srmul!(Array(B), A)
end

"""
    srmul!(B::AbstractVecOrMat, A)

Solve the linear fixed-point equation

```math
    XA + B = X,
```

over-writing B with the solution.
"""
function srmul!(B::AbstractVecOrMat, A::AbstractMatrix)
    return srmul!(B, slu(A))
end

function slres(A, B::AbstractVecOrMat)
    return slres!(A, Array(B))
end

function srres(B::AbstractVecOrMat, A)
    return srres!(Array(B), A)
end

function Base.:*(A::AbstractStarLU, B::AbstractVecOrMat)
    return slmul(A, B)
end

function Base.:*(B::AbstractVecOrMat, A::AbstractStarLU)
    return srmul(B, A)
end

function mul(A::AbstractStarLU, B::AbstractVecOrMat)
    return slmul(A, B)
end

function mul(B::AbstractVecOrMat, A::AbstractStarLU)
    return srmul(B, A)
end

function mul(A::AbstractMatrix{T}, B::AbstractMatrix{T}) where {T}
    C = fill(zero(T), size(A, 1), size(B, 2))
    mul_impl!(C, A, B, Val(:N), Val(:N))
    return C
end

function mul(A::AbstractMatrix{T}, B::AbstractVector{T}) where {T}
    C = fill(zero(T), size(A, 1))
    mul_impl!(C, A, B, Val(:N), Val(:N))
    return C
end

function lres(A::AbstractStarLU, B::AbstractVecOrMat)
    return slres(A, B)
end

function lres(A::AbstractMatrix{T}, B::AbstractMatrix{T}) where {T}
    C = fill(typemax(T), size(A, 2), size(B, 2))
    mul_impl!(C, A, B, Val(:C), Val(:N))
    return C
end

function lres(A::AbstractMatrix{T}, B::AbstractVector{T}) where {T}
    C = fill(typemax(T), size(A, 2))
    mul_impl!(C, A, B, Val(:C), Val(:N))
    return C
end

function rres(B::AbstractVecOrMat, A::AbstractStarLU)
    return srres(B, A)
end

function rres(B::AbstractMatrix{T}, A::AbstractMatrix{T}) where {T}
    C = fill(typemax(T), size(B, 1), size(A, 1))
    mul_impl!(C, B, A, Val(:N), Val(:C))
    return C
end

function rres(B::AbstractVector{T}, A::AbstractMatrix{T}) where {T}
    C = fill(typemax(T), size(A, 1))
    mul_impl!(C, B, A, Val(:N), Val(:C))
    return C
end

function mul_impl!(C::AbstractMatrix, A::AbstractMatrix, B::AbstractMatrix, tA::Val{TA}, tB::Val{:N}) where {TA}
    @assert size(C, 2) == size(B, 2)

    if TA == :N
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
    #   C ← C + A B
    #
    C[] += vmuladd(A, B)
    return
end

function mul_impl!(C::AbstractScalar, A::AbstractVector, B::AbstractVector, tA::Val{:C}, tB::Val{:N})
    @assert length(A) == length(B)
    #
    #   C ← C ∧ A \ B
    #
    C[] = inf(C[], vlresinf(A, B))
    return
end

function mul_impl!(C::AbstractScalar, A::AbstractVector, B::AbstractVector, tA::Val{:N}, tB::Val{:C})
    @assert length(A) == length(B)
    #
    #   C ← C ∧ A / B
    #
    C[] = inf(C[], vrresinf(A, B))
    return
end

function mul_impl!(C::AbstractScalar{T}, A::T, B::T, tA::Val{:C}, tB::Val{:N}) where {T}
    #
    #   C ← C ∧ A \ B
    #
    C[] = lresinf(A, B, C[])
    return
end

function mul_impl!(C::AbstractScalar{T}, A::T, B::T, tA::Val{:N}, tB::Val{:C}) where {T}
    #
    #   C ← C ∧ A / B
    #
    C[] = rresinf(A, B, C[])
    return
end

function mul_impl!(C::AbstractScalar{T}, A::T, B::T, tA::Val{:N}, tB::Val{:N}) where {T}
    #
    #   C ← C + A B
    #
    C[] = muladd(A, B, C[])
    return
end

# ---------------- #
# Jacobi Algorithm #
# ---------------- #

function jacobi(A::AbstractMatrix{T}, B::AbstractVecOrMat{T}, side::Val{S}; kw...) where {T, S}
    C = Array{T}(undef, size(B))
    D = Array{T}(undef, size(B))
    copyto!(C, B)

    for _ in 1:size(A, 1) + 1
        copyto!(D, B)

        if S == :L
            #
            #   D ← D + A C
            #
            mul_impl!(D, A, C, Val(:N), Val(:N))
        else
            #
            #   D ← D + C A
            #
            mul_impl!(D, C, A, Val(:N), Val(:N))
        end

        if isapprox(C, D; kw...)
            return D
        end

        copyto!(C, D)
    end

    error()
end
