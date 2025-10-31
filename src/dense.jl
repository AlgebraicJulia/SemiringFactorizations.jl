"""
    SemiringLU{T, M <: AbstractMatrix{T}} <: AbstractSemiringLU{T}

An LU factorization of a semiring-valued
matrix.
"""
struct SemiringLU{T, M <: AbstractMatrix{T}} <: AbstractSemiringLU{T}
    factors::M
end

function Base.getproperty(F::SemiringLU, d::Symbol)
    if d == :L
        p = StrictLowerTriangular(F.factors)
    elseif d == :U
        p = UpperTriangular(F.factors)
    else
        p = getfield(F, d)
    end

    return p
end

function Base.show(io::IO, mime::MIME"text/plain", F::T) where {T <: SemiringLU}
    print(io, "$T:")
    print(io, "\nL factor:\n")
    show(io, mime, F.L)
    print(io, "\nU factor:\n")
    show(io, mime, F.U)
    return
end

# ------------------------------ #
# Abstract Semiring LU Interface #
# ------------------------------ #

function Base.size(F::SemiringLU)
    return size(F.factors)
end

function slu!(A::AbstractMatrix)
    sgetrf!(A)
    return SemiringLU(A)
end

function sldiv!(A::Number, B::AbstractVecOrMat)
    strsm!(A, B, Val(:U), Val(:L))
    return B
end

function sldiv!(A::StrictLowerTriangular, B::AbstractVecOrMat)
    strsm!(parent(A), B, Val(:L), Val(:L))
    return B
end

function sldiv!(A::UpperTriangular, B::AbstractVecOrMat)
    strsm!(parent(A), B, Val(:U), Val(:L))
    return B
end

function sldiv!(A::SemiringLU, B::AbstractVecOrMat)
    return sldiv!(A.U, sldiv!(A.L, B))
end

function srdiv!(B::AbstractVecOrMat, A::Number)
    strsm!(A, B, Val(:U), Val(:R))
    return B
end

function srdiv!(B::AbstractVecOrMat, A::StrictLowerTriangular)
    strsm!(parent(A), B, Val(:L), Val(:R))
    return B
end

function srdiv!(B::AbstractVecOrMat, A::UpperTriangular)
    strsm!(parent(A), B, Val(:U), Val(:R))
    return B
end

function srdiv!(B::AbstractVecOrMat, A::SemiringLU)
    return srdiv!(srdiv!(B, A.U), A.L)
end

# ------------------------ #
# Low-Level Matrix Kernels #
# ------------------------ #

function sgetrf2!(A::AbstractMatrix{T}) where {T}
    @assert size(A, 1) == size(A, 2)

    n = size(A, 1)
    
    @inbounds for i in 1:n
        #
        #   A = [ Aii Ain ]
        #       [ Ani Ann ]
        #
        Aii =       A[i,       i]
        Ani = @view A[i + 1:n, i]
        #
        #   Ani ← Ani Aii*
        #
        strsm2!(Aii, Ani, Val(:U), Val(:R)) 
        #
        #   Ann ← Ann + Ani Ain
        #
        for j in i + 1:n
            Aij =       A[i,       j]
            Anj = @view A[i + 1:n, j]
            #
            #   Anj ← Anj + Ani Aij
            #
            sgemm!(Anj, Ani, Aij)
        end
    end

    return
end

function sgetrf!(A::AbstractMatrix{T}, blocksize::Int = DEFAULT_BLOCK_SIZE) where {T}
    @assert size(A, 1) == size(A, 2)

    n = size(A, 1)

    @inbounds for strt in 1:blocksize:n
        size = min(blocksize, n - strt + 1)
        stop = strt + size - 1
        #
        #   A = [ Abb Abn ]
        #       [ Anb Ann ]
        #
        Abb = @view A[strt:stop,  strt:stop]
        Abn = @view A[strt:stop,  stop + 1:n]
        Anb = @view A[stop + 1:n, strt:stop]
        Ann = @view A[stop + 1:n, stop + 1:n]
        #
        #   Abb ← Lbb + Ubb
        # 
        sgetrf2!(Abb)
        # 
        #   Abn ← Lbb* Abn
        #    
        strsm!(Abb, Abn, Val(:L), Val(:L))
        #
        #   Anb ← Anb Ubb*
        #
        strsm!(Abb, Anb, Val(:U), Val(:R))
        #
        #   Ann ← Ann + Anb Abn
        #
        sgemm!(Ann, Anb, Abn)
    end

    return
end

function strsm2!(A::AbstractMatrix{T}, B::AbstractVecOrMat{T}, uplo::Val{:L}, side::Val{:L}) where {T}
    @assert size(A, 1) == size(A, 2) == size(B, 1)

    n = size(A, 1)
    m = size(B, 2)

    @inbounds for j in 1:m, i in 1:n
        #
        #   A = [ 0   0   ]
        #       [ Ani Ann ]
        #
        Ani = @view A[i + 1:n, i]
        #
        #   B = [ Bi ]
        #       [ Bn ]
        #
        Bi =       B[i,       j]
        Bn = @view B[i + 1:n, j]
        #
        #   Bn ← Bn + Ani Bi
        #
        sgemm!(Bn, Ani, Bi)
    end

    return
end

function strsm!(A::AbstractMatrix{T}, B::AbstractVecOrMat{T}, uplo::Val{:L}, side::Val{:L}, blocksize = DEFAULT_BLOCK_SIZE) where {T}
    @assert size(A, 1) == size(A, 2) == size(B, 1)

    n = size(A, 1)

    @inbounds for strt in 1:blocksize:n
        size = min(blocksize, n - strt + 1)
        stop = strt + size - 1
        #
        #   A = [ Abb 0   ]
        #       [ Anb Ann ]
        #
        Abb = @view A[strt:stop,  strt:stop]
        Anb = @view A[stop + 1:n, strt:stop]
        #
        #   B = [ Bb ]
        #       [ Bn ]
        #
        if B isa AbstractVector
            Bb = @view B[strt:stop]
            Bn = @view B[stop + 1:n]
        else
            Bb = @view B[strt:stop,  :]
            Bn = @view B[stop + 1:n, :]
        end
        #
        #   Bb ← Abb* Bb
        #
        strsm2!(Abb, Bb, uplo, side)
        #
        #   Bn ← Bn + Anb Bb
        #
        sgemm!(Bn, Anb, Bb)
    end

    return
end

function strsm2!(A::T, B::AbstractVector{T}, uplo::Val{:U}, side::Val{:L}) where {T}
    n = length(B); invA = sinv(A)
    #
    #   B ← A* B
    #
    if iszero(invA)
        @inbounds @simd for i in 1:n
            B[i] = zero(T)
        end
    else
        @inbounds @simd for i in 1:n
            B[i] = invA * B[i]
        end
    end

    return
end

function strsm2!(A::AbstractMatrix{T}, B::AbstractVecOrMat{T}, uplo::Val{:U}, side::Val{:L}) where {T}
    @assert size(A, 1) == size(A, 2) == size(B, 1)

    n = size(A, 1)
    m = size(B, 2)

    @inbounds for j in 1:m, i in n:-1:1
        #
        #   A = [ Ann Ani ]
        #       [ 0   Aii ]
        #
        Ani = @view A[1:i - 1, i]
        Aii =       A[i,       i]
        #
        #   B = [ Bn ]
        #       [ Bi ]
        #
        Bn = @view B[1:i - 1, j]
        Bi =       B[i,       j]
        #
        #   Bi ← Aii* Bi
        #
        Bi = B[i, j] = sinv(Aii) * Bi
        #
        #   Bn ← Bn + Ani Bi
        #
        sgemm!(Bn, Ani, Bi)
    end

    return
end

function strsm!(A::AbstractMatrix{T}, B::AbstractVecOrMat{T}, uplo::Val{:U}, side::Val{:L}, blocksize::Int = DEFAULT_BLOCK_SIZE) where {T}
    @assert size(A, 1) == size(A, 2) == size(B, 1)

    n = size(A, 1)

    @inbounds for stop in n:-blocksize:1
        size = min(blocksize, stop)
        strt = stop - size + 1
        #
        #   A = [ Ann Anb ]
        #       [ 0   Abb ]
        #
        Abb = @view A[strt:stop,  strt:stop]
        Anb = @view A[1:strt - 1, strt:stop]
        #
        #   B = [ Bn ]
        #       [ Bb ]
        #
        if B isa AbstractVector
            Bb = @view B[strt:stop]
            Bn = @view B[1:strt - 1]
        else
            Bb = @view B[strt:stop,  :]
            Bn = @view B[1:strt - 1, :]
        end
        #
        #   Bb ← Abb* Bbb
        #
        strsm2!(Abb, Bb, uplo, side)
        #
        #   Bn ← Bn + Anb Bb
        #
        sgemm!(Bn, Anb, Bb)
    end

    return
end

function strsm2!(A::AbstractMatrix{T}, B::AbstractVector{T}, uplo::Val{:L}, side::Val{:R}) where {T}
    @assert size(A, 1) == size(A, 2) == length(B)

    n = size(A, 1)

    @inbounds for j in n:-1:1
        #
        #   A = [ 0   0   ]
        #       [ Anj Ann ]
        #
        #   B = [ Bj  Bn  ]
        #
        BnAnj = zero(T)
        #
        #   BnAnj ← Bn Anj
        #
        @simd for i in n:-1:j + 1
            BnAnj += B[i] * A[i, j]
        end

        B[j] += BnAnj
    end

    return
end

function strsm2!(A::AbstractMatrix{T}, B::AbstractMatrix{T}, uplo::Val{:L}, side::Val{:R}) where {T}
    @assert size(A, 1) == size(A, 2) == size(B, 2)

    n = size(A, 1)

    @inbounds for j in n:-1:1
        #
        #   A = [ 0   0   ]
        #       [ Anj Ann ]
        #
        Anj = @view A[j + 1:n, j]
        #
        #   B = [ Bj  Bn  ]
        #
        Bj = @view B[:, j]
        Bn = @view B[:, j + 1:n]
        #
        #   Bj ← Bj + Bn Anj
        #
        sgemm!(Bj, Bn, Anj)
    end

    return
end

function strsm!(A::AbstractMatrix{T}, B::AbstractVecOrMat{T}, uplo::Val{:L}, side::Val{:R}, blocksize::Int = DEFAULT_BLOCK_SIZE) where {T}
    if B isa AbstractVector
        @assert size(A, 1) == size(A, 2) == length(B)
    else
        @assert size(A, 1) == size(A, 2) == size(B, 2)
    end

    n = size(A, 1)

    @inbounds for stop in n:-blocksize:1
        size = min(blocksize, stop)
        strt = stop - size + 1
        #
        #   A = [ Abb 0   ]
        #       [ Anb Ann ]
        #
        Abb = @view A[strt:stop,  strt:stop]
        Anb = @view A[stop + 1:n, strt:stop]
        #
        #   B = [ Bb Bn ]
        #
        if B isa AbstractVector
            Bb = @view B[strt:stop]
            Bn = @view B[stop + 1:n]
        else
            Bb = @view B[:, strt:stop]
            Bn = @view B[:, stop + 1:n]
        end
        #
        #   Bb ← Bb + Bb Anb
        #
        sgemm!(Bb, Bn, Anb)
        #
        #   Bb ← Bb Abb*
        #
        strsm2!(Abb, Bb, uplo, side)
    end

    return
end

function strsm2!(A::T, B::AbstractVector{T}, uplo::Val{:U}, side::Val{:R}) where {T}
    n = length(B); invA = sinv(A)
    #
    #   B ← B A*
    #
    if iszero(invA)
        @inbounds @simd for i in 1:n
            B[i] = zero(T)
        end 
    else
        @inbounds @simd for i in 1:n
            B[i] = B[i] * invA
        end
    end

    return
end

function strsm2!(A::AbstractMatrix{T}, B::AbstractVector{T}, uplo::Val{:U}, side::Val{:R}) where {T}
    @assert size(A, 1) == size(A, 2) == length(B)

    n = size(A, 1)

    @inbounds for j in 1:n
        #
        #   A = [ Ann Anj ]
        #       [     Ajj ]
        #
        #   B = [ Bn  Bj ]
        #
        invAjj = sinv(A[j, j])

        if iszero(invAjj)
            Bj = zero(T)
        else
            BnAnj = zero(T)
            #
            #   BnAnj ← Bn Anj
            #
            @simd for i in 1:j - 1
                BnAnj += B[i] * A[i, j]
            end

            Bj = B[j]
            #
            #   Bj ← Bj + AnAnj
            #
            Bj += BnAnj
            #
            #   Bj ← Bj Ajj*
            #
            Bj *= invAjj
        end

        B[j] = Bj
    end

    return
end

function strsm2!(A::AbstractMatrix{T}, B::AbstractMatrix{T}, uplo::Val{:U}, side::Val{:R}) where {T}
    @assert size(A, 1) == size(A, 2) == size(B, 2)

    n = size(A, 1)

    @inbounds for j in 1:n
        #
        #   A = [ Ann Anj ]
        #       [     Ajj ]
        #
        Ajj =       A[j,       j]
        Anj = @view A[1:j - 1, j]
        #
        #   B = [ Bn Bj ]
        #
        Bj = @view B[:, j]
        Bn = @view B[:, 1:j - 1]
        #
        #   Bj ← Bj + Bn Anj
        #
        sgemm!(Bj, Bn, Anj)
        #
        #   Bj ← Bj Ajj*
        #
        strsm2!(Ajj, Bj, uplo, side)
    end

    return
end

function strsm!(A::AbstractMatrix{T}, B::AbstractVecOrMat{T}, uplo::Val{:U}, side::Val{:R}, blocksize::Int = DEFAULT_BLOCK_SIZE) where {T}
    if B isa AbstractVector
        @assert size(A, 1) == size(A, 2) == length(B)
    else
        @assert size(A, 1) == size(A, 2) == size(B, 2)
    end

    n = size(A, 1)

    @inbounds for strt in 1:blocksize:n
        size = min(blocksize, n - strt + 1)
        stop = strt + size - 1
        #
        #   A = [ Ann Anb ]
        #       [ 0   Abb ]
        #
        Abb = @view A[strt:stop,  strt:stop]
        Anb = @view A[1:strt - 1, strt:stop]
        #
        #   B = [ Bn Bb ]
        #
        if B isa AbstractVector
            Bb = @view B[strt:stop]
            Bn = @view B[1:strt - 1]
        else
            Bb = @view B[:, strt:stop]
            Bn = @view B[:, 1:strt - 1]
        end
        #
        #   Bb ← Bb + Bn Anb
        #
        sgemm!(Bb, Bn, Anb)
        #
        #   Bb ← Bb Abb*
        #
        strsm2!(Abb, Bb, uplo, side)
    end

    return
end

function sgemm!(C::AbstractMatrix{T}, A::AbstractMatrix{T}, B::AbstractMatrix{T}) where {T}
    @assert size(C, 1) == size(A, 1)
    @assert size(C, 2) == size(B, 2)
    @assert size(A, 2) == size(B, 1)

    @inbounds for j in axes(C, 2), k in axes(A, 2)
        Bkj = B[k, j]

        for i in axes(C, 1)
            C[i, j] += A[i, k] * Bkj
        end
    end

    return
end

function sgemm!(C::AbstractMatrix{T}, A::AbstractMatrix{T}, B::AbstractMatrix{T}) where {T <: Number}
    mul!(C, A, B, one(T), one(T))
    return
end

function sgemm!(C::AbstractVector{T}, A::AbstractMatrix{T}, B::AbstractVector{T}) where {T}
    @assert length(C) == size(A, 1)
    @assert length(B) == size(A, 2)

    @inbounds for j in axes(A, 2)
        Aj = @view A[:, j]
        Bj =       B[   j]
        #
        #   C ← C + Aj Bj
        #
        sgemm!(C, Aj, Bj)
    end

    return
end

function sgemm!(C::AbstractVector{T}, A::AbstractVector{T}, B::AbstractMatrix{T}) where {T}
    @assert length(A) == size(B, 1)
    @assert length(C) == size(B, 2)

    @inbounds for j in axes(B, 2)
        #
        #   C ← C + A Bj
        #
        ABj = zero(T)

        @simd for i in axes(B, 1)
            ABj += A[i] * B[i, j]
        end

        C[j] += ABj
    end

    return
end

function sgemm!(C::AbstractVector{T}, A::AbstractVector{T}, B::T) where {T}
    @assert length(C) == length(A)

    if !iszero(B)
        @inbounds @simd for i in eachindex(C)
            C[i] += A[i] * B
        end
    end

    return
end
