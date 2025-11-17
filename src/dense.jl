"""
    StarLU{T, M <: AbstractMatrix{T}} <: AbstractStarLU{T}

An LU factorization of a semiring-valued
matrix.
"""
struct StarLU{T, M <: AbstractMatrix{T}} <: AbstractStarLU{T}
    factors::M
end

function Base.getproperty(F::StarLU, d::Symbol)
    if d == :L
        p = StrictLowerTriangular(F.factors)
    elseif d == :U
        p = UpperTriangular(F.factors)
    else
        p = getfield(F, d)
    end

    return p
end

function Base.show(io::IO, mime::MIME"text/plain", F::T) where {T <: StarLU}
    print(io, "$T:")
    print(io, "\nL factor:\n")
    show(io, mime, F.L)
    print(io, "\nU factor:\n")
    show(io, mime, F.U)
    return
end

function Base.size(F::StarLU)
    return size(F.factors)
end

# --- #
# slu #
# --- #

function slu!(A::AbstractMatrix)
    slu_impl!(A)
    return StarLU(A)
end

function slu_impl2!(A::AbstractMatrix{T}) where {T}
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
        smul_impl2!(Aii, Ani, Val(:N), Val(:U), Val(:R)) 
        #
        #   Ann ← Ann + Ani Ain
        #
        for j in i + 1:n
            Aij =       A[i,       j]
            Anj = @view A[i + 1:n, j]
            #
            #   Anj ← Anj + Ani Aij
            #
            mul_impl!(Anj, Ani, Aij, Val(:N), Val(:N))
        end
    end

    return
end

function slu_impl!(A::AbstractMatrix{T}, blocksize::Int = DEFAULT_BLOCK_SIZE) where {T}
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
        slu_impl2!(Abb)
        # 
        #   Abn ← Lbb* Abn
        #    
        smul_impl!(Abb, Abn, Val(:N), Val(:L), Val(:L))
        #
        #   Anb ← Anb Ubb*
        #
        smul_impl!(Abb, Anb, Val(:N), Val(:U), Val(:R))
        #
        #   Ann ← Ann + Anb Abn
        #
        mul_impl!(Ann, Anb, Abn, Val(:N), Val(:N))
    end

    return
end

# ------ #
# slmul! #
# srmul! #
# ------ #

function slmul!(A::Number, B::AbstractVecOrMat)
    smul_impl!(A, B, Val(:N), Val(:U), Val(:L))
    return B
end

function slmul!(A::StrictLowerTriangular, B::AbstractVecOrMat)
    smul_impl!(parent(A), B, Val(:N), Val(:L), Val(:L))
    return B
end

function slmul!(A::UpperTriangular, B::AbstractVecOrMat)
    smul_impl!(parent(A), B, Val(:N), Val(:U), Val(:L))
    return B
end

function slmul!(A::StarLU, B::AbstractVecOrMat)
    return slmul!(A.U, slmul!(A.L, B))
end

function srmul!(B::AbstractVecOrMat, A::Number)
    smul_impl!(A, B, Val(:N), Val(:U), Val(:R))
    return B
end

function srmul!(B::AbstractVecOrMat, A::StrictLowerTriangular)
    smul_impl!(parent(A), B, Val(:N), Val(:L), Val(:R))
    return B
end

function srmul!(B::AbstractVecOrMat, A::UpperTriangular)
    smul_impl!(parent(A), B, Val(:N), Val(:U), Val(:R))
    return B
end

function srmul!(B::AbstractVecOrMat, A::StarLU)
    return srmul!(srmul!(B, A.U), A.L)
end

function slres!(A::StarLU, B::AbstractVecOrMat)
    return slres!(A.L, slres!(A.U, B))
end

function slres!(A::StrictLowerTriangular, B::AbstractVecOrMat)
    smul_impl!(parent(A), B, Val(:C), Val(:L), Val(:L))
    return B
end

function slres!(A::UpperTriangular, B::AbstractVecOrMat)
    smul_impl!(parent(A), B, Val(:C), Val(:U), Val(:L))
    return B
end

function srres!(B::AbstractVecOrMat, A::StarLU)
    return srres!(srres!(B, A.L), A.U)
end

function srres!(B::AbstractVecOrMat, A::StrictLowerTriangular)
    smul_impl!(parent(A), B, Val(:C), Val(:L), Val(:R))
    return B
end

function srres!(B::AbstractVecOrMat, A::UpperTriangular)
    smul_impl!(parent(A), B, Val(:C), Val(:U), Val(:R))
    return B
end

function smul_impl2!(A::AbstractMatrix{T}, B::AbstractVecOrMat{T}, tA::Val{:C}, uplo::Val{:L}, side::Val{:L}) where {T}
    @assert size(A, 1) == size(A, 2) == size(B, 1)

    n = size(A, 1)
    m = size(B, 2)

    @inbounds for j in 1:m, i in n:-1:1
        #
        #   A = [ Ann   ]
        #       [ Ain 0 ]
        #
        Ain = @view A[i, 1:i - 1]
        #
        #   B = [ Bn ]
        #       [ Bi ]
        #
        Bn = @view B[1:i - 1, j]
        Bi =       B[i,       j]
        #
        #   Bn ← Bn ∧ Ain \ Bi
        #
        mul_impl!(Bn, Ain, Bi, tA, Val(:N))
    end

    return
end

function smul_impl!(A::AbstractMatrix{T}, B::AbstractVecOrMat{T}, tA::Val{:C}, uplo::Val{:L}, side::Val{:L}, blocksize::Int = DEFAULT_BLOCK_SIZE) where {T}
    @assert size(A, 1) == size(A, 2) == size(B, 1)

    n = size(A, 1)

    @inbounds for stop in n:-blocksize:1
        size = min(blocksize, stop)
        strt = stop - size + 1
        #
        #   A = [ Ann 0   ]
        #       [ Abn Abb ]
        #
        Abb = @view A[strt:stop, strt:stop]
        Abn = @view A[strt:stop, 1:strt - 1]
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
        #   Bb ← Abb* \ Bbb
        #
        smul_impl2!(Abb, Bb, tA, uplo, side)
        #
        #   Bn ← Bn ∧ Abn \ Bb
        #
        mul_impl!(Bn, Abn, Bb, tA, Val(:N))
    end

    return
end

function smul_impl2!(A::T, B::AbstractScalar{T}, tA::Val{:C}, uplo::Val{:U}, side::Val{:L}) where {T}
    #
    #   B ← A* \ B
    #
    B[] = slres(A, B[])
    return
end

function smul_impl2!(A::AbstractMatrix{T}, B::AbstractVecOrMat{T}, tA::Val{:C}, uplo::Val{:U}, side::Val{:L}) where {T}
    @assert size(A, 1) == size(A, 2) == size(B, 1)

    n = size(A, 1)
    m = size(B, 2)

    @inbounds for j in 1:m, i in 1:n
        #
        #   A = [ Aii Ain ]
        #       [     Ann ]
        #
        Aii =       A[i,       i]
        Ain = @view A[i, i + 1:n]
        #
        #   B = [ Bi ]
        #       [ Bn ]
        #
        Bi = @view B[i,       j]
        Bn = @view B[i + 1:n, j]
        #
        #   Bi ← Aii* \ Bi
        #
        smul_impl2!(Aii, Bi, tA, uplo, side)
        #
        #   Bn ← Bn + Ain \ Bi
        #
        mul_impl!(Bn, Ain, Bi[], tA, Val(:N))
    end

    return
end

function smul_impl!(A::AbstractMatrix{T}, B::AbstractVecOrMat{T}, tA::Val{:C}, uplo::Val{:U}, side::Val{:L}, blocksize = DEFAULT_BLOCK_SIZE) where {T}
    @assert size(A, 1) == size(A, 2) == size(B, 1)

    n = size(A, 1)

    @inbounds for strt in 1:blocksize:n
        size = min(blocksize, n - strt + 1)
        stop = strt + size - 1
        #
        #   A = [ Abb Abn ]
        #       [ Anb Ann ]
        #
        Abb = @view A[strt:stop, strt:stop]
        Abn = @view A[strt:stop, stop + 1:n]
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
        #   Bb ← Abb* \ Bb
        #
        smul_impl2!(Abb, Bb, tA, uplo, side)
        #
        #   Bn ← Bn ∧ Abn \ Bb
        #
        mul_impl!(Bn, Abn, Bb, tA, Val(:N))
    end

    return
end

function smul_impl2!(A::AbstractMatrix{T}, B::AbstractVecOrMat{T}, tA::Val{:C}, uplo::Val{:L}, side::Val{:R}) where {T}
    if B isa AbstractVector
        @assert size(A, 1) == size(A, 2) == length(B)
    else
        @assert size(A, 1) == size(A, 2) == size(B, 2)
    end

    n = size(A, 1)

    @inbounds for j in 1:n
        #
        #   A = [ Ann   ]
        #       [ Ajn 0 ]
        #
        Ajn = @view A[j, 1:j - 1]
        #
        #   B = [ Bn Bj ]
        #
        if B isa AbstractVector
            Bj = @view B[j]
            Bn = @view B[1:j - 1]
        else
            Bj = @view B[:, j]
            Bn = @view B[:, 1:j - 1]
        end
        #
        #   Bj ← Bj ∧ Bn / Ajn
        #
        mul_impl!(Bj, Bn, Ajn, Val(:N), tA)
    end

    return
end

function smul_impl!(A::AbstractMatrix{T}, B::AbstractVecOrMat{T}, tA::Val{:C}, uplo::Val{:L}, side::Val{:R}, blocksize::Int = DEFAULT_BLOCK_SIZE) where {T}
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
        #   A = [ Ann     ]
        #       [ Abn Abb ]
        #
        Abb = @view A[strt:stop, strt:stop]
        Abn = @view A[strt:stop, 1:strt - 1]
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
        #   Bb ← Bb ∧ Bn / Abn
        #
        mul_impl!(Bb, Bn, Abn, Val(:N), tA)
        #
        #   Bb ← Bb / Abb*
        #
        smul_impl2!(Abb, Bb, tA, uplo, side)
    end

    return
end

function smul_impl2!(A::T, B::AbstractScalar{T}, tA::Val{:C}, uplo::Val{:U}, side::Val{:R}) where {T}
    #
    #   B ← B / A*
    #
    B[] = srres(B[], A)
    return
end

function smul_impl2!(A::T, B::AbstractVector{T}, tA::Val{:C}, uplo::Val{:U}, side::Val{:R}) where {T}
    n = length(B)
    #
    #   B ← B / A*
    #
    @inbounds @simd for i in 1:n
        B[i] = srres(B[i], A)
    end

    return
end

function smul_impl2!(A::AbstractMatrix{T}, B::AbstractVecOrMat{T}, tA::Val{:C}, uplo::Val{:U}, side::Val{:R}) where {T}
    if B isa AbstractVector
         @assert size(A, 1) == size(A, 2) == length(B)
    else
        @assert size(A, 1) == size(A, 2) == size(B, 2)
    end

    n = size(A, 1)

    @inbounds for j in n:-1:1
        #
        #   A = [ Ajj Ajn ]
        #       [     Ann ]
        #
        Ajj =       A[j,       j]
        Ajn = @view A[j, j + 1:n]
        #
        #   B = [ Bj  Bn  ]
        #
        if B isa AbstractVector
            Bj = @view B[j]
            Bn = @view B[j + 1:n]
        else
            Bj = @view B[:, j]
            Bn = @view B[:, j + 1:n]
        end
        #
        #   Bj ← Bj ∧ Bn / Ajn
        #
        mul_impl!(Bj, Bn, Ajn, Val(:N), tA)
        #
        #   Bb ← Bb / Ajj*
        #
        smul_impl2!(Ajj, Bj, tA, uplo, side)
    end

    return
end

function smul_impl!(A::AbstractMatrix{T}, B::AbstractVecOrMat{T}, tA::Val{:C}, uplo::Val{:U}, side::Val{:R}, blocksize::Int = DEFAULT_BLOCK_SIZE) where {T}
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
        #   A = [ Abb Abn ]
        #       [     Ann ]
        #
        Abb = @view A[strt:stop, strt:stop]
        Abn = @view A[strt:stop, stop + 1:n]
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
        #   Bb ← Bb ∧ Bb / Abn
        #
        mul_impl!(Bb, Bn, Abn, Val(:N), tA)
        #
        #   Bb ← Bb / Abb*
        #
        smul_impl2!(Abb, Bb, tA, uplo, side)
    end

    return
end


function smul_impl2!(A::AbstractMatrix{T}, B::AbstractVecOrMat{T}, tA::Val{:N}, uplo::Val{:L}, side::Val{:L}) where {T}
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
        mul_impl!(Bn, Ani, Bi, tA, Val(:N))
    end

    return
end

function smul_impl!(A::AbstractMatrix{T}, B::AbstractVecOrMat{T}, tA::Val{:N}, uplo::Val{:L}, side::Val{:L}, blocksize = DEFAULT_BLOCK_SIZE) where {T}
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
        smul_impl2!(Abb, Bb, tA, uplo, side)
        #
        #   Bn ← Bn + Anb Bb
        #
        mul_impl!(Bn, Anb, Bb, tA, Val(:N))
    end

    return
end

function smul_impl2!(A::T, B::AbstractScalar{T}, tA::Val{:N}, uplo::Val{:U}, side::Val{:L}) where {T}
    #
    #   B ← A* B
    #
    B[] = slmul(A, B[])
    return
end

function smul_impl2!(A::T, B::AbstractVector{T}, tA::Val{:N}, uplo::Val{:U}, side::Val{:L}) where {T}
    n = length(B)
    #
    #   B ← A* B
    #
    @inbounds @simd for i in 1:n
        B[i] = slmul(A, B[i])
    end

    return
end

function smul_impl2!(A::AbstractMatrix{T}, B::AbstractVecOrMat{T}, tA::Val{:N}, uplo::Val{:U}, side::Val{:L}) where {T}
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
        Bi = @view B[i,       j]
        #
        #   Bi ← Aii* Bi
        #
        smul_impl2!(Aii, Bi, tA, uplo, side)
        #
        #   Bn ← Bn + Ani Bi
        #
        mul_impl!(Bn, Ani, Bi[], tA, Val(:N))
    end

    return
end

function smul_impl!(A::AbstractMatrix{T}, B::AbstractVecOrMat{T}, tA::Val{:N}, uplo::Val{:U}, side::Val{:L}, blocksize::Int = DEFAULT_BLOCK_SIZE) where {T}
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
        smul_impl2!(Abb, Bb, tA, uplo, side)
        #
        #   Bn ← Bn + Anb Bb
        #
        mul_impl!(Bn, Anb, Bb, tA, Val(:N))
    end

    return
end

function smul_impl2!(A::AbstractMatrix{T}, B::AbstractVecOrMat{T}, tA::Val{:N}, uplo::Val{:L}, side::Val{:R}) where {T}
    if B isa AbstractVector
         @assert size(A, 1) == size(A, 2) == length(B)
    else
        @assert size(A, 1) == size(A, 2) == size(B, 2)
    end

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
        if B isa AbstractVector
            Bj = @view B[j]
            Bn = @view B[j + 1:n]
        else
            Bj = @view B[:, j]
            Bn = @view B[:, j + 1:n]
        end
        #
        #   Bj ← Bj + Bn Anj
        #
        mul_impl!(Bj, Bn, Anj, Val(:N), tA)
    end

    return
end

function smul_impl!(A::AbstractMatrix{T}, B::AbstractVecOrMat{T}, tA::Val{:N}, uplo::Val{:L}, side::Val{:R}, blocksize::Int = DEFAULT_BLOCK_SIZE) where {T}
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
        mul_impl!(Bb, Bn, Anb, Val(:N), tA)
        #
        #   Bb ← Bb Abb*
        #
        smul_impl2!(Abb, Bb, tA, uplo, side)
    end

    return
end

function smul_impl2!(A::T, B::AbstractScalar{T}, tA::Val{:N}, uplo::Val{:U}, side::Val{:R}) where {T}
    #
    #   B ← B A*
    #
    B[] = srmul(B[], A)
    return
end

function smul_impl2!(A::T, B::AbstractVector{T}, tA::Val{:N}, uplo::Val{:U}, side::Val{:R}) where {T}
    n = length(B)
    #
    #   B ← B A*
    #
    @inbounds @simd for i in 1:n
        B[i] = srmul(B[i], A)
    end

    return
end

function smul_impl2!(A::AbstractMatrix{T}, B::AbstractVecOrMat{T}, tA::Val{:N}, uplo::Val{:U}, side::Val{:R}) where {T}
    if B isa AbstractVector
        @assert size(A, 1) == size(A, 2) == length(B)
    else
        @assert size(A, 1) == size(A, 2) == size(B, 2)
    end

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
        if B isa AbstractVector
            Bj = @view B[j]
            Bn = @view B[1:j - 1]
        else
            Bj = @view B[:, j]
            Bn = @view B[:, 1:j - 1]
        end
        #
        #   Bj ← Bj + Bn Anj
        #
        mul_impl!(Bj, Bn, Anj, Val(:N), tA)
        #
        #   Bj ← Bj Ajj*
        #
        smul_impl2!(Ajj, Bj, tA, uplo, side)
    end

    return
end

function smul_impl!(A::AbstractMatrix{T}, B::AbstractVecOrMat{T}, tA::Val{:N}, uplo::Val{:U}, side::Val{:R}, blocksize::Int = DEFAULT_BLOCK_SIZE) where {T}
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
        mul_impl!(Bb, Bn, Anb, Val(:N), tA)
        #
        #   Bb ← Bb Abb*
        #
        smul_impl2!(Abb, Bb, tA, uplo, side)
    end

    return
end

# ---- #
# mul  #
# lres #
# rres #
# ---- #

function mul_impl!(C::AbstractVector, A::AbstractMatrix, B::AbstractVector, tA::Val{:N}, tB::Val)
    @assert length(C) == size(A, 1)
    @assert length(B) == size(A, 2)

    @inbounds for j in axes(A, 2)
        Aj = @view A[:, j]
        Bj =       B[   j]  
        #   
        #   C ← C + Aj Bj
        #   
        mul_impl!(C, Aj, Bj, tA, tB) 
    end 

    return
end

function mul_impl!(C::AbstractVector, A::AbstractVector, B::AbstractMatrix, tA::Val{:N}, tB::Val{:C})
    @assert length(C) == size(B, 1)
    @assert length(A) == size(B, 2)

    @inbounds for j in eachindex(A)
        Aj =       A[   j]
        Bj = @view B[:, j]
        #
        #   C ← C ∧ Aj / Bj
        #
        mul_impl!(C, Aj, Bj, tA, tB)
    end

    return
end

function mul_impl!(C::AbstractMatrix, A::AbstractVector, B::AbstractVector, tA::Val{:N}, tB::Val)
    @assert size(C, 1) == length(A)
    @assert size(C, 2) == length(B)

    @inbounds for j in eachindex(B)
        Cj = @view C[:, j]
        Bj =       B[   j]  
        #   
        #   Cj ← Cj + A Bj
        #   
        mul_impl!(Cj, A, Bj, tA, tB) 
    end 

    return
end

function mul_impl!(C::AbstractVector{T}, A::T, B::AbstractVector{T}, tA::Val, tB::Val) where {T}
    @assert length(C) == length(B)
    #
    #   C ← C + A B
    #
    @inbounds for i in eachindex(C)
        Ci = @view C[i]
        Bi =       B[i]
        #
        #   Ci ← A Bi
        #
        mul_impl!(Ci, A, Bi, tA, tB)
    end

    return
end

function mul_impl!(C::AbstractVector{T}, A::AbstractVector{T}, B::T, tA::Val, tB::Val) where {T}
    @assert length(C) == length(A)
    #
    #   C ← C + A B
    #
    @inbounds for i in eachindex(C)
        Ci = @view C[i]
        Ai =       A[i]
        #
        #   Ci ← Ai B
        #
        mul_impl!(Ci, Ai, B, tA, tB)
    end

    return
end

function mul_impl!(C::StridedMatrix{T}, A::StridedMatrix{T}, B::StridedMatrix{T}, tA::Val{:N}, tB::Val{:N}) where {T <: Number}
    matmul!(C, A, B, one(T), one(T))
    return
end

function vmuladd(A::AbstractVector{T}, B::AbstractVector{T}) where {T}
    @assert length(A) == length(B)
    #
    #   C ← A B
    #
    C = zero(T)

    @inbounds @simd for i in eachindex(A)
        #
        #   C ← C + Ai Bi
        #
        C = muladd(A[i], B[i], C)
    end

    return C
end

function vlresinf(A::AbstractVector{T}, B::AbstractVector{T}) where {T}
    @assert length(A) == length(B)
    #
    #   C ← A \ B
    #
    C = typemax(T)

    @inbounds @simd for i in eachindex(A)
        #
        #   C ← C ∧ Ai \ Bi
        #
        C = lresinf(A[i], B[i], C)
    end

    return C
end

function vrresinf(A::AbstractVector{T}, B::AbstractVector{T}) where {T}
    @assert length(A) == length(B)
    #
    #   C ← A / B
    #
    C = typemax(T)

    @inbounds @simd for i in eachindex(A)
        #
        #   C ← C ∧ Ai / Bi
        #
        C = rresinf(A[i], B[i], C)
    end

    return C
end
