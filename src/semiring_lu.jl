"""
    SemiringLU{T, M <: AbstractMatrix{T}}

An LU factorization of a semiring-valued
matrix.
"""
struct SemiringLU{T, M <: AbstractMatrix{T}}
    factors::M
end

function Base.size(F::SemiringLU)
    return size(F.factors)
end

function Base.getproperty(F::SemiringLU, d::Symbol)
    if d == :L
        p = StrictLowerTriangular(F.factors)
    elseif d === :U
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

"""
    slu!(A::AbstractMatrix)

Compute an LU factorization of a semiring-
valued matrix A. The factors are stored
in A.
"""
function slu!(A::AbstractMatrix)
    sgetrf!(A)
    return SemiringLU(A)
end

"""
    slu(A::AbstractMatrix)

Compute an LU factorization of a semiring-
valued matrix A.
"""
function slu(A::AbstractMatrix)
    return slu!(Matrix(A))
end

"""
    sldiv!(C::AbstractArray, A, B::AbstractArray)

Solve the fixed-point equation

    AX + B = X.

The result is stored in C.
"""
function sldiv!(C::AbstractArray, A, B::AbstractArray)
    return sldiv!(A, copyto!(C, B))
end

"""
    sldiv!(A, B::AbstractArray)

Solve the fixed-point equation

    AX + B = X.

The result is stored in B.
"""
sldiv!(A, B::AbstractArray)

function sldiv!(A::Number, B::AbstractArray)
    @. B = sinv(A) * B
    return B
end

function sldiv!(A::StrictLowerTriangular, B::AbstractArray)
    strsm!(parent(A), B, Val(:L), Val(:L))
    return B
end

function sldiv!(A::UpperTriangular, B::AbstractArray)
    strsm!(parent(A), B, Val(:U), Val(:L))
    return B
end

function sldiv!(A::SemiringLU, B::AbstractArray)
    return sldiv!(A.U, sldiv!(A.L, B))
end

function sldiv!(A::AbstractMatrix, B::AbstractArray)
    return sldiv!(slu(A), B)
end

"""
    srdiv!(C::AbstractArray, B::AbstractArray, A)

Solve the fixed-point equation

    XA + B = X.

The result is stored in C.
"""
function srdiv!(C::AbstractArray, B::AbstractArray, A)
    return srdiv!(copyto!(C, B), A)
end

"""
    srdiv!(B::AbstractArray, A)

Solve the fixed-point equation

    XA + B = X.

The result is stored in B.
"""
srdiv!(B::AbstractArray, A)

function srdiv!(B::AbstractArray, A::Number)
    B .*= sinv(A)
    return B
end

function srdiv!(B::AbstractArray, A::StrictLowerTriangular)
    strsm!(parent(A), B, Val(:L), Val(:R))
    return B
end

function srdiv!(B::AbstractArray, A::UpperTriangular)
    strsm!(parent(A), B, Val(:U), Val(:R))
    return B
end

function srdiv!(B::AbstractArray, A::SemiringLU)
    return srdiv!(srdiv!(B, A.U), A.L)
end

function srdiv!(B::AbstractArray, A::AbstractMatrix)
    return srdiv!(B, slu(A))
end

function sinv(A::AbstractMatrix)
    return sinv(slu(A))
end

function sinv(A::StrictLowerTriangular{T}) where {T}
    B = Matrix{T}(undef, size(A))
    copyto!(B, A)
    strtri!(B, Val(:L))
    return B
end

function sinv(A::UpperTriangular{T}) where {T}
    B = Matrix{T}(undef, size(A))
    copyto!(B, A)
    strtri!(B, Val(:U))
    return B
end

function sinv(A::SemiringLU)
    return sldiv!(A.U, sinv(A.L))
end

# ------------------------ #
# Low-Level Matrix Kernels #
# ------------------------ #

function sgetrf2!(A::AbstractMatrix{T}) where {T}
    @assert size(A, 1) == size(A, 2)
    
    @inbounds for i in axes(A, 1)
        #
        #   A = [ Aii Ain ]
        #       [ Ani Ann ]
        #
        Aii =       A[i,         i]
        Ain = @view A[i,         i + 1:end]
        Ani = @view A[i + 1:end, i]
        Ann = @view A[i + 1:end, i + 1:end]

        #
        #   Ani ← Ani Aii*
        # 
        srdiv!(Ani, Aii)

        #
        #   Ann ← Ann + Ani Ain
        #
        mul!(Ann, Ani, Ain |> transpose, one(T), one(T))
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
        mul!(Ann, Anb, Abn, one(T), one(T))
    end

    return
end

function strsm2!(A::AbstractMatrix{T}, B::AbstractArray{T}, uplo::Val{:L}, side::Val{:L}) where {T}
    @assert size(A, 1) == size(A, 2) == size(B, 1)

    @inbounds for j in axes(B, 2), i in axes(A, 1)
        #
        #   A = [ 0   0   ]
        #       [ Ani Ann ]
        #
        Ani = @view A[i + 1:end, i]

        #
        #   B = [ Bi ]
        #       [ Bn ]
        #
        Bi =       B[i,         j]
        Bn = @view B[i + 1:end, j]

        if Bi != zero(T)
            #
            #   Bn ← Bn + Ani Bi
            #
            mul!(Bn, Ani, Bi, one(T), one(T))
        end
    end

    return
end

function strsm!(A::AbstractMatrix{T}, B::AbstractArray{T}, uplo::Val{:L}, side::Val{:L}, blocksize = DEFAULT_BLOCK_SIZE) where {T}
    @assert size(A, 1) == size(A, 2) == size(B, 1)

    n = size(A, 1)

    @inbounds for strt in 1:blocksize:n
        size = min(blocksize, n - strt + 1)
        stop = strt + size - 1

        #
        #   A = [ Abb 0   ]
        #       [ Anb Ann ]
        #
        Abb = @view A[strt:stop,    strt:stop]
        Anb = @view A[stop + 1:end, strt:stop]

        #
        #   B = [ Bb ]
        #       [ Bn ]
        #
        Bb = @view B[strt:stop,    :]
        Bn = @view B[stop + 1:end, :]

        #
        #   Bb ← Abb* Bb
        #
        strsm2!(Abb, Bb, uplo, side)

        #
        #   Bn ← Bn + Anb Bb
        #
        mul!(Bn, Anb, Bb, one(T), one(T))
    end

    return
end

function strsm2!(A::AbstractMatrix{T}, B::AbstractArray{T}, uplo::Val{:U}, side::Val{:L}) where {T}
    @assert size(A, 1) == size(A, 2) == size(B, 1)

    @inbounds for j in axes(B, 2), i in reverse(axes(A, 1))
        #
        #   A = [ Ann Ani ]
        #       [ 0   Aii ]
        #
        Ani = @view A[begin:i - 1, i]
        Aii =       A[i,           i]

        #
        #   B = [ Bn ]
        #       [ Bi ]
        #
        Bn = @view B[begin:i - 1, j]
        Bi =       B[i,           j]

        #
        #   Bi ← Aii* Bi
        #
        Bi = B[i, j] = sinv(Aii) * Bi

        if Bi != zero(T)
            #
            #   Bn ← Bn + Ani Bi
            #
            mul!(Bn, Ani, Bi, one(T), one(T))
        end
    end

    return
end

function strsm!(A::AbstractMatrix{T}, B::AbstractArray{T}, uplo::Val{:U}, side::Val{:L}, blocksize::Int = DEFAULT_BLOCK_SIZE) where {T}
    @assert size(A, 1) == size(A, 2) == size(B, 1)

    n = size(A, 1)

    @inbounds for stop in n:-blocksize:1
        size = min(blocksize, stop)
        strt = stop - size + 1

        #
        #   A = [ Ann Anb ]
        #       [ 0   Abb ]
        #
        Abb = @view A[strt:stop, strt:stop]
        Anb = @view A[begin:strt - 1, strt:stop]

        #
        #   B = [ Bn ]
        #       [ Bb ]
        #
        Bb = @view B[strt:stop, :]
        Bn = @view B[begin:strt - 1, :]

        #
        #   Bb ← Abb* Bbb
        #
        strsm2!(Abb, Bb, uplo, side)

        #
        #   Bn ← Bn + Anb Bb
        #
        mul!(Bn, Anb, Bb, one(T), one(T))
    end

    return
end

function strsm2!(A::AbstractMatrix{T}, B::AbstractArray{T}, uplo::Val{:L}, side::Val{:R}) where {T}
    @assert size(A, 1) == size(A, 2) == size(B, 2)

    @inbounds for i in reverse(axes(A, 1))
        #
        #   A = [ Ann 0 ]
        #       [ Ain 0 ]
        #
        Ain = @view A[i, begin:i - 1]

        #
        #   B = [ Bn Bi ]
        #
        Bi = @view B[:, i]
        Bn = @view B[:, begin:i - 1]
 
        #
        #
        #   Bn ← Bn + Bi Ain
        #
        mul!(Bn, Bi, Ain |> transpose, one(T), one(T))
    end

    return
end

function strsm!(A::AbstractMatrix{T}, B::AbstractArray{T}, uplo::Val{:L}, side::Val{:R}, blocksize::Int = DEFAULT_BLOCK_SIZE) where {T}
    @assert size(A, 1) == size(A, 2) == size(B, 2)

    n = size(A, 1)

    @inbounds for stop in n:-blocksize:1
        size = min(blocksize, stop)
        strt = stop - size + 1

        #
        #   A = [ Ann 0   ]
        #       [ Abn Abb ]
        #
        Abb = @view A[strt:stop, strt:stop]
        Abn = @view A[strt:stop, begin:strt - 1]

        #
        #   B = [ Bn Bb ]
        #
        Bb = @view B[:, strt:stop]
        Bn = @view B[:, begin:strt - 1]

        #
        #   Bb ← Bb Abb*
        #
        strsm2!(Abb, Bb, uplo, side)

        #
        #
        #   Bn ← Bn + Bb Abn
        #
        mul!(Bn, Bb, Abn, one(T), one(T))
    end

    return
end

function strsm2!(A::AbstractMatrix{T}, B::AbstractArray{T}, uplo::Val{:U}, side::Val{:R}) where {T}
    @assert size(A, 1) == size(A, 2) == size(B, 2)

    @inbounds for i in axes(A, 1)
        #
        #   A = [ Aii Ain ]
        #       [ 0   Ann ]
        #
        Aii =       A[i, i]
        Ain = @view A[i, i + 1:end]
 
        #
        #   B = [ Bi Bn ]
        #
        Bi = @view B[:, i]
        Bn = @view B[:, i + 1:end]

        #
        #   Bi ← Bi Aii*
        #
        srdiv!(Bi, Aii)
 
        #
        #
        #   Bn ← Bn + Bi Ain
        #
        mul!(Bn, Bi, Ain |> transpose, one(T), one(T))
    end

    return
end

function strsm!(A::AbstractMatrix{T}, B::AbstractArray{T}, uplo::Val{:U}, side::Val{:R}, blocksize::Int = DEFAULT_BLOCK_SIZE) where {T}
    @assert size(A, 1) == size(A, 2) == size(B, 2)

    n = size(A, 1)

    @inbounds for strt in 1:blocksize:n
        size = min(blocksize, n - strt + 1)
        stop = strt + size - 1

        #
        #   A = [ Abb Abn ]
        #       [ 0   Ann ]
        #
        Abb = @view A[strt:stop, strt:stop]
        Abn = @view A[strt:stop, stop + 1:end]

        #
        #   B = [ Bb Bn ]
        #
        Bb = @view B[:, strt:stop]
        Bn = @view B[:, stop + 1:end]

        #
        #   Bb ← Bb Abb*
        #
        strsm2!(Abb, Bb, uplo, side)
 
        #
        #
        #   Bn ← Bn + Bb Abn
        #
        mul!(Bn, Bb, Abn, one(T), one(T))
    end

    return
end

function strtri2!(A::AbstractMatrix{T}, uplo::Val{:L}) where {T}
    @assert size(A, 1) == size(A, 2)

    @inbounds for i in axes(A, 1)
        #
        #   A = [ Aim 0   0   ]
        #       [ Anm Ani Ann ]
        #
        Ani = @view A[i + 1:end, i]
        Aim = @view A[i,         begin:i - 1]
        Anm = @view A[i + 1:end, begin:i - 1]

        #
        #   Anm ← Anm + Ani Aim 
        #
        mul!(Anm, Ani, Aim |> transpose, one(T), one(T))

        #
        #   Aii ← 1
        #
        A[i, i] = one(T)
    end

    return
end

function strtri!(A::AbstractMatrix{T}, uplo::Val{:L}, blocksize = DEFAULT_BLOCK_SIZE) where {T}
    @assert size(A, 1) == size(A, 2)

    n = size(A, 1)

    @inbounds for strt in 1:blocksize:n
        size = min(blocksize, n - strt + 1)
        stop = strt + size - 1

        #
        #   A = [ Abm Abb 0   ]
        #       [ Anm Anb Ann ]
        #
        Abb = @view A[strt:stop,    strt:stop]
        Anb = @view A[stop + 1:end, strt:stop]
        Anm = @view A[stop + 1:end, begin:strt - 1]
        Abm = @view A[strt:stop,    begin:strt - 1]

        #
        #   Abm ← Abb* Abm
        #
        strsm2!(Abb, Abm, uplo, Val(:L))

        #
        #   Anm ← Anm + Anb Abn
        #
        mul!(Anm, Anb, Abm, one(T), one(T))

        #
        #   Anb ← Anb Abb*
        #
        strsm2!(Abb, Anb, uplo, Val(:R))

        #
        #   Abb ← Abb*
        #
        strtri2!(Abb, uplo)
    end
    
    return     
end

function strtri2!(A::AbstractMatrix{T}, uplo::Val{:U}) where {T}
    @assert size(A, 1) == size(A, 2)

    @inbounds for i in reverse(axes(A, 1))
        #
        #   A = [ Ann Ani Anm ]
        #       [ 0   Aii Aim ]
        #
        Aii = A[i, i]
        Ani = @view A[begin:i - 1, i]
        Aim = @view A[i,           i + 1:end]
        Anm = @view A[begin:i - 1, i + 1:end]

        #
        #   Aim ← Aii* Aim
        #
        sldiv!(Aii, Aim)

        #
        #   Anm ← Anm + Ani Aim 
        #
        mul!(Anm, Ani, Aim |> transpose, one(T), one(T))

        #
        #   Ani ← Ani Aii*
        #
        srdiv!(Ani, Aii)

        #
        #   Aii ← Aii*
        #
        A[i, i] = sinv(Aii)
    end

    return
end

function strtri!(A::AbstractMatrix{T}, uplo::Val{:U}, blocksize = DEFAULT_BLOCK_SIZE) where {T}
    @assert size(A, 1) == size(A, 2)

    n = size(A, 1)

    @inbounds for stop in n:-blocksize:1
        size = min(blocksize, stop)
        strt = stop - size + 1

        #
        #   A = [ Ann Anb Anm ]
        #       [ 0   Abb Abm ]
        #
        Abb = @view A[strt:stop,      strt:stop]
        Anb = @view A[begin:strt - 1, strt:stop]
        Abm = @view A[strt:stop,      stop + 1:end]
        Anm = @view A[begin:strt - 1, stop + 1:end]

        #
        #   Abm ← Abb* Abm
        #
        strsm2!(Abb, Abm, uplo, Val(:L))

        #
        #   Anm ← Anm + Anb Abm 
        #
        mul!(Anm, Anb, Abm, one(T), one(T))

        #
        #   Anb ← Anb Abb*
        #
        strsm2!(Abb, Anb, uplo, Val(:R))

        #
        #   Abb ← Abb*
        #
        strtri2!(Abb, uplo)
    end
    
    return     
end
