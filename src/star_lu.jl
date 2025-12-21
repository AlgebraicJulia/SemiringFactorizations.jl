struct StarLU{T, M <: AbstractMatrix{T}} <: AbstractStarLU{T}
    matrix::M
end

struct StarTriangular{Q, T, M <: AbstractMatrix{T}} <: AbstractStarTriangular{Q, T}
    matrix::M
end

function AbstractStarTriangular{Q}(F::StarLU) where {Q}
    return StarTriangular{Q}(F)
end

function StarTriangular{Q}(F::StarLU) where {Q}
    return StarTriangular{Q}(parent(F))
end

function StarTriangular{Q}(A::M) where {Q, T, M <: AbstractMatrix{T}}
    return StarTriangular{Q, T, M}(A)
end

function Base.parent(F::Union{StarLU, StarTriangular})
    return F.matrix
end

function Base.size(F::StarTriangular)
    return size(parent(F))
end

function Semirings.add_impl(A::AbstractMatrix, B::AbstractMatrix, dual::Val{:C})
    return A .& B
end

# -------------------- #
# Matrix Factorization #
# -------------------- #

"""
    slu(A::AbstractMatrix)

Construct an [`AbstractStarLU`](@ref) factorization
object for the Kleene star ``A^*``.
"""
function slu(A::AbstractMatrix)
    return slu!(Matrix(A))
end

function slu(A::StrictLowerTriangular)
    return StarTriangular{:L}(A)
end

function slu(A::UpperTriangular)
    return StarTriangular{:U}(A)
end

function slu!(A::AbstractMatrix)
    slu_impl!(A)
    return StarLU(A)
end

function slu_impl2!(A::AbstractMatrix)
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
            #   Anj ← Ani Aij + Anj
            #
            mul_add_impl!(Ani, Aij, Anj, Val(:N), Val(:N), Val(:N))
        end
    end

    return A
end

function slu_impl!(A::AbstractMatrix, blocksize::Int = DEFAULT_BLOCK_SIZE)
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
        #   Ann ← Anb Abn + Ann
        #
        mul_add_impl!(Anb, Abn, Ann, Val(:N), Val(:N), Val(:N))
    end

    return A
end

# ---------------- #
# Triangular Solve #
# ---------------- #

function LinearAlgebra.lmul!(A::StarTriangular{Q}, B::AbstractVecOrMat) where {Q}
    return smul_impl!(parent(A), B, Val(:N), Val(Q), Val(:L))
end

function LinearAlgebra.rmul!(B::AbstractVecOrMat, A::StarTriangular{Q}) where {Q}
    return smul_impl!(parent(A), B, Val(:N), Val(Q), Val(:R))
end

function LinearAlgebra.ldiv!(A::StarTriangular{Q}, B::AbstractVecOrMat) where {Q}
    return smul_impl!(parent(A), B, Val(:C), Val(Q), Val(:L))
end

function LinearAlgebra.rdiv!(B::AbstractVecOrMat, A::StarTriangular{Q}) where {Q}
    return smul_impl!(parent(A), B, Val(:C), Val(Q), Val(:R))
end

function smul_impl2!(A::Number, B::AbstractScalar, tA::Val, uplo::Val{:U}, side::Val)
    #
    #   B ← A* B
    #
    B[] = smul_impl(A, B[], tA, side)
    return B
end

function smul_impl2!(A::Number, B::AbstractVector, tA::Val, uplo::Val{:U}, side::Val)
    n = length(B)
    #
    #   B ← A* B
    #
    @inbounds @simd for i in 1:n
        B[i] = smul_impl(A, B[i], tA, side)
    end

    return B
end

function smul_impl2!(A::AbstractMatrix, B::AbstractVecOrMat, tA::Val{:C}, uplo::Val{:L}, side::Val{:L})
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
        #   Bn ← Ain \ Bi & Bn
        #
        mul_add_impl!(Ain, Bi, Bn, tA, Val(:N), tA)
    end

    return B
end

function smul_impl!(A::AbstractMatrix, B::AbstractVecOrMat, tA::Val{:C}, uplo::Val{:L}, side::Val{:L}, blocksize::Int = DEFAULT_BLOCK_SIZE)
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
        #   Bn ← Abn \ Bb & Bn
        #
        mul_add_impl!(Abn, Bb, Bn, tA, Val(:N), tA)
    end

    return B
end

function smul_impl2!(A::AbstractMatrix, B::AbstractVecOrMat, tA::Val{:C}, uplo::Val{:U}, side::Val{:L})
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
        #   Bn ← Ain \ Bi + Bn
        #
        mul_add_impl!(Ain, Bi[], Bn, tA, Val(:N), tA)
    end

    return B
end

function smul_impl!(A::AbstractMatrix, B::AbstractVecOrMat, tA::Val{:C}, uplo::Val{:U}, side::Val{:L}, blocksize = DEFAULT_BLOCK_SIZE)
    @assert size(A, 1) == size(A, 2) == size(B, 1)

    n = size(A, 1)

    @inbounds for strt in 1:blocksize:n
        size = min(blocksize, n - strt + 1)
        stop = strt + size - 1
        #
        #   A = [ Abb Abn ]
        #       [     Ann ]
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
        #   Bn ← Abn \ Bb & Bn
        #
        mul_add_impl!(Abn, Bb, Bn, tA, Val(:N), tA)
    end

    return B
end

function smul_impl2!(A::AbstractMatrix, B::AbstractVecOrMat, tA::Val{:C}, uplo::Val{:L}, side::Val{:R})
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
        #   Bj ← Bn / Ajn & Bj
        #
        mul_add_impl!(Bn, Ajn, Bj, Val(:N), tA, tA)
    end

    return
end

function smul_impl!(A::AbstractMatrix, B::AbstractVecOrMat, tA::Val{:C}, uplo::Val{:L}, side::Val{:R}, blocksize::Int = DEFAULT_BLOCK_SIZE)
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
        #   Bb ← Bn / Abn & Bb
        #
        mul_add_impl!(Bn, Abn, Bb, Val(:N), tA, tA)
        #
        #   Bb ← Bb / Abb*
        #
        smul_impl2!(Abb, Bb, tA, uplo, side)
    end

    return B
end

function smul_impl2!(A::AbstractMatrix, B::AbstractVecOrMat, tA::Val{:C}, uplo::Val{:U}, side::Val{:R})
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
        #   Bj ← Bn / Ajn & Bj
        #
        mul_add_impl!(Bn, Ajn, Bj, Val(:N), tA, tA)
        #
        #   Bb ← Bb / Ajj*
        #
        smul_impl2!(Ajj, Bj, tA, uplo, side)
    end

    return B
end

function smul_impl!(A::AbstractMatrix, B::AbstractVecOrMat, tA::Val{:C}, uplo::Val{:U}, side::Val{:R}, blocksize::Int = DEFAULT_BLOCK_SIZE)
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
        #   Bb ← Bn / Abn & Bb
        #
        mul_add_impl!(Bn, Abn, Bb, Val(:N), tA, tA)
        #
        #   Bb ← Bb / Abb*
        #
        smul_impl2!(Abb, Bb, tA, uplo, side)
    end

    return B
end


function smul_impl2!(A::AbstractMatrix, B::AbstractVecOrMat, tA::Val{:N}, uplo::Val{:L}, side::Val{:L})
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
        #   Bn ← Ani Bi + Bn
        #
        mul_add_impl!(Ani, Bi, Bn, tA, Val(:N), tA)
    end

    return B
end

function smul_impl!(A::AbstractMatrix, B::AbstractVecOrMat, tA::Val{:N}, uplo::Val{:L}, side::Val{:L}, blocksize = DEFAULT_BLOCK_SIZE)
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
        #   Bn ← Anb Bb + Bn
        #
        mul_add_impl!(Anb, Bb, Bn, tA, Val(:N), tA)
    end

    return B
end

function smul_impl2!(A::AbstractMatrix, B::AbstractVecOrMat, tA::Val{:N}, uplo::Val{:U}, side::Val{:L})
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
        #   Bn ← Ani Bi + Bn
        #
        mul_add_impl!(Ani, Bi[], Bn, tA, Val(:N), tA)
    end

    return B
end

function smul_impl!(A::AbstractMatrix, B::AbstractVecOrMat, tA::Val{:N}, uplo::Val{:U}, side::Val{:L}, blocksize::Int = DEFAULT_BLOCK_SIZE)
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
        #   Bn ← Anb Bb + Bn
        #
        mul_add_impl!(Anb, Bb, Bn, tA, Val(:N), tA)
    end

    return B
end

function smul_impl2!(A::AbstractMatrix, B::AbstractVecOrMat, tA::Val{:N}, uplo::Val{:L}, side::Val{:R})
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
        #   Bj ← Bn Anj + Bj
        #
        mul_add_impl!(Bn, Anj, Bj, Val(:N), tA, tA)
    end

    return B
end

function smul_impl!(A::AbstractMatrix, B::AbstractVecOrMat, tA::Val{:N}, uplo::Val{:L}, side::Val{:R}, blocksize::Int = DEFAULT_BLOCK_SIZE)
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
        #   Bb ← Bn Anb + Bb
        #
        mul_add_impl!(Bn, Anb, Bb, Val(:N), tA, tA)
        #
        #   Bb ← Bb Abb*
        #
        smul_impl2!(Abb, Bb, tA, uplo, side)
    end

    return B
end

function smul_impl2!(A::AbstractMatrix, B::AbstractVecOrMat, tA::Val{:N}, uplo::Val{:U}, side::Val{:R})
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
        #   Bj ← Bn Anj + Bj
        #
        mul_add_impl!(Bn, Anj, Bj, Val(:N), tA, tA)
        #
        #   Bj ← Bj Ajj*
        #
        smul_impl2!(Ajj, Bj, tA, uplo, side)
    end

    return B
end

function smul_impl!(A::AbstractMatrix, B::AbstractVecOrMat, tA::Val{:N}, uplo::Val{:U}, side::Val{:R}, blocksize::Int = DEFAULT_BLOCK_SIZE)
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
        #   Bb ← Bn Anb + Bb
        #
        mul_add_impl!(Bn, Anb, Bb, Val(:N), tA, tA)
        #
        #   Bb ← Bb Abb*
        #
        smul_impl2!(Abb, Bb, tA, uplo, side)
    end

    return B
end

# --------------------- #
# Matrix Multiplication #
# --------------------- #

function Semirings.mul_impl(A::AbstractMatrix{T}, B::AbstractMatrix{T}, tA::Val, tB::Val, dual::Val) where {T <: SemiringNumber}
    return mul_add_impl!(A, B, fill(typemax(T), size(A, 2), size(B, 2)), tA, tB, dual)
end

function Semirings.mul_impl(A::AbstractMatrix{T}, B::AbstractVector{T}, tA::Val, tB::Val, dual::Val) where {T <: SemiringNumber}
    return mul_add_impl!(A, B, fill(typemax(T), size(A, 2)), tA, tB, dual)
end

function Semirings.mul_impl(A::AbstractRowVector{T}, B::AbstractVector{T}, tA::Val, tB::Val, dual::Val) where {T <: SemiringNumber}
    return mul_add_impl(parent(A), B, typemax(T), tA, tB, dual)
end

function Semirings.mul_add_impl(A, B, C::AbstractVecOrMat, tA::Val, tB::Val, dual::Val)
    return mul_add_impl!(A, B, Array(C), tA, tB, dual)
end

function Semirings.mul_add_impl(A::AbstractVector, B::AbstractVector, C::Number, tA::Val, tB::Val, dual::Val)
    @assert length(A) == length(B)
    #
    #   C ← C + A B
    #
    @inbounds @simd for i in eachindex(A)
        #
        #   C ← C + Ai Bi
        #
        C = mul_add_impl(A[i], B[i], C, tA, tB, dual)
    end

    return C
end

function mul_add_impl!(A::AbstractMatrix, B::AbstractMatrix, C::AbstractMatrix, tA::Val{R}, tB::Val{:N}, dual::Val) where {R}
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
        #   Cj ← A Bj + Cj
        #
        mul_add_impl!(A, Bj, Cj, tA, tB, dual)
    end

    return C
end

function mul_add_impl!(A::AbstractMatrix, B::AbstractMatrix, C::AbstractMatrix, tA::Val{:N}, tB::Val{:C}, dual::Val)
    @assert size(C, 1) == size(A, 1)
    @assert size(C, 2) == size(B, 1)
    @assert size(B, 2) == size(A, 2)

    @inbounds for j in axes(A, 2)
        Aj = @view A[:, j]
        Bj = @view B[:, j]
        #
        #   C ← Aj / Bj & C
        #
        mul_add_impl!(Aj, Bj, C, tA, tB, dual)
    end

    return C
end

function mul_add_impl!(A::StridedMatrix{T}, B::StridedMatrix{T}, C::StridedMatrix{T}, tA::Val{:N}, tB::Val{:N}, dual::Val{:N}) where {T <: NativeTypes}
    matmul!(C, A, B, StaticInt{1}(), StaticInt{1}())
    return
end

function mul_add_impl!(A::AbstractMatrix, B::AbstractVector, C::AbstractVector, tA::Val{:N}, tB::Val, dual::Val)
    @assert length(C) == size(A, 1)
    @assert length(B) == size(A, 2)

    @inbounds for j in axes(A, 2)
        Aj = @view A[:, j]
        Bj =       B[   j]  
        #   
        #   C ← Aj Bj + C
        #   
        mul_add_impl!(Aj, Bj, C, tA, tB, dual)
    end 

    return C
end

function mul_add_impl!(A::AbstractMatrix, B::AbstractVector, C::AbstractVector, tA::Val{:C}, tB::Val{:N}, dual::Val)
    @assert length(C) == size(A, 2)
    @assert length(B) == size(A, 1)

    @inbounds for i in eachindex(C)
        Ci = @view C[   i]
        Ai = @view A[:, i]
        #
        #   Ci ← Ai \ B & Ci
        #
        mul_add_impl!(Ai, B, Ci, tA, tB, dual)
    end

    return C
end

function mul_add_impl!(A::AbstractVector, B::AbstractMatrix, C::AbstractVector, tA::Val{:N}, tB::Val{:N}, dual::Val)
    @assert length(A) == size(B, 1)
    @assert length(C) == size(B, 2)

    @inbounds for j in axes(B, 2)
        Cj = @view C[   j]
        Bj = @view B[:, j]
        #
        #   Cj ← A Bj + Cj
        #
        mul_add_impl!(A, Bj, Cj, tA, tB, dual)
    end

    return C
end

function mul_add_impl!(A::AbstractVector, B::AbstractMatrix, C::AbstractVector, tA::Val{:N}, tB::Val{:C}, dual::Val)
    @assert length(C) == size(B, 1)
    @assert length(A) == size(B, 2)

    @inbounds for j in eachindex(A)
        Aj =       A[   j]
        Bj = @view B[:, j]
        #
        #   C ← Aj / Bj & C
        #
        mul_add_impl!(Aj, Bj, C, tA, tB, dual)
    end

    return C
end

function mul_add_impl!(A::AbstractVector, B::AbstractVector, C::AbstractMatrix, tA::Val{:N}, tB::Val, dual::Val)
    @assert size(C, 1) == length(A)
    @assert size(C, 2) == length(B)

    @inbounds for j in eachindex(B)
        Cj = @view C[:, j]
        Bj =       B[   j]  
        #   
        #   Cj ← A Bj + Cj
        #   
        mul_add_impl!(A, Bj, Cj, tA, tB, dual)
    end 

    return C
end

function mul_add_impl!(A::Number, B::AbstractVector, C::AbstractVector, tA::Val, tB::Val, dual::Val)
    @assert length(C) == length(B)
    #
    #   C ← A B + C
    #
    @inbounds for i in eachindex(C)
        Ci = @view C[i]
        Bi =       B[i]
        #
        #   Ci ← A Bi + Ci
        #
        mul_add_impl!(A, Bi, Ci, tA, tB, dual)
    end

    return C
end

function mul_add_impl!(A::AbstractVector, B::Number, C::AbstractVector, tA::Val, tB::Val, dual::Val)
    @assert length(C) == length(A)
    #
    #   C ← A B + C
    #
    @inbounds for i in eachindex(C)
        Ci = @view C[i]
        Ai =       A[i]
        #
        #   Ci ← Ai B + Ci
        #
        mul_add_impl!(Ai, B, Ci, tA, tB, dual)
    end

    return C
end

function mul_add_impl!(A, B, C::AbstractScalar, tA::Val, tB::Val, dual::Val)
    #
    #   C ← A B + C
    #
    C[] = mul_add_impl(A, B, C[], tA, tB, dual)
    return C
end

# ---------------- #
# Jacobi Algorithm #
# ---------------- #

function jacobi(A::AbstractMatrix, B::AbstractVecOrMat, tA::Val, side::Val; kw...)
    return jacobi!(A, Array(B), tA, side; kw...)
end

function jacobi!(A::AbstractMatrix{T}, B::AbstractVecOrMat{T}, tA::Val, side::Val; kw...) where {T}
    jacobi_impl!(A, B, Array{T}(undef, size(B)), tA, side)
    return B
end

function jacobi_impl!(A::AbstractMatrix{T}, B::AbstractVecOrMat{T}, C::AbstractVecOrMat{T}, tA::Val, side::Val{S}; kw...) where {T, S}
    for _ in 1:size(A, 1)
        copyto!(C, B)

        if S == :L
            #
            #   B ← A C + B
            #
            mul_add_impl!(A, C, B, tA, Val(:N), tA)
        else
            #
            #   B ← C A + B
            #
            mul_add_impl!(C, A, B, Val(:N), tA, tA)
        end
        #
        #   B ≈ C
        #
        all(i -> isapprox(B[i], C[i]; kw...), eachindex(B)) && break
    end

    return
end

# ------------------ #
# Newton's Algorithm #
# ------------------ #

function horner(x::AbstractVector, A::AbstractArray...)
    return horner!(x, map(Array, A)...)
end

function horner!(x::AbstractVector, A::AbstractArray...)
    m = length(A)
    n = length(first(A))

    for i in m - 1:-1:1
        #
        #   Ai ← Ai+1 x + Ai
        #
        mul_add_impl!(reshape(A[i + 1], :, n), x, reshape(A[i], :), Val(:N), Val(:N), Val(:N))
    end

    return A
end

function newton(A::AbstractVector; kw...)
    return Vector(A)
end

function newton(A::AbstractArray...; kw...)
    return newton!(map(Array, A)...; kw...)
end

function newton!(A::AbstractArray{T}...; kw...) where {T}
    n = length(first(A))

    J = Matrix{T}(undef, n, n)
    x = Vector{T}(undef, n)
    #
    #   x ← 0
    #
    fill!(x, zero(T))

    for _ in 1:n + 1
        #
        #   A₁ ←  f(x)
        #   A₂ ← Df(x)
        # 
        horner!(x, A...)
        #
        #   x ≈ A₁
        #
        all(i -> isapprox(x[i], A[1][i]; kw...), 1:n) && break
        #
        #   J ← A₂
        #
        copyto!(J, A[2])
        #
        #   x ← J* A₁
        #
        lmul!(slu!(J), copyto!(x, A[1]))
    end

    return x
end

# -------------------- #
# Sinkhorn's Algorithm #
# -------------------- #

function softmin(beta::Real)
    return a -> softmin(a, beta)
end

function softmin(a::MinPlus, beta::Real)
    return exp(-beta * parent(a))
end

function softmin(A::AbstractArray, beta::Real)
    return map(softmin(beta), A)
end

function sinkhorn(C::AbstractMatrix{T}, A::AbstractVecOrMat, B::AbstractVecOrMat, beta::Real; kw...) where {T <: MinPlus}
    return sinkhorn(slu(softmin(C)), A, B, beta; kw...)
end

function sinkhorn(K::AbstractStar{T}, A::AbstractVector, B::AbstractVector, beta::Real; kw...) where {T <: Real}
    @assert size(K, 1) == length(A)
    @assert size(K, 1) == length(B)

    n = length(A)

    E = Vector{T}(undef, n)
    F = Vector{T}(undef, n)
    G = Vector{T}(undef, n)
    #
    # F ← 1
    # G ← 1
    #
    fill!(F, one(T))
    fill!(G, one(T))

    for _ in 1:1000
        #
        #   E ← F
        #
        copyto!(E, F)
        #
        #   F ← K G \ A
        #
        lmul!(K, copyto!(F, G)) .\= A
        #
        #   G ← Kᵀ F \ B
        #
        rmul!(copyto!(G, F), K) .\= B
        #
        #   F ≈ E
        #
        all(i -> isapprox(F[i], E[i]; kw...), 1:n) && break
    end
    #
    #   E ← 1/β (A log F + B log G)
    #
    @. E = (A * log(F) + B * log(G)) / beta

    return MinPlus(sum(E)) 
end
