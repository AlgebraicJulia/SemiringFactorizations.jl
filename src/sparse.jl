struct SparseStarLU{T, I} <: AbstractStarLU{T}
    symb::SymbolicStarLU{I}
    Rptr::FVector{I}
    Rval::FVector{T}
    Lptr::FVector{I}
    Lval::FVector{T}
    Uval::FVector{T}
end

function Base.show(io::IO, ::MIME"text/plain", fact::T) where {T <: SparseStarLU}
    frt = fact.symb.nFval
    nnz = fact.symb.nRval + fact.symb.nLval + fact.symb.nLval

    print(io, "$T:")
    print(io, "\n  maximum front-size: $frt")
    print(io, "\n  Lnz + Unz: $nnz")
end

function Base.size(F::SparseStarLU)
    n = convert(Int, nov(F.symb.res))
    return (n, n)
end

function slu(
        A::SparseMatrixCSC;
        alg::PermutationOrAlgorithm = DEFAULT_ELIMINATION_ALGORITHM,
        snd::SupernodeType = DEFAULT_SUPERNODE_TYPE,
    )
    return slu(A, alg, snd)
end

function slu(A::SparseMatrixCSC, alg::PermutationOrAlgorithm, snd::SupernodeType)
    return slu(A, SymbolicStarLU(A, alg, snd))
end

function slu(matrix::SparseMatrixCSC{T, I}, symb::SymbolicStarLU{I}) where {T, I <: Integer}
    res = symb.res
    sep = symb.sep
    rel = symb.rel
    chd = symb.chd

    nMptr = symb.nMptr
    nMval = symb.nMval
    nRval = symb.nRval
    nLval = symb.nLval
    nFval = symb.nFval

    nRptr = nv(res) + one(I)

    Mptr = FVector{I}(undef, nMptr)
    Mval = FVector{T}(undef, nMval)
    Rptr = FVector{I}(undef, nRptr)
    Rval = FVector{T}(undef, nRval)
    Lptr = FVector{I}(undef, nRptr)
    Lval = FVector{T}(undef, nLval)
    Uval = FVector{T}(undef, nLval)
    Fval = FVector{T}(undef, nFval * nFval)

    # the LU factor is stored as a block
    # sparse matrix
    #
    #   + - + - -
    #   | R | U ⋯
    #   + - + - -
    #   | L | ⋱
    #   | ⋮ |
    #
    # the R L, and U blocks are stored
    # respectively in the pairs
    #
    #   - (Rptr, Rval)
    #   - (Lptr, Lval)
    #   - (Uptr, Uval)
    #
    # we begin by copying the matrix into this
    # data structure
    A = permute(matrix, symb.ord, symb.ord)

    # copy A into R
    sp_slu_copy_R!(Rptr, Rval, res, A)

    # copy A into L
    sp_slu_copy_L!(Lptr, Lval, res, sep, A) 

    # copy A into U
    sp_slu_copy_U!(Uval, res, sep, A)

    sp_slu_impl!(Mptr, Mval, Rptr, Rval, Lptr,
        Lval, Uval, Fval, res, rel, chd)

    return SparseStarLU(symb, Rptr, Rval, Lptr, Lval, Uval)    
end

function lmul!(A::SparseStarLU, B::AbstractVecOrMat)
    sp_smul_impl!(A, B, Val(:N), Val(:L))
    return B
end

function rmul!(B::AbstractMatrix, A::SparseStarLU)
    sp_smul_impl!(A, B, Val(:N), Val(:R))
    return B
end

function rmul!(B::AbstractRowVector, A::SparseStarLU)
    sp_smul_impl!(A, parent(B), Val(:N), Val(:R))
    return B
end

function ldiv!(A::SparseStarLU, B::AbstractVecOrMat)
    sp_smul_impl!(A, B, Val(:C), Val(:L))
    return B
end

function rdiv!(B::AbstractMatrix, A::SparseStarLU)
    sp_smul_impl!(A, B, Val(:C), Val(:R))
    return B
end

function rdiv!(B::AbstractRowVector, A::SparseStarLU)
    sp_smul_impl!(A, parent(B), Val(:C), Val(:R))
    return B
end

function sp_smul_impl!(A::SparseStarLU{T, I}, B::AbstractVecOrMat, tA::Val, side::Val{S}) where {T, I <: Integer, S}
    if B isa AbstractVector
        neqn = convert(I, length(B))
        nrhs = one(I)
    elseif S == :L
        neqn = convert(I, size(B, 1))
        nrhs = convert(I, size(B, 2))
    else
        neqn = convert(I, size(B, 2))
        nrhs = convert(I, size(B, 1))
    end

    ord = A.symb.ord
    res = A.symb.res
    rel = A.symb.rel
    chd = A.symb.chd

    Rptr = A.Rptr
    Rval = A.Rval
    Lptr = A.Lptr
    Lval = A.Lval
    Uval = A.Uval

    nMptr = A.symb.nMptr
    nNval = A.symb.nNval
    nFval = A.symb.nFval

    Mptr = FVector{I}(undef, nMptr)
    Mval = FVector{T}(undef, nNval * nrhs)
    Fval = FVector{T}(undef, nFval * nrhs)

    if B isa AbstractVector
        C = FVector{T}(undef, neqn)
    elseif S == :L
        C = FMatrix{T}(undef, neqn, nrhs)
    else
        C = FMatrix{T}(undef, nrhs, neqn)
    end

    if B isa AbstractVector
        copyto!(C, view(B, ord))
    elseif S == :L
        copyto!(C, view(B, ord, :))
    else
        copyto!(C, view(B, :, ord))
    end

    sp_smul_impl!(C, Mptr, Mval, Rptr, Rval, Lptr,
        Lval, Uval, Fval, res, rel, chd, tA, side)

    if B isa AbstractVector
        copyto!(view(B, ord), C)
    elseif S == :L
        copyto!(view(B, ord, :), C)
    else
        copyto!(view(B, :, ord), C)
    end

    return
end

"""
    mtlmul!(A::SparseStarLU, B::AbstractMatrix)

A multi-threaded version of [`lmul!`](@ref).
"""
function mtlmul!(A::SparseStarLU, B::AbstractMatrix)
    sp_mtsmul_impl!(A, B, Val(:N), Val(:L))
    return B
end

"""
    mtrmul!(B::AbstractMatrix, A::SparseStarLU)

A multi-threaded version of [`rmul!`](@ref).
"""
function mtrmul!(B::AbstractMatrix, A::SparseStarLU)
    sp_mtsmul_impl!(A, B, Val(:N), Val(:R))
    return B
end

"""
    mtldiv!(A::SparseStarLU, B::AbstractMatrix)

A multi-threaded version of [`ldiv!`](@ref).
"""
function mtldiv!(A::SparseStarLU, B::AbstractMatrix)
    sp_mtsmul_impl!(A, B, Val(:C), Val(:L))
    return B
end

"""
    mtrdiv!(B::AbstractMatrix, A::SparseStarLU)

A multi-threaded version of [`rdiv!`](@ref).
"""
function mtrdiv!(B::AbstractMatrix, A::SparseStarLU)
    sp_mtsmul_impl!(A, B, Val(:C), Val(:R))
    return B
end

function sp_mtsmul_impl!(A::SparseStarLU{T, I}, B::AbstractMatrix, tA::Val, side::Val{S}) where {T, I <: Integer, S}
    if S == :L
        neqn = convert(I, size(B, 1))
        nrhs = convert(I, size(B, 2))
    else
        neqn = convert(I, size(B, 2))
        nrhs = convert(I, size(B, 1))
    end

    ord = A.symb.ord
    res = A.symb.res
    rel = A.symb.rel
    chd = A.symb.chd

    Rptr = A.Rptr
    Rval = A.Rval
    Lptr = A.Lptr
    Lval = A.Lval
    Uval = A.Uval

    nMptr = A.symb.nMptr
    nNval = A.symb.nNval
    nFval = A.symb.nFval

    blocksize = convert(I, max(32, div(nrhs, 4nthreads())))

    @threads for strt in one(I):blocksize:nrhs
        size = min(blocksize, nrhs - strt + one(I))
        stop = strt + size - one(I)

        if S == :L
            C = FMatrix{T}(undef, neqn, size)
        else
            C = FMatrix{T}(undef, size, neqn)
        end

        if S == :L
            copyto!(C, view(B, ord, strt:stop))
        else
            copyto!(C, view(B, strt:stop, ord))
        end

        Mptr = FVector{I}(undef, nMptr)
        Mval = FVector{T}(undef, nNval * size)
        Fval = FVector{T}(undef, nFval * size)

        sp_smul_impl!(C, Mptr, Mval, Rptr, Rval, Lptr,
            Lval, Uval, Fval, res, rel, chd, tA, side)

        if S == :L
            copyto!(view(B, ord, strt:stop), C)
        else
            copyto!(view(B, strt:stop, ord), C)
        end
    end

    return
end

function sp_slu_copy_R!(
        Rptr::AbstractVector{I},
        Rval::AbstractVector{T},
        res::AbstractGraph{I},
        A::AbstractMatrix{T},
    ) where {T, I <: Integer}
    @assert nv(res) < length(Rptr)
    @assert nov(res) == size(A, 1)
    @assert nov(res) == size(A, 2)
    pj = zero(I); nwr = one(I)

    for j in vertices(res)
        Rptr[j] = pj + one(I)

        swr = nwr
        nwr = pointers(res)[j + one(I)]

        for vr in swr:nwr - one(I)
            wr = swr

            for pa in nzrange(A, vr)
                wa = rowvals(A)[pa]
                wa < swr && continue
                wa < nwr || break

                while wr < wa
                    pj += one(I); Rval[pj] = zero(T)
                    wr += one(I)
                end

                pj += one(I); Rval[pj] = nonzeros(A)[pa]
                wr += one(I)
            end

            while wr < nwr
                pj += one(I); Rval[pj] = zero(T)
                wr += one(I)
            end
        end 
    end

    Rptr[nv(res) + one(I)] = pj + one(I)
    return
end

function sp_slu_copy_L!(
        Lptr::AbstractVector{I},
        Lval::AbstractVector{T},
        res::AbstractGraph{I},
        sep::AbstractGraph{I},
        A::AbstractMatrix{T},
    ) where {T, I <: Integer}
    @assert nv(res) < length(Lptr)
    @assert nv(res) == nv(sep)
    @assert nov(res) == size(A, 1)
    @assert nov(res) == size(A, 2)
    @assert nov(res) == nov(sep)
    pj = zero(I); npr = one(I)

    for j in vertices(res)
        Lptr[j] = pj + one(I)

        spr = npr
        npr = pointers(sep)[j + one(I)]
        spr >= npr && continue

        swr = targets(sep)[spr]
        nwr = targets(sep)[npr - one(I)] + one(I) 

        for vr in neighbors(res, j)
            pr = spr

            for pa in nzrange(A, vr)
                wr = targets(sep)[pr]
                wa = rowvals(A)[pa]
                wa < swr && continue
                wa < nwr || break

                while wr < wa
                    pj += one(I); Lval[pj] = zero(T)
                    pr += one(I); wr = targets(sep)[pr]
                end

                pj += one(I); Lval[pj] = nonzeros(A)[pa]
                pr += one(I)
            end

            while pr < npr
                pj += one(I); Lval[pj] = zero(T)
                pr += one(I)
            end
        end
    end

    Lptr[nv(res) + one(I)] = pj + one(I)
    return
end

function sp_slu_copy_U!(
        Uval::AbstractVector{T},
        res::AbstractGraph{I},
        sep::AbstractGraph{I},
        A::AbstractMatrix{T},
    ) where {T, I <: Integer}
    @assert nv(res) == nv(sep)
    @assert nov(res) == size(A, 1)
    @assert nov(res) == size(A, 2)
    @assert nov(res) == nov(sep)
    pj = zero(I); nwr = one(I)

    for j in vertices(res)
        swr = nwr
        nwr = pointers(res)[j + one(I)]

        for vr in neighbors(sep, j)
            wr = swr

            for pa in nzrange(A, vr)
                wa = rowvals(A)[pa]
                wa < swr && continue
                wa < nwr || break

                while wr < wa
                    pj += one(I); Uval[pj] = zero(T)
                    wr += one(I)
                end

                pj += one(I); Uval[pj] = nonzeros(A)[pa]
                wr += one(I)
            end

            while wr < nwr
                pj += one(I); Uval[pj] = zero(T)
                wr += one(I)
            end
        end 
    end

    return
end

function sp_slu_impl!(
        Mptr::AbstractVector{I},
        Mval::AbstractVector{T},
        Rptr::AbstractVector{I},
        Rval::AbstractVector{T},
        Lptr::AbstractVector{I},
        Lval::AbstractVector{T},
        Uval::AbstractVector{T},
        Fval::AbstractVector{T},
        res::AbstractGraph{I},
        rel::AbstractGraph{I}, 
        chd::AbstractGraph{I},
    ) where {T, I <: Integer}
    ns = zero(I); Mptr[one(I)] = one(I)

    for j in vertices(res)
        ns = sp_slu_loop!(Mptr, Mval, Rptr, Rval, Lptr,
            Lval, Uval, Fval, res, rel, chd, ns, j)
    end

    return
end

function sp_slu_loop!(
        Mptr::AbstractVector{I},
        Mval::AbstractVector{T},
        Rptr::AbstractVector{I},
        Rval::AbstractVector{T},
        Lptr::AbstractVector{I},
        Lval::AbstractVector{T},
        Uval::AbstractVector{T},
        Fval::AbstractVector{T},
        res::AbstractGraph{I},
        rel::AbstractGraph{I}, 
        chd::AbstractGraph{I},
        ns::I,
        j::I,
    ) where {T, I <: Integer}
    # nn is the size of the residual at node j
    #
    #     nn = | res(j) |
    #
    nn = eltypedegree(res, j)

    # na is the size of the separator at node j
    #
    #     na = | sep(j) |
    #
    na = eltypedegree(rel, j)

    # nj is the size of the bag at node j
    #
    #     nj = | bag(j) |
    #
    nj = nn + na

    # F is the frontal matrix at node j
    F = reshape(view(Fval, oneto(nj * nj)), nj, nj)

    #
    #           nn  na
    #     F = [ F₁₁ F₁₂ ] nn
    #         [ F₂₁ F₂₂ ] na
    #
    F₁₁ = view(F, oneto(nn),      oneto(nn))
    F₂₁ = view(F, nn + one(I):nj, oneto(nn))
    F₁₂ = view(F, oneto(nn),      nn + one(I):nj)
    F₂₂ = view(F, nn + one(I):nj, nn + one(I):nj)

    # B is part of the LU factor
    #
    #          res(j) sep(j)
    #     B = [ B₁₁    B₁₂  ] res(j)
    #         [ B₂₁         ] sep(j)
    #
    Rp = Rptr[j]
    Lp = Lptr[j]
    B₁₁ = reshape(view(Rval, Rp:Rp + nn * nn - one(I)), nn, nn)
    B₂₁ = reshape(view(Lval, Lp:Lp + nn * na - one(I)), na, nn)
    B₁₂ = reshape(view(Uval, Lp:Lp + nn * na - one(I)), nn, na)

    # copy B into F
    #
    #     F₁₁ ← B₁₁
    #     F₂₁ ← B₂₁
    #     F₁₂ ← B₁₂
    #
    copyto!(F₁₁, B₁₁)
    copyto!(F₂₁, B₂₁)
    copyto!(F₁₂, B₁₂)
    #
    #    F₂₂ ← 0
    #
    fill!(F₂₂, zero(T))

    for i in Iterators.reverse(neighbors(chd, j))
        sp_slu_add_update!(F, Mptr, Mval, rel, ns, i)
        ns -= one(I)
    end

    # copy F₁ into B
    #
    #     B₁₁ ← F₁₁
    #     B₂₁ ← F₂₁
    #     B₁₂ ← F₁₂
    #
    copyto!(B₁₁, F₁₁)
    copyto!(B₂₁, F₂₁)
    copyto!(B₁₂, F₁₂)

    # factorize B₁₁ as
    #
    #   B₁₁* = U₁₁* L₁₁*
    #
    # and store
    #
    #   B₁₁ ← L₁₁ + U₁₁
    #
    slu_impl!(B₁₁)

    if ispositive(na)
        ns += one(I)

        # B₂₂ is the na × na update matrix for node j
        strt = Mptr[ns]
        stop = Mptr[ns + one(I)] = strt + na * na
        B₂₂ = reshape(view(Mval, strt:stop - one(I)), na, na)

        #
        #   M₂₂ ← F₂₂
        #
        B₂₂ .= F₂₂

        #
        #   B₂₁ ← B₂₁ U₁₁*
        #   
        smul_impl!(B₁₁, B₂₁, Val(:N), Val(:U), Val(:R))

        #
        #   B₁₂ ← L₁₁* B₁₂
        #   
        smul_impl!(B₁₁, B₁₂, Val(:N), Val(:L), Val(:L))

        #
        #   B₂₂ ← B₂₂ + B₂₁ B₁₂
        #
        mul_impl!(B₂₂, B₂₁, B₁₂, Val(:N), Val(:N))
    end
 
    return ns
end

function sp_smul_impl!(
        C::AbstractVecOrMat{T},
        Mptr::AbstractVector{I},
        Mval::AbstractVector{T},
        Rptr::AbstractVector{I},
        Rval::AbstractVector{T},
        Lptr::AbstractVector{I},
        Lval::AbstractVector{T},
        Uval::AbstractVector{T},
        Fval::AbstractVector{T},
        res::AbstractGraph{I},
        rel::AbstractGraph{I}, 
        chd::AbstractGraph{I},
        tA::Val,
        side::Val,
    ) where {T, I <: Integer}
    ns = zero(I); Mptr[one(I)] = one(I)

    # forward substitution loop
    for j in vertices(res)
        ns = sp_smul_fwd_loop!(C, Mptr, Mval, Rptr, Rval, Lptr,
            Lval, Uval, Fval, res, rel, chd, ns, j, tA, side)
    end

    # backward substitution loop
    for j in reverse(vertices(res))
        ns = sp_smul_bwd_loop!(C, Mptr, Mval, Rptr, Rval, Lptr,
            Lval, Uval, Fval, res, rel, chd, ns, j, tA, side)
    end

    return
end

function sp_smul_fwd_loop!(
        C::AbstractVecOrMat{T},
        Mptr::AbstractVector{I},
        Mval::AbstractVector{T},
        Rptr::AbstractVector{I},
        Rval::AbstractVector{T},
        Lptr::AbstractVector{I},
        Lval::AbstractVector{T},
        Uval::AbstractVector{T},
        Fval::AbstractVector{T},
        res::AbstractGraph{I},
        rel::AbstractGraph{I}, 
        chd::AbstractGraph{I},
        ns::I,
        j::I,
        tA::Val{R},
        side::Val{S},
    ) where {T, I, R, S}
    #
    #   nrhs is the number of columns in C
    #
    if C isa AbstractVector
        nrhs = one(I)
    elseif S == :L
        nrhs = convert(I, size(C, 2))
    else
        nrhs = convert(I, size(C, 1))
    end

    # nn is the size of the residual at node j
    #
    #   nn = | res(j) |
    #
    nn = eltypedegree(res, j)

    # na is the size of the separator at node j.
    #
    #   na = | sep(j) |
    #
    na = eltypedegree(rel, j)

    # nj is the size of the bag at node j
    #
    #   nj = | bag(j) |
    #
    nj = nn + na    

    # F is the frontal matrix at node j
    if C isa AbstractVector
        F = view(Fval, oneto(nj))
    elseif S == :L
        F = reshape(view(Fval, oneto(nj * nrhs)), nj, nrhs)
    else
        F = reshape(view(Fval, oneto(nj * nrhs)), nrhs, nj)
    end

    #
    #        nrhs
    #   F = [ F₁ ] nn
    #     = [ F₂ ] na
    #
    if C isa AbstractVector
        F₁ = view(F, oneto(nn))
        F₂ = view(F, nn + one(I):nj)
    elseif S == :L
        F₁ = view(F, oneto(nn),      oneto(nrhs))
        F₂ = view(F, nn + one(I):nj, oneto(nrhs))
    else
        F₁ = view(F, oneto(nrhs), oneto(nn))
        F₂ = view(F, oneto(nrhs), nn + one(I):nj)
    end

    # B is part of the L factor
    #
    #        res(j)
    #   B = [ B₁₁  ] res(j)
    #       [ B₂₁  ] sep(j)
    #
    Rp = Rptr[j]
    Lp = Lptr[j]
    B₁₁ = reshape(view(Rval, Rp:Rp + nn * nn - one(I)), nn, nn)

    if R == :N && S == :L || R == :C && S == :R
        B₂₁ = reshape(view(Lval, Lp:Lp + nn * na - one(I)), na, nn)
    else
        B₂₁ = reshape(view(Uval, Lp:Lp + nn * na - one(I)), nn, na)
    end

    # C₁ is part of the right-hand side
    #
    #        nrhs
    #   C = [ C₁ ] res(j)
    #
    if C isa AbstractVector
        C₁ = view(C, neighbors(res, j))
    elseif S == :L
        C₁ = view(C, neighbors(res, j), oneto(nrhs))
    else
        C₁ = view(C, oneto(nrhs), neighbors(res, j))
    end

    # copy C into F
    #
    #   F₁ ← C₁
    #
    copyto!(F₁, C₁)
    #
    #   F₂ ← 0
    #
    if R == :N
        fill!(F₂, zero(T))
    else
        fill!(F₂, typemax(T))
    end

    for i in Iterators.reverse(neighbors(chd, j))
        sp_smul_fwd_update!(F, Mptr, Mval, rel, ns, i, tA, side)
        ns -= one(I)
    end

    # copy F into C
    #   
    #   C₁ ← F₁
    #
    copyto!(C₁, F₁)
    #
    #   C₁ ← B₁₁* C₁
    #
    if R == :N && S == :L || R == :C && S == :R
        smul_impl!(B₁₁, C₁, tA, Val(:L), side)
    else
        smul_impl!(B₁₁, C₁, tA, Val(:U), side)
    end

    if ispositive(na)
        ns += one(I)

        # C₂ is the update matrix at node j
        strt = Mptr[ns]
        stop = Mptr[ns + one(I)] = strt + na * nrhs

        if C isa AbstractVector
            C₂ = view(Mval, strt:stop - one(I))
        elseif S == :L
            C₂ = reshape(view(Mval, strt:stop - one(I)), na, nrhs)
        else
            C₂ = reshape(view(Mval, strt:stop - one(I)), nrhs, na)
        end
        #
        #   C₂ ← F₂
        #
        copyto!(C₂, F₂)
        #
        #   C₂ ← C₂ + B₂₁ C₁
        #
        if S == :L
            mul_impl!(C₂, B₂₁, C₁, tA, Val(:N))
        else
            mul_impl!(C₂, C₁, B₂₁, Val(:N), tA)
        end
    end

    return ns
end

function sp_smul_bwd_loop!(
        C::AbstractVecOrMat{T},
        Mptr::AbstractVector{I},
        Mval::AbstractVector{T},
        Rptr::AbstractVector{I},
        Rval::AbstractVector{T},
        Lptr::AbstractVector{I},
        Lval::AbstractVector{T},
        Uval::AbstractVector{T},
        Fval::AbstractVector{T},
        res::AbstractGraph{I},
        rel::AbstractGraph{I}, 
        chd::AbstractGraph{I},
        ns::I,
        j::I,
        tA::Val{R},
        side::Val{S},
    ) where {T, I <: Integer, R, S}
    #
    #   nrhs is the number of columns in C
    #
    if C isa AbstractVector
        nrhs = one(I)
    elseif S == :L
        nrhs = convert(I, size(C, 2))
    else
        nrhs = convert(I, size(C, 1))
    end

    # nn is the size of the residual at node j
    #
    #   nn = | res(j) |
    #
    nn = eltypedegree(res, j)

    # na is the size of the separator at node j.
    #
    #   na = | sep(j) |
    #
    na = eltypedegree(rel, j)

    # nj is the size of the bag at node j
    #
    #   nj = | bag(j) |
    #
    nj = nn + na    

    # F is the frontal matrix at node j
    if C isa AbstractVector
        F = view(Fval, oneto(nj))
    elseif S == :L
        F = reshape(view(Fval, oneto(nj * nrhs)), nj, nrhs)
    else
        F = reshape(view(Fval, oneto(nj * nrhs)), nrhs, nj)
    end

    #
    #        nrhs
    #   F = [ F₁ ] nn
    #     = [ F₂ ] na
    #
    if C isa AbstractVector
        F₁ = view(F, oneto(nn))
        F₂ = view(F, nn + one(I):nj)
    elseif S == :L
        F₁ = view(F, oneto(nn),      oneto(nrhs))
        F₂ = view(F, nn + one(I):nj, oneto(nrhs))
    else
        F₁ = view(F, oneto(nrhs), oneto(nn))
        F₂ = view(F, oneto(nrhs), nn + one(I):nj)
    end

    # B is part of the U factor
    #
    #        res(j) sep(j)
    #   B = [ B₁₁    B₁₂  ] res(j)
    #
    Rp = Rptr[j]
    Lp = Lptr[j]
    B₁₁ = reshape(view(Rval, Rp:Rp + nn * nn - one(I)), nn, nn)

    if R == :N && S == :L || R == :C && S == :R
        B₁₂ = reshape(view(Uval, Lp:Lp + nn * na - one(I)), nn, na)
    else
        B₁₂ = reshape(view(Lval, Lp:Lp + nn * na - one(I)), na, nn)
    end

    # C₁ is part of the right-hand side
    #
    #        nrhs
    #   C = [ C₁ ] res(j)
    #
    if C isa AbstractVector
        C₁ = view(C, neighbors(res, j))
    elseif S == :L
        C₁ = view(C, neighbors(res, j), oneto(nrhs))
    else
        C₁ = view(C, oneto(nrhs), neighbors(res, j))
    end

    if ispositive(na)
        # C₂ is the update matrix at node j
        strt = Mptr[ns]

        if C isa AbstractVector
            C₂ = view(Mval, strt:strt + na - one(I))
        elseif S == :L
            C₂ = reshape(view(Mval, strt:strt + na * nrhs - one(I)), na, nrhs)
        else
            C₂ = reshape(view(Mval, strt:strt + na * nrhs - one(I)), nrhs, na)
        end

        ns -= one(I)
        #
        #   C₁ ← C₁ + B₁₂ C₂
        #
        if S == :L
            mul_impl!(C₁, B₁₂, C₂, tA, Val(:N))
        else
            mul_impl!(C₁, C₂, B₁₂, Val(:N), tA)
        end
        #
        #   F₂ ← M₂
        #
        copyto!(F₂, C₂)
    end

    #
    #   C₁ ← B₁₁* C₁
    #
    if R == :N && S == :L || R == :C && S == :R
        smul_impl!(B₁₁, C₁, tA, Val(:U), side)
    else
        smul_impl!(B₁₁, C₁, tA, Val(:L), side)
    end

    # copy C into F
    #
    #   F₁ ← C₁
    #
    copyto!(F₁, C₁)

    for i in neighbors(chd, j)
        ns += one(I)
        sp_smul_bwd_update!(F, Mptr, Mval, rel, ns, i, side)
    end

    return ns
end

function sp_slu_add_update!(
        F::AbstractMatrix{T},
        ptr::AbstractVector{I},
        val::AbstractVector{T},
        rel::AbstractGraph{I},
        ns::I,
        i::I,
    ) where {T, I}
    # na is the size of the separator at node i
    #
    #   na = | sep(i) |
    #
    na = eltypedegree(rel, i)

    # inj is the subset inclusion
    #
    #   inj: sep(i) → bag(parent(i))
    #
    inj = neighbors(rel, i)

    # B is the na × na update matrix at node i
    strt = ptr[ns]
    B = reshape(view(val, strt:strt + na * na - one(I)), na, na)

    #
    #   F ← F + inj B injᵀ
    #
    @inbounds for w in oneto(na)
        iw = inj[w]

        for v in oneto(na)
            F[inj[v], iw] += B[v, w]
        end
    end

    return
end

function sp_smul_fwd_update!(
        F::AbstractVecOrMat{T},
        ptr::AbstractVector{I},
        val::AbstractVector{T},
        rel::AbstractGraph{I},
        ns::I,
        i::I,
        tA::Val{R},
        side::Val{S},
    ) where {T, I <: Integer, R, S}
    #
    #   nrhs is the number of columns in F
    #
    if F isa AbstractVector
        nrhs = one(I)
    elseif S == :L
        nrhs = convert(I, size(F, 2))
    else
        nrhs = convert(I, size(F, 1))
    end

    # na is the size of the separator at node i
    #
    #   na = | sep(i) |
    #
    na = eltypedegree(rel, i)

    # inj is the subset inclusion
    #
    #   inj: sep(i) → bag(parent(i))
    #
    inj = neighbors(rel, i)

    # C is the na × nrhs update matrix at node i
    strt = ptr[ns]

    if F isa AbstractVector
        C = view(val, strt:strt + na - one(I))
    elseif S == :L
        C = reshape(view(val, strt:strt + na * nrhs - one(I)), na, nrhs)
    else
        C = reshape(view(val, strt:strt + na * nrhs - one(I)), nrhs, na)
    end

    #
    #   F ← F + inj C
    #
    if F isa AbstractVector
        @inbounds for v in oneto(na)
            if R == :N
                F[inj[v]] += C[v]
            else
                F[inj[v]] = inf(F[inj[v]], C[v])
            end
        end
    elseif S == :L
        @inbounds for w in oneto(nrhs), v in oneto(na)
            if R == :N
                F[inj[v], w] += C[v, w]
            else
                F[inj[v], w] = inf(F[inj[v], w], C[v, w])
            end
        end
    else
        @inbounds for v in oneto(na)
            iv = inj[v]

            for w in oneto(nrhs)
                if R == :N
                    F[w, iv] += C[w, v]
                else
                    F[w, iv] = inf(F[w, iv], C[w, v])
                end
            end
        end
    end

    return
end

function sp_smul_bwd_update!(
        F::AbstractVecOrMat{T},
        ptr::AbstractVector{I},
        val::AbstractVector{T},
        rel::AbstractGraph{I},
        ns::I,
        i::I,
        side::Val{S},
    ) where {T, I <: Integer, S}
    #
    #   nrhs is the number of columns in F
    #
    if F isa AbstractVector
        nrhs = one(I)
    elseif S == :L
        nrhs = convert(I, size(F, 2))
    else
        nrhs = convert(I, size(F, 1))
    end

    # na is the size of the separator at node i
    #
    #   na = | sep(i) |
    #
    na = eltypedegree(rel, i)

    # inj is the subset inclusion
    #
    #   inj: sep(i) → bag(parent(i))
    #
    inj = neighbors(rel, i)

    # C is the na × nrhs update matrix at node i
    strt = ptr[ns]
    stop = ptr[ns + one(I)] = strt + na * nrhs

    if F isa AbstractVector
        C = view(val, strt:stop - one(I))
    elseif S == :L
        C = reshape(view(val, strt:stop - one(I)), na, nrhs)
    else
        C = reshape(view(val, strt:stop - one(I)), nrhs, na)
    end

    #
    #   C ← injᵀ F
    #
    if F isa AbstractVector
        @inbounds for v in oneto(na)
            C[v] = F[inj[v]]
        end
    elseif S == :L
        @inbounds for w in oneto(nrhs), v in oneto(na)
            C[v, w] = F[inj[v], w]
        end
    else
        @inbounds for v in oneto(na)
            iv = inj[v]

            for w in oneto(nrhs)
                C[w, v] = F[w, iv]
            end
        end
    end

    return
end

# --------------------- #
# Matrix Multiplication #
# --------------------- #

function mul_impl!(C::AbstractVector, A::AbstractMatrix, B::SparseColumnCSC, tA::Val{:N}, tB::Val)
    @assert length(C) == size(A, 1)
    @assert length(B) == size(A, 2)
    #
    #   B = Dj
    #
    D, j = unpack(B)
    #
    #   C ← C + A B
    #
    @inbounds for p in nzrange(D, j)
        Ai = @view A[:, rowvals(D)[p]]
        #   
        #   C ← C + Ai Bi
        #
         mul_impl!(C, Ai, nonzeros(D)[p], tA, tB)
    end

    return
end

function mul_impl!(C::AbstractMatrix, A::AbstractVector, B::SparseColumnCSC, tA::Val{:N}, tB::Val)
    @assert size(C, 1) == length(A)
    @assert size(C, 2) == length(B)
    #
    #   B = Dj
    #
    D, j = unpack(B)
    #
    #   C ← C + A B
    #
    @inbounds for p in nzrange(D, j)
        Ci = @view C[:, rowvals(D)[p]]
        #   
        #   Ci ← Ci + A Bi
        #   
        mul_impl!(Ci, A, nonzeros(D)[p], tA, tB)
    end

    return
end

function mul_impl!(C::AbstractVector{T}, A::T, B::SparseColumnCSC{T}, tA::Val{:N}, tB::Val) where {T}
    @assert length(C) == length(B)
    #
    #   B = Dj
    #
    D, j = unpack(B)
    #
    #   C ← C + A B
    #
    @inbounds for p in nzrange(D, j)
        Ci = @view C[rowvals(D)[p]]
        #
        #   Ci ← Ci + Ai B
        #
        mul_impl!(Ci, A, nonzeros(D)[p], tA, tB)
    end

    return
end

function mul_impl!(C::AbstractVector{T}, A::SparseColumnCSC{T}, B::T, tA::Val, tB::Val{:N}) where {T}
    @assert length(C) == length(A)
    #
    #   A = Dj
    #
    D, j = unpack(A)
    #
    #   C ← C + A B
    #
    @inbounds for p in nzrange(D, j)
        Ci = @view C[rowvals(D)[p]]
        #
        #   Ci ← Ci + Ai B
        #
        mul_impl!(Ci, nonzeros(D)[p], B, tA, tB)
    end

    return
end

function Semirings.fma(A::SparseColumnCSC{T}, B::AbstractVector{T}, C::T) where {T}
    @assert length(A) == length(B)
    #
    #   A = Dj
    #
    D, j = unpack(A)
    #
    #   C ← C + A B
    #
    @inbounds for p in nzrange(D, j)
        Bi = B[rowvals(D)[p]]
        #
        #   C ← C + Ai Bi
        #
        C = fma(nonzeros(D)[p], Bi, C)
    end

    return C
end

function Semirings.fma(A::AbstractVector{T}, B::SparseColumnCSC{T}, C::T) where {T}
    @assert length(A) == length(B)
    #
    #   B = Dj
    #
    D, j = unpack(B)
    #
    #   C ← C + A B
    #
    @inbounds for p in nzrange(D, j)
        Ai = A[rowvals(D)[p]]
        #
        #   C ← C + Ai Bi
        #
        C = fma(Ai, nonzeros(D)[p], C)
    end

    return C
end

function Semirings.fli(A::SparseColumnCSC{T}, B::AbstractVector{T}, C::T) where {T}
    @assert length(A) == length(B)
    #
    #   A = Dj
    #
    D, j = unpack(A)
    #
    #   C ← C ∧ A \ B
    #
    @inbounds for p in nzrange(D, j)
        Bi = B[rowvals(D)[p]]
        #
        #   C ← C ∧ Ai \ Bi
        #
        C = fli(nonzeros(D)[p], Bi, C)
    end

    return C
end

function Semirings.fri(A::AbstractVector{T}, B::SparseColumnCSC{T}, C::T) where {T}
    @assert length(A) == length(B)
    #
    #   B = Dj
    #
    D, j = unpack(B)
    #
    #   C ← C ∧ A / B
    #
    @inbounds for p in nzrange(D, j)
        Ai = A[rowvals(D)[p]]
        #
        #   C ← C ∧ Ai / Bi
        #
        C = fri(Ai, nonzeros(D)[p], C)
    end

    return C
end

# ------- #
# Infimum #
# ------- #

function Semirings.inf(A::SparseMatrixCSC{T, I}, B::SparseMatrixCSC{T, I}) where {T, I}
    @assert size(A) == size(B)

    m, n = size(A)
    C = spzeros(T, m, n)

    k = min(nnz(A), nnz(B))
    resize!(rowvals(C), k)
    resize!(nonzeros(C), k)

    c = zero(I)

    for j in 1:n
        C.colptr[j] = c + one(I)

        a = A.colptr[j]; astop = A.colptr[j + 1] - 1
        b = B.colptr[j]; bstop = B.colptr[j + 1] - 1

        while a <= astop && b <= bstop
            ai = rowvals(A)[a]
            bi = rowvals(B)[b]

            if ai < bi
                a += one(I)
            elseif ai > bi
                b += one(I)
            else
                ax = nonzeros(A)[a]
                bx = nonzeros(B)[b]

                a += one(I)
                b += one(I)
                c += one(I)

                rowvals(C)[c] = ai
                nonzeros(C)[c] = ax ∧ bx
            end
        end
    end

    C.colptr[n + 1] = c + one(I)
    resize!(rowvals(C), c)
    resize!(nonzeros(C), c)
    return C
end
