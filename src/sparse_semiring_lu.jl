"""
    SparseSemiringLU{T, I}

An LU factorization of a sparse semiring-
valued matrix.
"""
struct SparseSemiringLU{T, I}
    symb::SymbolicSemiringLU{I}
    Rptr::FVector{I}
    Rval::FVector{T}
    Lptr::FVector{I}
    Lval::FVector{T}
    Uval::FVector{T}
end

function Base.size(F::SparseSemiringLU)
    n = convert(Int, nov(F.symb.res))
    return (n, n)
end

function Base.show(io::IO, ::MIME"text/plain", fact::T) where {T <: SparseSemiringLU}
    frt = fact.symb.nFval
    nnz = fact.symb.nRval + fact.symb.nLval + fact.symb.nLval

    print(io, "$T:")
    print(io, "\n  maximum front-size: $frt")
    print(io, "\n  Lnz + Unz: $nnz")
end

function slu(
        A::SparseMatrixCSC;
        alg::PermutationOrAlgorithm = DEFAULT_ELIMINATION_ALGORITHM,
        snd::SupernodeType = DEFAULT_SUPERNODE_TYPE,
    )
    return slu(A, alg, snd)
end

function slu(A::SparseMatrixCSC, alg::PermutationOrAlgorithm, snd::SupernodeType)
    return slu(A, SymbolicSemiringLU(A, alg, snd))
end

function slu(matrix::SparseMatrixCSC{T, I}, symb::SymbolicSemiringLU{I}) where {T, I <: Integer}
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
    sslu_copy_R!(Rptr, Rval, res, A)

    # copy A into L
    sslu_copy_L!(Lptr, Lval, res, sep, A) 

    # copy A into U
    sslu_copy_U!(Uval, res, sep, A)

    sslu_impl!(Mptr, Mval, Rptr, Rval, Lptr,
        Lval, Uval, Fval, res, rel, chd)

    return SparseSemiringLU(symb, Rptr, Rval, Lptr, Lval, Uval)    
end

function sinv(
        A::SparseMatrixCSC;
        alg::PermutationOrAlgorithm = DEFAULT_ELIMINATION_ALGORITHM,
        snd::SupernodeType = DEFAULT_SUPERNODE_TYPE,
    )
    return sinv(A, alg, snd)
end

function sinv(A::SparseMatrixCSC, alg::PermutationOrAlgorithm, snd::SupernodeType)
    return sinv(slu(A, alg, snd))
end

function sinv(A::SparseSemiringLU{T}) where {T}
    B = zeros(T, size(A))
    B[diagind(B)] .= one(T)
    return sldiv!(A, B)
end

function mtsinv(
        A::SparseMatrixCSC;
        alg::PermutationOrAlgorithm = DEFAULT_ELIMINATION_ALGORITHM,
        snd::SupernodeType = DEFAULT_SUPERNODE_TYPE,
    )
    return mtsinv(A, alg, snd)
end

function mtsinv(A::SparseMatrixCSC, alg::PermutationOrAlgorithm, snd::SupernodeType)
    return mtsinv(A, slu(A, alg, snd))
end

function mtsinv(A::SparseSemiringLU{T}) where {T}
    B = zeros(T, size(A))
    B[diagind(B)] .= one(T)
    return mtsldiv!(A, B)
end

function sldiv!(A::SparseSemiringLU{T, I}, B::AbstractArray) where {T, I <: Integer}
    neqn = convert(I, size(B, 1))
    nrhs = convert(I, size(B, 2))

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

    C  = FMatrix{T}(undef, neqn, nrhs)
    C .= view(B, ord, :)

    ssldiv_impl!(C, Mptr, Mval, Rptr, Rval, Lptr,
        Lval, Uval, Fval, res, rel, chd)

    view(B, ord, :) .= C
    return B
end

function mtsldiv!(A::SparseSemiringLU{T, I}, B::AbstractArray) where {T, I <: Integer}
    neqn = convert(I, size(B, 1))
    nrhs = convert(I, size(B, 2))

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

    D = FMatrix{T}(undef, neqn, nrhs)

    blocksize = convert(I, max(32, div(nrhs, 4nthreads())))

    @threads for strt in one(I):blocksize:nrhs
        size = min(blocksize, nrhs - strt + one(I))
        stop = strt + size - one(I)

        C  = view(D, oneto(neqn), strt:stop)
        C .= view(B, ord,         strt:stop)

        Mptr = FVector{I}(undef, nMptr)
        Mval = FVector{T}(undef, nNval * size)
        Fval = FVector{T}(undef, nFval * size)

        ssldiv_impl!(C, Mptr, Mval, Rptr, Rval, Lptr,
            Lval, Uval, Fval, res, rel, chd)

        view(B, ord, strt:stop) .= C
    end

    return B
end

function sslu_copy_R!(
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

function sslu_copy_L!(
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

function sslu_copy_U!(
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

function sslu_impl!(
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
        ns = sslu_loop!(Mptr, Mval, Rptr, Rval, Lptr,
            Lval, Uval, Fval, res, rel, chd, ns, j)
    end

    return
end

function sslu_loop!(
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
    #     F₂₂ ← 0
    #
    F₁₁ .= B₁₁
    F₂₁ .= B₂₁
    F₁₂ .= B₁₂
    F₂₂ .= zero(T)

    for i in Iterators.reverse(neighbors(chd, j))
        sslu_add_update!(F, Mptr, Mval, rel, ns, i)
        ns -= one(I)
    end

    # copy F₁ into B
    #
    #     B₁₁ ← F₁₁
    #     B₂₁ ← F₂₁
    #     B₁₂ ← F₁₂
    #
    B₁₁ .= F₁₁
    B₂₁ .= F₂₁
    B₁₂ .= F₁₂

    # factorize B₁₁ as
    #
    #   B₁₁* = U₁₁* L₁₁*
    #
    # and store
    #
    #   B₁₁ ← L₁₁ + U₁₁
    #
    fact = slu!(B₁₁)

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
        srdiv!(B₂₁, fact.U)

        #
        #   B₁₂ ← L₁₁* B₁₂
        #   
        sldiv!(fact.L, B₁₂)

        #
        #   B₂₂ ← B₂₂ + B₂₁ B₁₂
        #
        mul!(B₂₂, B₂₁, B₁₂, one(T), one(T))
    end
 
    return ns
end

function ssldiv_impl!(
        C::AbstractArray{T},
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

    # forward substitution loop
    for j in vertices(res)
        ns = ssldiv_fwd_loop!(C, Mptr, Mval, Rptr, Rval, Lptr,
            Lval, Fval, res, rel, chd, ns, j)
    end

    # backward substitution loop
    for j in reverse(vertices(res))
        ns = ssldiv_bwd_loop!(C, Mptr, Mval, Rptr, Rval, Lptr,
            Uval, Fval, res, rel, chd, ns, j)
    end

    return
end

function ssldiv_fwd_loop!(
        C::AbstractMatrix{T},
        Mptr::AbstractVector{I},
        Mval::AbstractVector{T},
        Rptr::AbstractVector{I},
        Rval::AbstractVector{T},
        Lptr::AbstractVector{I},
        Lval::AbstractVector{T},
        Fval::AbstractVector{T},
        res::AbstractGraph{I},
        rel::AbstractGraph{I}, 
        chd::AbstractGraph{I},
        ns::I,
        j::I,
    ) where {T, I}
    #
    #   nrhs is the number of columns in C
    #
    nrhs = convert(I, size(C, 2))

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
    F = reshape(view(Fval, oneto(nj * nrhs)), nj, nrhs)

    #
    #        nrhs
    #   F = [ F₁ ] nn
    #     = [ F₂ ] na
    #
    F₁ = view(F, oneto(nn),      oneto(nrhs))
    F₂ = view(F, nn + one(I):nj, oneto(nrhs))

    # B is part of the L factor
    #
    #        res(j)
    #   B = [ B₁₁  ] res(j)
    #       [ B₂₁  ] sep(j)
    #
    Rp = Rptr[j]
    Lp = Lptr[j]
    B₁₁ = reshape(view(Rval, Rp:Rp + nn * nn - one(I)), nn, nn)
    B₂₁ = reshape(view(Lval, Lp:Lp + nn * na - one(I)), na, nn)
    L₁₁ = StrictLowerTriangular(B₁₁)

    # C₁ is part of the right-hand side
    #
    #        nrhs
    #   C = [ C₁ ] res(j)
    #
    C₁ = view(C, neighbors(res, j), oneto(nrhs))

    # copy C into F
    #
    #   F₁ ← C₁
    #   F₂ ← 0
    #
    F₁ .= C₁
    F₂ .= zero(T)

    for i in Iterators.reverse(neighbors(chd, j))
        ssldiv_fwd_update!(F, Mptr, Mval, rel, ns, i)
        ns -= one(I)
    end

    # copy F into C
    #   
    #   C₁ ← F₁
    #
    C₁ .= F₁

    #
    #   C₁ ← B₁₁* C₁
    #
    sldiv!(L₁₁, C₁)

    if ispositive(na)
        ns += one(I)

        # C₂ is the update matrix at node j
        strt = Mptr[ns]
        stop = Mptr[ns + one(I)] = strt + na * nrhs
        C₂ = reshape(view(Mval, strt:stop - one(I)), na, nrhs)

        #
        #   C₂ ← F₂
        #
        C₂ .= F₂

        #
        #   C₂ ← C₂ + B₂₁ C₁
        #
        mul!(C₂, B₂₁, C₁, one(T), one(T))
    end

    return ns
end

function ssldiv_bwd_loop!(
        C::AbstractMatrix{T},
        Mptr::AbstractVector{I},
        Mval::AbstractVector{T},
        Rptr::AbstractVector{I},
        Rval::AbstractVector{T},
        Lptr::AbstractVector{I},
        Uval::AbstractVector{T},
        Fval::AbstractVector{T},
        res::AbstractGraph{I},
        rel::AbstractGraph{I}, 
        chd::AbstractGraph{I},
        ns::I,
        j::I,
    ) where {T, I}
    #
    #   nrhs is the number of columns in C
    #
    nrhs = convert(I, size(C, 2))

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
    F = reshape(view(Fval, oneto(nj * nrhs)), nj, nrhs)

    #
    #        nrhs
    #   F = [ F₁ ] nn
    #     = [ F₂ ] na
    #
    F₁ = view(F, oneto(nn),      oneto(nrhs))
    F₂ = view(F, nn + one(I):nj, oneto(nrhs))

    # B is part of the U factor
    #
    #        res(j) sep(j)
    #   B = [ B₁₁    B₁₂  ] res(j)
    #
    Rp = Rptr[j]
    Lp = Lptr[j]
    B₁₁ = reshape(view(Rval, Rp:Rp + nn * nn - one(I)), nn, nn)
    B₁₂ = reshape(view(Uval, Lp:Lp + nn * na - one(I)), nn, na)
    U₁₁ = UpperTriangular(B₁₁)

    # C₁ is part of the right-hand side
    #
    #        nrhs
    #   C = [ C₁ ] res(j)
    #
    C₁ = view(C, neighbors(res, j), oneto(nrhs))

    if ispositive(na)
        # C₂ is the update matrix at node j
        strt = Mptr[ns]
        C₂ = reshape(view(Mval, strt:strt + na * nrhs - one(I)), na, nrhs)

        ns -= one(I)

        #
        #   C₁ ← C₁ + B₁₂ C₂
        #
        mul!(C₁, B₁₂, C₂, one(T), one(T))

        #
        #   F₂ ← M₂
        #
        F₂ .= C₂
    end

    #
    #   C₁ ← B₁₁* C₁
    #
    sldiv!(U₁₁, C₁)

    # copy C into F
    #
    #   F₁ ← C₁
    #
    F₁ .= C₁

    for i in neighbors(chd, j)
        ns += one(I)
        ssldiv_bwd_update!(F, Mptr, Mval, rel, ns, i)
    end

    return ns
end

function sslu_add_update!(
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
    view(F, inj, inj) .+= B
    return
end

function ssldiv_fwd_update!(
        F::AbstractMatrix{T},
        ptr::AbstractVector{I},
        val::AbstractVector{T},
        rel::AbstractGraph{I},
        ns::I,
        i::I,
    ) where {T, I}
    #
    #   nrhs is the number of columns in F
    #
    nrhs = convert(I, size(F, 2))

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
    C = reshape(view(val, strt:strt + na * nrhs - one(I)), na, nrhs)

    #
    #   F ← F + inj C
    #
    view(F, inj, oneto(nrhs)) .+= C
    return
end

function ssldiv_bwd_update!(
        F::AbstractMatrix{T},
        ptr::AbstractVector{I},
        val::AbstractVector{T},
        rel::AbstractGraph{I},
        ns::I,
        i::I,
    ) where {T, I}
    #
    #   nrhs is the number of columns in F
    #
    nrhs = convert(I, size(F, 2))

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
    C = reshape(view(val, strt:stop - one(I)), na, nrhs)

    #
    #   C ← injᵀ F
    #
    C .= view(F, inj, oneto(nrhs))
    return
end
