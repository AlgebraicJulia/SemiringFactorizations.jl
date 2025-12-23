include("or_and_rel.jl")
include("and_or_rel.jl")

const RelationSemiring = Union{AndOrRelSemiring, OrAndRelSemiring}

#
#   aᶜ
#
function OrAndRel(a::AndOrRel)
    return OrAndRel(btr(~parent(a)))
end

#
#   aᶜ
#
function AndOrRel(a::OrAndRel)
    return AndOrRel(btr(~parent(a)))
end

#
#   aᶜ
#
function id_impl(::Type{A}, a, dual::Val{:C}) where {A <: RelationSemiring}
    return btr(~a)
end

function bml(a, b)
    COL = 0x00000000000000ff
    ROW = 0x0101010101010101

    c = (COL &  a)        * (ROW & b)        |
        (COL & (a >> 8))  * (ROW & (b >> 1)) |
        (COL & (a >> 16)) * (ROW & (b >> 2)) |
        (COL & (a >> 24)) * (ROW & (b >> 3)) |
        (COL & (a >> 32)) * (ROW & (b >> 4)) |
        (COL & (a >> 40)) * (ROW & (b >> 5)) |
        (COL & (a >> 48)) * (ROW & (b >> 6)) |
        (COL & (a >> 56)) * (ROW & (b >> 7))

    return c
end

function bst(a)
    COL = 0x00000000000000ff
    ROW = 0x0101010101010101
    I   = 0x8040201008040201

    c = a | I

    c |= (ROW &  c)       * (COL &  c)
    c |= (ROW & (c >> 1)) * (COL & (c >> 8))
    c |= (ROW & (c >> 2)) * (COL & (c >> 16))
    c |= (ROW & (c >> 3)) * (COL & (c >> 24))
    c |= (ROW & (c >> 4)) * (COL & (c >> 32))
    c |= (ROW & (c >> 5)) * (COL & (c >> 40))
    c |= (ROW & (c >> 6)) * (COL & (c >> 48))
    c |= (ROW & (c >> 7)) * (COL & (c >> 56))

    return c
end

function btr(a)
    b = ((a >> 7)  ⊻ a) & 0x00aa00aa00aa00aa
    a = a ⊻ b ⊻ (b << 7)
    
    b = ((a >> 14) ⊻ a) & 0x0000cccc0000cccc
    a = a ⊻ b ⊻ (b << 14)
    
    b = ((a >> 28) ⊻ a) & 0x00000000f0f0f0f0
    a = a ⊻ b ⊻ (b << 28)
    
    return a
end

function compress(A::AbstractMatrix{Bool})
    am, an = size(A)
    bm = cld(am, 8)
    bn = cld(an, 8)
    B = zeros(UInt64, bm, bn)

    for bj in 1:bn
        ajstrt = 8bj - 7
        ajstop = 8bj
        
        for aj in ajstrt:min(ajstop, an)
            cj = aj - ajstrt
            
            for bi in 1:bm
                aistrt = 8bi - 7
                aistop = 8bi
                
                Cij = zero(UInt64)
                
                for ai in aistrt:min(aistop, am)
                    ci = ai - aistrt

                    if A[ai, aj]
                        Cij |= one(UInt64) << (ci + 8cj)
                    end
                end

                B[bi, bj] |= Cij
            end
        end
    end

    return B
end

function compress(A::SparseMatrixCSC{Bool})
    am, an = size(A)
    bm = cld(am, 8)
    bn = cld(an, 8)

    work = Vector{Int}(undef, bm)

    B = spzeros(UInt64, bm, bn)
    resize!(rowvals(B), nnz(A))
    resize!(nonzeros(B), nnz(A))
    
    bp = 0

    for bj in 1:bn
        B.colptr[bj] = bp + 1
        
        ajstrt = 8bj - 7
        ajstop = 8bj
        
        bpstrt = bp + 1
        bpstop = bp

        for aj in ajstrt:min(ajstop, an)
            bp = bpstrt - 1
            wp = 0
            bh = 0
            
            for ap in nzrange(A, aj)
                ai = rowvals(A)[ap]
                bi = cld(ai, 8)
                
                if bh < bi
                    bh = bi
                    
                    while bp < bpstop && rowvals(B)[bp + 1] < bi
                        bp += 1; wp += 1; work[wp] = rowvals(B)[bp]
                    end
    
                    if bp < bpstop && rowvals(B)[bp + 1] == bi
                        bp += 1
                    end
    
                    wp += 1; work[wp] = bi
                end
            end
            
            while bp < bpstop
                bp += 1; wp += 1; work[wp] = rowvals(B)[bp]
            end

            wpstop = wp; bp = bpstrt - 1

            for wp in 1:wpstop
                bp += 1; rowvals(B)[bp] = work[wp]
            end

            bpstop = bp
        end
    end

    B.colptr[bn + 1] = bp + 1
    resize!(rowvals(B), bp)
    resize!(nonzeros(B), bp)
    fill!(nonzeros(B), zero(UInt64))

    for bj in 1:bn
        ajstrt = 8bj - 7
        ajstop = 8bj
        bpstrt, bpstop = extrema(nzrange(B, bj))

        for aj in ajstrt:min(ajstop, an)
            cj = aj - ajstrt
            bp = bpstrt - 1
            
            for ap in nzrange(A, aj)
                ai = rowvals(A)[ap]
                bi = cld(ai, 8)
                ci = (ai - 1) & 7

                while bp < bpstop && rowvals(B)[bp + 1] < bi
                    bp += 1
                end

                if bp < bpstop && rowvals(B)[bp + 1] == bi
                    bp += 1; nonzeros(B)[bp] |= one(UInt64) << (ci + 8cj)
                end
            end
        end
    end

    return B
end

function decompress(B::AbstractMatrix{UInt64})
    bm, bn = size(B)
    am = 8bm
    an = 8bn
    A = zeros(Bool, am, an)

    for bj in 1:bn
        ajstrt = 8bj - 7
        ajstop = 8bj
        
        for aj in ajstrt:ajstop
            cj = aj - ajstrt
            
            for bi in 1:bm
                aistrt = 8bi - 7
                aistop = 8bi
                
                Bij = B[bi, bj]
                
                for ai in aistrt:aistop
                    ci = ai - aistrt
                    A[ai, aj] = isone((Bij >> (ci + 8cj)) & 1)
                end
            end
        end
    end

    return A
end

function decompress(B::SparseMatrixCSC{UInt64})
    bm, bn = size(B)
    am = 8bm
    an = 8bn
    
    A = spzeros(Bool, am, an)
    resize!(rowvals(A), 64nnz(B))
    resize!(nonzeros(A), 64nnz(B))

    ap = 0

    for bj in 1:bn
        ajstrt = 8bj - 7
        ajstop = 8bj

        for aj in ajstrt:ajstop
            cj = aj - ajstrt
            A.colptr[aj] = ap + 1

            for bp in nzrange(B, bj)
                bi = rowvals(B)[bp]
                aistrt = 8bi - 7
                aistop = 8bi

                Bij = nonzeros(B)[bp]

                for ai in aistrt:aistop
                    ci = ai - aistrt
                    ap += 1
                    rowvals(A)[ap] = ai
                    nonzeros(A)[ap] = isone((Bij >> (ci + 8cj)) & 1)
                end
            end
        end
    end

    A.colptr[an + 1] = ap + 1
    return A
end
