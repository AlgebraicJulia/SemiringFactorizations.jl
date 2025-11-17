struct KleeneAlgebra{M <: AbstractMatrix} <: AbstractAlgebra{M} end

function WiringDiagrams.apply(algebra::KleeneAlgebra{M}, diagram::AbstractWiringDiagram{I}, arguments) where {T, M <: AbstractMatrix{T}, I <: Integer}
    W = nw(diagram)
    Q = nop(diagram)
    
    A = M(undef, W, W)
    C = M(undef, Q, Q)
    f = FVector{I}(undef, W)

    #
    #    A ← I
    #
    for w in wires(diagram)
        f[w] = zero(I)

        for ww in wires(diagram)
            if w == ww
                A[ww, w] = one(T)
            else
                A[ww, w] = zero(T)
            end
        end
    end
    
    ww = W

    for w in Iterators.reverse(outportwires(diagram))
        if iszero(f[w])
            f[w] = ww; ww -= one(I)
        end
    end

    R = ww

    for w in Iterators.reverse(wires(diagram))
        if iszero(f[w])
            f[w] = ww; ww -= one(I)
        end
    end

    for b in boxes(diagram)
        i = zero(I)
        B = arguments[b]

        #
        #    A ← A + f B fᵀ
        #
        for w in portwires(diagram, b)
            ii = zero(I)
            i += one(I); j = f[w]

            for ww in portwires(diagram, b)
                ii += one(I); jj = f[ww]
                A[jj, j] += B[ii, i]
            end
        end
    end

    #
    #    A = [ RR RS ]
    #        [ SR SS ]
    #
    RR = view(A, oneto(R),     oneto(R))
    RS = view(A, oneto(R),     R + one(I):W)
    SR = view(A, R + one(I):W, oneto(R))
    SS = view(A, R + one(I):W, R + one(I):W)  

    #
    #    SS ← SS + SR RR* RS
    #
    mul_impl!(SS, SR, slmul!(slu!(RR), RS), Val(:N), Val(:N))

    for p in outports(diagram)
        w = outwire(diagram, p)
        j = f[w]

        for pp in outports(diagram)
            ww = outwire(diagram, pp)
            jj = f[ww]
            
            C[pp, p] = A[jj, j]
        end
    end

    return C
end
