abstract type AbstractSemiringLU{T} end

# ------------------------------ #
# Abstract Semiring LU Interface #
# ------------------------------ #

"""
    size(F::AbstractSemiringLU)

Get the size of a factorized matrix.
"""
Base.size(F::AbstractSemiringLU)

"""
    slu!(A::AbstractMatrix)

Compute an LU factorization of a semiring-
valued matrix A. The factors are stored
in A.
"""
slu(A::AbstractMatrix)

"""
    slu(A::AbstractMatrix)

Compute an LU factorization of a semiring-
valued matrix A.
"""
function slu(A::AbstractMatrix)
    return slu!(FMatrix(A))
end

"""
    sldiv!(A, B::AbstractVecOrMat)

Solve the linear fixed-point equation

```math
    AX + B = X.
```

The result is stored in B.
"""
sldiv!(A::AbstractSemiringLU, B::AbstractVecOrMat)

function sldiv!(A::AbstractMatrix, B::AbstractVecOrMat)
    return sldiv!(slu(A), B)
end

"""
    srdiv!(B::AbstractVecOrMat, A)

Solve the linear fixed-point equation

```math
    XA + B = X.
```

The result is stored in B.
"""
srdiv!(B::AbstractVecOrMat, A::AbstractSemiringLU)

function srdiv!(B::AbstractVecOrMat, A::AbstractMatrix)
    return srdiv!(B, slu(A))
end
