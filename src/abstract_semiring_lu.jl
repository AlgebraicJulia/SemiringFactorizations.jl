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
    sldiv!(A, B::AbstractVecOrMat)

Solve the linear fixed-point equation

```math
    AX + B = X,
```

over-writing B with the solution.
"""
sldiv!(A, B::AbstractVecOrMat)

function sldiv!(A::AbstractMatrix, B::AbstractVecOrMat)
    return sldiv!(slu(A), B)
end

"""
    srdiv!(B::AbstractVecOrMat, A)

Solve the linear fixed-point equation

```math
    XA + B = X,
```

over-writing B with the solution.
"""
srdiv!(B::AbstractVecOrMat, A)

function srdiv!(B::AbstractVecOrMat, A::AbstractMatrix)
    return srdiv!(B, slu(A))
end
