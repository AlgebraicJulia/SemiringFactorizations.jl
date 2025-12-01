"""
    AbstractSemiring

A closed semiring is a sextuple (R, +, ×, *, 0, 1), where

  - (R, +, 0) is a commutative monoid
  - (R, ×, 1) is a monoid
  - multiplication (×) distributes over addition (+)
  - star (*) satisfies 1 + a*a = 1 * aa* = a*

To create a new semiring, define a concrete subtype `A <: AbstractSemiring`
as well as the following methods.

  - `zero_impl(A, T)`
  - `one_impl(A, T)`
  - `star_impl(A, a)`
  - `add_impl(A, a, b)`
  - `mul_impl(A, a, b)`

The following additional methods are optional.

  - `mul_add_impl(A, a, b, c)`
  - `slmul_impl(A, a, b)`
  - `srmul_impl(A, b, a)`

"""
abstract type AbstractSemiring end


"""
    AbstractQuantale <: AbstractSemiring

A unital quantale is a quadruple (R, ≤, ×, 1), where

  - (R, ≤) is a complete lattice
  - (R, ×, 1) is a monoid
  - multiplication (×) distributes over joins

Every unital quantale is also a closed semiring (R, +, ×, *, 0, 1),
where

  - a + b is the join of a and b
  - 0 is the least element
  - a* is the infinite sum 1 + a + a² + a³ + ⋯

To create a new unital quantale, define a concrete subtype
`A <: AbstractQuantale` as well as the following methods

  - `zero_impl(A, T)`
  - `one_impl(A, T)`
  - `typemax_impl(A, T)`
  - `star_impl(A, a)`
  - `add_impl(A, a, b)`
  - `mul_impl(A, a, b)`
  - `inf_impl(A, a, b)`
  - `ldiv_impl(A, a, b)`
  - `rdiv_impl(A, b, a)`

The following additional methods are optional.

  - `le_impl(A, a, b)`
  - `lt_impl(A, a, b)`
  - `slmul_impl(A, a, b)`
  - `srmul_impl(A, b, a)`
  - `sldiv_impl(A, a, b)`
  - `srdiv_impl(A, b, a)`
  - `mul_add_impl(A, a, b, c)`
  - `inf_ldiv_impl(A, a, b, c)`
  - `inf_rdiv_impl(A, b, a, c)`

"""
abstract type AbstractQuantale <: AbstractSemiring end

"""
    AbstractCommutativeQuantale <: AbstractQuantale

A unital quantale (R, ≤, ×, 1) is commutative if multiplication (×)
commutes. To create a new commutative unital quantale, define a concrete
subtype `A <: AbstractCommutativeQuantale` as well as the following methods

  - `zero_impl(A, T)`
  - `one_impl(A, T)`
  - `typemax_impl(A, T)`
  - `star_impl(A, a)`
  - `add_impl(A, a, b)`
  - `mul_impl(A, a, b)`
  - `inf_impl(A, a, b)`
  - `ldiv_impl(A, a, b)`

The following additional methods are optional.

  - `le_impl(A, a, b)`
  - `lt_impl(A, a, b)`
  - `slmul_impl(A, a, b)`
  - `sldiv_impl(A, a, b)`
  - `mul_add_impl(A, a, b, c)`
  - `inf_ldiv_impl(A, a, b, c)`

"""
abstract type AbstractCommutativeQuantale <: AbstractQuantale end

"""
    AbstractLattice <: AbstractCommutativeQuantale

A complete lattice (R, ≤) is called a frame if meets distribute
over joins. Every frame defines a commutative quantale (R, ≤, ×, 1),
where

  - a × b is the meet of a and b
  - 1 is the largest element

To create a new frame, define a concrete subtype `A <: AbstractLattice`
as well as the following methods

  - `zero_impl(A, T)`
  - `one_impl(A, T)`
  - `add_impl(A, a, b)`
  - `mul_impl(A, a, b)`
  - `ldiv_impl(A, a, b)`

The following additional methods are optional.

  - `le_impl(A, a, b)`
  - `lt_impl(A, a, b)`
  - `mul_add_impl(A, a, b, c)`
  - `inf_ldiv_impl(A, a, b, c)`

"""
abstract type AbstractLattice <: AbstractCommutativeQuantale end

# ---------------- #
# Closed Semirings #
# ---------------- #

"""
    zero_impl(A, T)

Construct an additive identity of type `T`.
"""
zero_impl(A, T)

"""
    one_impl(A, T)

Construct a multiplicative identity of type `T`.
"""
one_impl(A, T)

"""
    star_impl(A, a)

Compute the star a*.
"""
star_impl(A, a)

function star_impl(::Type{A}, a::T) where {A <: AbstractLattice, T}
    return one_impl(A, T)
end

"""
    add_impl(A, a, b)

Compute the sum a + b.
"""
add_impl(A, a, b)

"""
    mul_impl(A, a, b)

Compute the product a × b.
"""
mul_impl(A, a, b)

"""
    mul_add_impl(A, a, b, c)

Compute the sum (a × b) + c.
"""
function mul_add_impl(::Type{A}, a, b, c) where {A <: AbstractSemiring}
    return add_impl(A, mul_impl(A, a, b), c)
end

"""
    slmul_impl(A, a, b)

Compute the product a* × b.
"""
function slmul_impl(::Type{A}, a::T, b::T) where {A <: AbstractSemiring, T}
    return mul_impl(A, star_impl(A, a), b)
end

function slmul_impl(::Type{A}, a::T, b::T) where {A <: AbstractLattice, T}
    return b
end

"""
    srmul_impl(A, b, a)

Compute the product b × a*.
"""
function srmul_impl(::Type{A}, b::T, a::T) where {A <: AbstractSemiring, T}
    return mul_impl(A, b, star_impl(A, a))
end

function srmul_impl(::Type{A}, b::T, a::T) where {A <: AbstractCommutativeQuantale, T}
    return slmul_impl(A, a, b)
end

# --------- #
# Quantales #
# --------- #

"""
    typemax_impl(A, T)

Construct a top element of type `T`.
"""
typemax_impl(A, T)

function typemax_impl(::Type{A}, ::Type{T}) where {A <: AbstractLattice, T}
    return one_impl(A, T)
end

"""
    inf_impl(A, a, b)

Compute the meet a ∧ b.
"""
inf_impl(A, a, b)

function inf_impl(::Type{A}, a::T, b::T) where {A <: AbstractLattice, T}
    return mul_impl(A, a, b)
end

"""
    le_impl(A, a, b)

Evaluate a ≤ b.
"""
function le_impl(::Type{A}, a::T, b::T) where {A <: AbstractQuantale, T}
    return add_impl(A, a, b) == b
end

"""
    lt_impl(A, a, b)

Evaluate a < b.
"""
function lt_impl(::Type{A}, a::T, b::T) where {A <: AbstractQuantale, T}
    return (a != b) & le_impl(A, a, b)
end

"""
    ldiv_impl(A, a, b)

Compute the residual a \\ b.
"""
ldiv_impl(A, a, b)

"""
    rdiv_impl(A, b, a)

Compute the residual b / a.
"""
rdiv_impl(A, b, a)

function rdiv_impl(::Type{A}, b::T, a::T) where {A <: AbstractCommutativeQuantale, T}
    return ldiv_impl(A, a, b)
end

"""
    sldiv_impl(A, a, b)

Compute the residual a* \\ b.
"""
function sldiv_impl(::Type{A}, a::T, b::T) where {A <: AbstractQuantale, T}
    return ldiv_impl(A, star_impl(A, a), b)
end

"""
    srdiv_impl(A, b, a)

Compute the residual b / a*.
"""
function srdiv_impl(::Type{A}, b::T, a::T) where {A <: AbstractQuantale, T}
    return rdiv_impl(A, b, star_impl(A, a))
end

function srdiv_impl(::Type{A}, b::T, a::T) where {A <: AbstractCommutativeQuantale, T}
    return sldiv_impl(A, a, b)
end

"""
    inf_ldiv_impl(A, a, b, c)

Compute the meet (a \\ b) ∧ c.
"""
function inf_ldiv_impl(::Type{A}, a::T, b::T, c::T) where {A <: AbstractQuantale, T}
    return inf_impl(A, ldiv_impl(A, a, b), c)
end

"""
    inf_rdiv_impl(A, b, a, c)

Compute the meet (b / a) ∧ c.
"""
function inf_rdiv_impl(::Type{A}, b::T, a::T, c::T) where {A <: AbstractQuantale, T}
    return inf_impl(A, rdiv_impl(A, b, a), c)
end

function inf_rdiv_impl(::Type{A}, b::T, a::T, c::T) where {A <: AbstractCommutativeQuantale, T}
    return inf_ldiv_impl(A, a, b, c)
end
