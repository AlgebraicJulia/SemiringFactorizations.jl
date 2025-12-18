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

#=
"""
    AbstractSemiring <: AbstractSemiring

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
`A <: AbstractSemiring` as well as the following methods

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
  - `ldiv_inf_impl(A, a, b, c)`
  - `rdiv_inf_impl(A, b, a, c)`

"""
abstract type AbstractSemiring <: AbstractSemiring end
=#

"""
    AbstractCommutativeSemiring <: AbstractSemiring

A unital quantale (R, ≤, ×, 1) is commutative if multiplication (×)
commutes. To create a new commutative unital quantale, define a concrete
subtype `A <: AbstractCommutativeSemiring` as well as the following methods

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
  - `ldiv_inf_impl(A, a, b, c)`

"""
abstract type AbstractCommutativeSemiring <: AbstractSemiring end

"""
    AbstractIntegralSemiring <: AbstractCommutativeSemiring
"""
abstract type AbstractIntegralSemiring <: AbstractCommutativeSemiring end

"""
    AbstractTriNorm <: AbstractCommutativeSemiring

A function ⊤: [0, 1] × [0, 1] → [0, 1] is called a
triangular norm if it is

  - commutative
  - increasing
  - associative

Each left-continuous triangular norm defines a
commutative quantale ([0, 1], ∨, ⊤). To create a
new quantale of this type, define a concrete subtype
`A <: AbstractTriNorm` as well as the following methods.

  - `mul_impl(A, a, b)`
  - `ldiv_impl(A, a, b)`

The following additional methods are optional.

  - `mul_add_impl(A, a, b, c)`
  - `ldiv_inf_impl(A, a, b, c)`
  
"""
abstract type AbstractTriNorm <: AbstractIntegralSemiring end

"""
    AbstractTriConorm <: AbstractCommutativeSemiring

A function ⊥: [0, 1] × [0, 1] → [0, 1] is called a
triangular conorm if it is

  - commutative
  - decreasing
  - associative

Each left-continuous triangular cinorm defines a
commutative quantale ([0, 1], ⊥, 0). To create a
new quantale of this type, define a concrete subtype
`A <: AbstractTriCoNorm` as well as the following methods.

  - `mul_impl(A, a, b)`
  - `ldiv_impl(A, a, b)`

The following additional methods are optional.

  - `mul_add_impl(A, a, b, c)`
  - `ldiv_inf_impl(A, a, b, c)`
  
"""
abstract type AbstractTriConorm <: AbstractIntegralSemiring end

"""
    AbstractLattice <: AbstractCommutativeSemiring

A complete Heyting algebra (R, ≤) defines a commutative quantale
(R, ≤, ∧, ⊤). To create a new Heyting algebra, define a concrete
subtype `A <: AbstractLattice` as well as the following methods.

  - `zero_impl(A, T)`
  - `one_impl(A, T)`
  - `add_impl(A, a, b)`
  - `mul_impl(A, a, b)`
  - `ldiv_impl(A, a, b)`

The following additional methods are optional.

  - `le_impl(A, a, b)`
  - `lt_impl(A, a, b)`
  - `mul_add_impl(A, a, b, c)`
  - `ldiv_inf_impl(A, a, b, c)`

"""
abstract type AbstractLattice <: AbstractIntegralSemiring end

dc(::Val{:N}, ::Val{:N}) = Val(:N)
dc(::Val{:N}, ::Val{:C}) = Val(:C)
dc(::Val{:C}, ::Val{:N}) = Val(:C)
dc(::Val{:C}, ::Val{:C}) = Val(:N)

"""
    zero_impl(A, T, dual)

Construct an additive identity of type `T`.
"""
function zero_impl(::Type{A}, ::Type{T}, dual::Val) where {A <: AbstractSemiring, T}
    return id_impl(A, zero_impl(A, T, Val(:N)), dual)
end

function zero_impl(::Type{A}, ::Type{T}, dual::Val{:N}) where {A <: AbstractTriNorm, T}
    return zero(T)
end

function zero_impl(::Type{A}, ::Type{T}, dual::Val{:N}) where {A <: AbstractTriConorm, T}
    return one(T)
end

function zero_impl(::Type{A}, ::Type{T}, dual::Val{:C}) where {A <: AbstractTriNorm, T}
    return one(T)
end

function zero_impl(::Type{A}, ::Type{T}, dual::Val{:C}) where {A <: AbstractTriConorm, T}
    return zero(T)
end

"""
    one_impl(A, T, dual)

Construct a multiplicative identity of type `T`.
"""
function one_impl(::Type{A}, ::Type{T}, dual::Val) where {A <: AbstractSemiring, T}
    return id_impl(A, one_impl(A, T, Val(:N)), dual)
end

function one_impl(::Type{A}, ::Type{T}, dual::Val{:N}) where {A <: AbstractIntegralSemiring, T}
    return zero_impl(A, T, Val(:C))
end

function one_impl(::Type{A}, ::Type{T}, dual::Val{:C}) where {A <: AbstractIntegralSemiring, T}
    return zero_impl(A, T, Val(:N))
end

"""
    star_impl(A, a)

Compute the star a*.
"""
star_impl(A, a)

function star_impl(::Type{A}, a::T) where {A <: AbstractIntegralSemiring, T}
    return one_impl(A, T, Val(:N))
end

"""
"""
id_impl(A, a, dual)

function id_impl(::Type{A}, a, dual::Val{:N}) where {A <: AbstractSemiring}
    return a
end

"""
    add_impl(A, a, b, dual)

Compute the sum a + b.
"""
function add_impl(::Type{A}, a::T, b::T, dual::Val) where {A <: AbstractSemiring, T}
    return id_impl(A, add_impl(A, id_impl(A, a, dual), id_impl(A, a, dual)), dual)
end

function add_impl(::Type{A}, a::T, b::T, dual::Val{:N}) where {A <: AbstractTriNorm, T}
    return max(a, b)
end

function add_impl(::Type{A}, a::T, b::T, dual::Val{:N}) where {A <: AbstractTriConorm, T}
    return min(a, b)
end

function add_impl(::Type{A}, a::T, b::T, dual::Val{:C}) where {A <: AbstractTriNorm, T}
    return min(a, b)
end

function add_impl(::Type{A}, a::T, b::T, dual::Val{:C}) where {A <: AbstractTriConorm, T}
    return max(a, b)
end

"""
    mul_impl(A, a, b, dual)

Compute the product a × b.
"""
function mul_impl(::Type{A}, a::T, b::T, ta::Val, tb::Val, dual::Val) where {A <: AbstractSemiring, T}
    return id_impl(A, mul_impl(A, id_impl(A, a, dc(ta, dual)), id_impl(A, b, dc(tb, dual)), Val(:N), Val(:N), Val(:N)), dual)
end

function mul_impl(::Type{A}, a::T, b::T, ta::Val{:N}, tb::Val{:C}, dual::Val) where {A <: AbstractCommutativeSemiring, T}
    return mul_impl(A, b, a, tb, ta, dual)
end 

function mul_impl(::Type{A}, a::T, b::T, ta::Val{:N}, tb::Val{:N}, dual::Val{:N}) where {A <: AbstractLattice, T}
    return add_impl(A, a, b, Val(:C))
end

function mul_impl(::Type{A}, a::T, b::T, ta::Val{:N}, tb::Val{:N}, dual::Val{:C}) where {A <: AbstractLattice, T}
    return add_impl(A, a, b, Val(:N))
end

"""
    mul_add_impl(A, a, b, c, ta, tb, dual)

Compute the sum (a × b) + c.
"""
function mul_add_impl(::Type{A}, a::T, b::T, c::T, ta::Val, tb::Val, dual::Val) where {A <: AbstractSemiring, T}
    return add_impl(A, mul_impl(A, a, b, ta, tb, dual), c, dual)
end

function mul_add_impl(::Type{A}, a::T, b::T, c::T, ta::Val{:N}, tb::Val{:C}, dual::Val) where {A <: AbstractCommutativeSemiring, T}
    return mul_add_impl(A, b, a, c, tb, ta, dual)
end

"""
    smul_impl(A, a, b, ta, tb, side)

Compute the product a* × b.
"""
function smul_impl(::Type{A}, a::T, b::T, ta::Val, tb::Val, side::Val{:L}) where {A <: AbstractSemiring, T}
    return mul_impl(A, star_impl(A, a), b, ta, tb, ta)
end

function smul_impl(::Type{A}, a::T, b::T, ta::Val, tb::Val, side::Val{:L}) where {A <: AbstractIntegralSemiring, T}
    return id_impl(A, b, tb)
end

function smul_impl(::Type{A}, a::T, b::T, ta::Val, tb::Val, side::Val{:R}) where {A <: AbstractSemiring, T}
    return mul_impl(A, b, star_impl(A, a), tb, ta, ta)
end

function smul_impl(::Type{A}, a::T, b::T, ta::Val, tb::Val, side::Val{:R}) where {A <: AbstractCommutativeSemiring, T}
    return smul_impl(A, a, b, ta, tb, Val(:L))
end

# --------- #
# Quantales #
# --------- #

"""
    le_impl(A, a, b)

Evaluate a ≤ b.
"""
function le_impl(::Type{A}, a::T, b::T) where {A <: AbstractSemiring, T}
    return add_impl(A, a, b, Val(:N)) == b
end

function le_impl(::Type{A}, a::T, b::T) where {A <: AbstractTriNorm, T}
    return a <= b
end

function le_impl(::Type{A}, a::T, b::T) where {A <: AbstractTriConorm, T}
    return a >= b
end

"""
    lt_impl(A, a, b)

Evaluate a < b.
"""
function lt_impl(::Type{A}, a::T, b::T) where {A <: AbstractSemiring, T}
    return (a != b) & le_impl(A, a, b)
end

function lt_impl(::Type{A}, a::T, b::T) where {A <: AbstractTriNorm, T}
    return a < b
end

function lt_impl(::Type{A}, a::T, b::T) where {A <: AbstractTriConorm, T}
    return a > b
end
