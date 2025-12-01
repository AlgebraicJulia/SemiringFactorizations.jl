struct ChainQuantale{A <: AbstractQuantale} <: AbstractQuantale end

const Chain{A, N, T} = SemiringNumber{ChainQuantale{A}, NTuple{N, T}}

function Chain(tup::NTuple{N, SemiringNumber{A, T}}) where {A, N, T}
    return Chain{A, N, T}(map(parent, tup))
end

#
#   (0, 0, ..., 0)
#
function zero_impl(::Type{ChainQuantale{A}}, ::Type{NTuple{N, T}}) where {A <: AbstractQuantale, N, T}
    return ntuple(_ -> zero_impl(A, T), N)
end

#
#   (1, 0, ..., 0)
#
function one_impl(::Type{ChainQuantale{A}}, ::Type{NTuple{N, T}}) where {A <: AbstractQuantale, N, T}
    return ntuple(i -> ifelse(isone(i), one_impl(A, T), zero_impl(A, T)), N)
end

#
#   (⊤, ⊤, ..., ⊤)
#
function typemax_impl(::Type{ChainQuantale{A}}, ::Type{NTuple{N, T}}) where {A <: AbstractQuantale, N, T}
    return ntuple(_ -> typemax_impl(A, T), N)
end

#
#   b₁ = a₁*
#   ⋮
#   bᵢ = a₁* (a₂ × bᵢ₋₁ + ... + aᵢ × b₁)
#
function star_impl(::Type{ChainQuantale{A}}, a::NTuple{N, T}) where {A <: AbstractQuantale, N, T}
    b = setindex(a, star_impl(A, a[1]), 1)

    for i in 2:N
        b = setindex(b, mul_impl(A, b[1], reduce((c, j) -> mul_add_impl(A, a[j], b[i - j + 1], c), 2:i; init=zero_impl(A, T))), i)
    end

    return b
end

#
#   cᵢ = aᵢ ∨ bᵢ
#
function add_impl(::Type{ChainQuantale{A}}, a::NTuple{N, T}, b::NTuple{N, T}) where {A <: AbstractQuantale, N, T}
    return ntuple(i -> add_impl(A, a[i], b[i]), N)
end

#
#   cᵢ = aᵢ ∧ bᵢ
#
function inf_impl(::Type{ChainQuantale{A}}, a::NTuple{N, T}, b::NTuple{N, T}) where {A <: AbstractQuantale, N, T}
    return ntuple(i -> inf_impl(A, a[i], b[i]), N)
end

#
#   cᵢ = a₁ × bᵢ ∨ ... ∨ aᵢ × b₁
#
function mul_impl(::Type{ChainQuantale{A}}, a::NTuple{N, T}, b::NTuple{N, T}) where {A <: AbstractQuantale, N, T}
    return ntuple(i -> reduce((c, j) -> mul_add_impl(A, a[j], b[i - j + 1], c), 1:i; init=zero_impl(A, T)), N)
end

#
#   cᵢ = a₁ \ bᵢ ∧ ... ∧ aₙ₋ᵢ₊₁ \ bₙ
#
function ldiv_impl(::Type{ChainQuantale{A}}, a::NTuple{N, T}, b::NTuple{N, T}) where {A <: AbstractQuantale, N, T}
    return ntuple(i -> reduce((c, j) -> inf_ldiv_impl(A, a[j - i + 1], b[j], c), i:N; init=typemax_impl(A, T)), N)
end

#
#   cᵢ = bᵢ / a₁ ∧ ... ∧ bₙ / aₙ₋ᵢ₊₁
#
function rdiv_impl(::Type{ChainQuantale{A}}, b::NTuple{N, T}, a::NTuple{N, T}) where {A <: AbstractQuantale, N, T}
    return ntuple(i -> reduce((c, j) -> inf_rdiv_impl(A, b[j], a[j - i + 1], c), i:N; init=typemax_impl(A, T)), N)
end

#
#   aᵢ ≤ bᵢ ∀i
#
function le_impl(::Type{ChainQuantale{A}}, a::NTuple{N, T}, b::NTuple{N, T}) where {A <: AbstractQuantale, N, T}
    return all(i -> le_impl(A, a[i], b[i]), 1:N)
end
