include("max_luk.jl")
include("min_luk.jl")

const Lukasiewicz = Union{MaxLukQuantale, MinLukQuantale}

function MaxLuk(a::MinLuk{T}) where {T}
    return MaxLuk(one(T) - parent(a))
end

function MinLuk(a::MaxLuk{T}) where {T}
    return MinLuk(one(T) - parent(a))
end

function typemax_impl(::Type{A}, ::Type{T}) where {A <: Lukasiewicz, T}
    return one_impl(A, T)
end

function star_impl(::Type{A}, a::T) where {A <: Lukasiewicz, T}
    return one_impl(A, T)
end
