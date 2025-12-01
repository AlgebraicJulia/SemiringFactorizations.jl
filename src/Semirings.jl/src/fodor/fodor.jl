include("max_fod.jl")
include("min_fod.jl")

const Foduen = Union{MaxFodQuantale, MinFodQuantale}

function MaxFod(a::MinFod{T}) where {T}
    return MaxFod(one(T) - parent(a))
end

function MinFod(a::MaxFod{T}) where {T}
    return MinFod(one(T) - parent(a))
end

function typemax_impl(::Type{A}, ::Type{T}) where {A <: Foduen, T}
    return one_impl(A, T)
end

function star_impl(::Type{A}, a::T) where {A <: Foduen, T}
    return one_impl(A, T)
end
