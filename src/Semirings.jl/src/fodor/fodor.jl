include("max_fod.jl")
include("min_fod.jl")

const FodorSemiring = Union{MaxFodSemiring, MinFodSemiring}

function MaxFod(a::MinFod{T}) where {T}
    return MaxFod(one(T) - parent(a))
end

function MinFod(a::MaxFod{T}) where {T}
    return MinFod(one(T) - parent(a))
end

function id_impl(::Type{A}, a::T, dual::Val{:C}) where {A <: FodorSemiring, T}
    return one(T) - a
end
