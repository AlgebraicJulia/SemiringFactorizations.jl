include("max_fod.jl")
include("min_fod.jl")

const Foduen = Union{MaxFodSemiring, MinFodSemiring}

function MaxFod(a::MinFod{T}) where {T}
    return MaxFod(one(T) - parent(a))
end

function MinFod(a::MaxFod{T}) where {T}
    return MinFod(one(T) - parent(a))
end
