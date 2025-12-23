include("max_god.jl")
include("min_god.jl")

const GodelSemiring = Union{MaxGodSemiring, MinGodSemiring}

function MaxGod(a::MinGod{T}) where {T}
    return MaxGod(one(T) - parent(a))
end

function MinGod(a::MaxGod{T}) where {T}
    return MinGod(one(T) - parent(a))
end
