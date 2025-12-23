include("max_gog.jl")
include("min_gog.jl")

const GoguenSemiring = Union{MaxGogSemiring, MinGogSemiring}

function MaxGog(a::MinGog{T}) where {T}
    return MaxGog(one(T) - parent(a))
end

function MinGog(a::MaxGog{T}) where {T}
    return MinGog(one(T) - parent(a))
end
