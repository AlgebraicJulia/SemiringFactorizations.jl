include("max_luk.jl")
include("min_luk.jl")

const Lukasiewicz = Union{MaxLukSemiring, MinLukSemiring}

function MaxLuk(a::MinLuk{T}) where {T}
    return MaxLuk(one(T) - parent(a))
end

function MinLuk(a::MaxLuk{T}) where {T}
    return MinLuk(one(T) - parent(a))
end
