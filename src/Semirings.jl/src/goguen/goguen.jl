include("max_gog.jl")
include("min_gog.jl")

const Goguen = Union{MaxGogQuantale, MinGogQuantale}

function MaxGog(a::MinGog{T}) where {T}
    return MaxGog(one(T) - parent(a))
end

function MinGog(a::MaxGog{T}) where {T}
    return MinGog(one(T) - parent(a))
end

function typemax_impl(::Type{A}, ::Type{T}) where {A <: Goguen, T}
    return one_impl(A, T)
end

function star_impl(::Type{A}, a::T) where {A <: Goguen, T}
    return one_impl(A, T)
end
