module SemiringFactorizations

using AbstractTrees
using Accessors
using Base: oneto, @propagate_inbounds, OneTo, Slice, AbstractVecOrMat
using Base.FastMath: add_fast
using Base.Threads: nthreads, @threads
using CliqueTrees
using CliqueTrees.Utilities
using CliqueTrees: incident, nov, PermutationOrAlgorithm, SupernodeType,
    DEFAULT_ELIMINATION_ALGORITHM, DEFAULT_SUPERNODE_TYPE
using Graphs
using LinearAlgebra
using Octavian
using SparseArrays
using Static

include("Semirings.jl/src/Semirings.jl")

using .Semirings
using .Semirings: NativeTypes

const DEFAULT_BLOCK_SIZE = 32
const SparseColumnCSC{T, I} = SubArray{T, 1, SparseMatrixCSC{T, I}, Tuple{Slice{OneTo{Int}}, Int}, false}
const AbstractRowVector{T, Vec} = Transpose{T, Vec} where {Vec <: AbstractVector{T}}

function unpack(A::SparseColumnCSC)
    return A.parent, A.indices[2]
end

function wrapcopy(::Type{T}, A::AbstractArray) where {T}
    return Array{T}(A)
end

function wrapcopy(::Type{T}, A::Transpose) where {T}
    return transpose(wrapcopy(T, parent(A)))
end

export SemiringNumber, AndOr, OrAnd, MaxPlus, MinPlus, MaxMul, ⅋
export StrictLowerTriangular
export StarLU, StarTriangular, star, slu
export SymbolicStarLU
export SparseStarLU, SparseStarTriangular

include("strict_lower_triangular.jl")
include("abstract_star_lu.jl")
include("star_lu.jl")
include("symbolic_star_lu.jl")
include("sparse_star_lu.jl")

end
