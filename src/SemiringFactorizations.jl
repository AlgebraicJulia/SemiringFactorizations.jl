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
import .Semirings: fma

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

export SemiringNumber, AndOr, OrAnd, AndOrRel, OrAndRel, MaxMin, MinMax, LCMMul, GCDMul, GCDMulPos, MaxMul, MinMul, MaxPlus, MinPlus, MaxPlusPos, MinPlusPos, MaxLSE, MinLSE, MaxGod, MinGod, MaxGog, MinGog, MaxLuk, MinLuk, MaxFod, MinFod, Chain
export StrictLowerTriangular
export StarLU, star, slu, slmul, srmul, sldiv, srdiv, inf, fli, fri, ∧
export SymbolicStarLU
export SparseStarLU

include("strict_lower_triangular.jl")
include("array.jl")
include("dense.jl")
include("symbolic.jl")
include("sparse.jl")

end
