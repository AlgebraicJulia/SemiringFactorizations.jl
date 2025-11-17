module SemiringFactorizations

using AbstractTrees
using Accessors
using Base: oneto, @propagate_inbounds, OneTo, Slice, AbstractVecOrMat
using Base.Threads: nthreads, @threads
using CliqueTrees
using CliqueTrees.Utilities
using CliqueTrees: incident, nov, PermutationOrAlgorithm, SupernodeType,
    DEFAULT_ELIMINATION_ALGORITHM, DEFAULT_SUPERNODE_TYPE
using Graphs
using LinearAlgebra
using LinearAlgebra: StridedMatrix
using Octavian
using SparseArrays
using TropicalGEMM
using TropicalNumbers
using WiringDiagrams

const DEFAULT_BLOCK_SIZE = 32
const SparseColumnCSC{T, I} = SubArray{T, 1, SparseMatrixCSC{T, I}, Tuple{Slice{OneTo{Int}}, Int}, false}

function unpack(A::SparseColumnCSC)
    return A.parent, A.indices[2]
end

export StrictLowerTriangular
export StarLU, star, slu, slu!, slmul!, slmul, srmul!, srmul, slres!, slres, srres!, srres, lres, rres
export SymbolicStarLU
export SparseStarLU, mtstar, mtslmul!, mtsrmul!
export TropicalMinMax, TropicalMinMaxF64, TropicalMinMaxF32,
    TropicalMinMaxF16, TropicalMinMaxI64, TropicalMinMaxI32,
    TropicalMinMaxI16
export RE
export TopN, Top2, Top3, Top4, Top5
export KleeneAlgebra

include("regular_expression.jl")
include("topn.jl")
include("strict_lower_triangular.jl")
include("scalar.jl")
include("array.jl")
include("dense.jl")
include("symbolic.jl")
include("sparse.jl")
include("kleene_algebras.jl")

end
