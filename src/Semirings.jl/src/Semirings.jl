module Semirings

using Base: setindex
using LinearAlgebra, VectorizationBase, LoopVectorization, Octavian, FixedSizeArrays
using Random
using Random: Repetition, Sampler
using SparseArrays

export SemiringNumber, AndOr, OrAnd, AndOrRel, OrAndRel, MaxMin, MinMax, LCMMul, GCDMul, GCDMulPos, MaxMul, MinMul, MaxPlus, MinPlus, MaxPlusPos, MinPlusPos, MaxLSE, MinLSE, MaxGod, MinGod, MaxGog, MinGog, MaxLuk, MinLuk, MaxFod, MinFod, Chain
export MulMatrix
export fli, fri, inf, ∧, star, slmul, srmul, sldiv, srdiv

include("abstract_semiring.jl")
include("semiring_number.jl")
include("semiring_sampler.jl")
include("powerset/powerset.jl")
include("relation/relation.jl")
include("bottleneck/bottleneck.jl")
include("division/division.jl")
include("gcd_mul_pos.jl")
include("tropical/tropical.jl")
include("lawvere/lawvere.jl")
include("godel/godel.jl")
include("goguen/goguen.jl")
include("lukasiewicz/lukasiewicz.jl")
include("fodor/fodor.jl")
include("chain.jl")
include("mul.jl")

const ∧ = inf

end
