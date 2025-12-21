include("lcm_mul.jl")
include("gcd_mul.jl")

const DivisionSemiring = Union{LCMMulSemiring, GCDMulSemiring}

function LCMMul(a::GCDMul)
    return LCMMul(inv(parent(a)))
end

function GCDMul(a::LCMMul)
    return GCDMul(inv(parent(a)))
end

function id_impl(::Type{A}, a, dual::Val{:C}) where {A <: DivisionSemiring}
    return inv(a)
end
