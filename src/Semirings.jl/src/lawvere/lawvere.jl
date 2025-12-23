include("max_plus_pos.jl")
include("min_plus_pos.jl")
include("max_lse.jl")
include("min_lse.jl")

const LawvereSemiring = Union{MaxPlusPosSemiring, MinPlusPosSemiring, MaxLSESemiring, MinLSESemiring}

#
#   a⁻¹
#
function MaxPlusPos(a::MinPlusPos)
    return MaxPlusPos(inv(parent(a)))
end

#
#   eᵃ
#
function MaxPlusPos(a::MaxLSE)
    return MaxPlusPos(exp(parent(a)))
end

#
#   e⁻ᵃ
#
function MaxPlusPos(a::MinLSE)
    return MaxPlusPos(exp(-parent(a)))
end

#
#   a⁻¹
#
function MinPlusPos(a::MaxPlusPos)
    return MinPlusPos(inv(parent(a)))
end

#
#   e⁻ᵃ
#
function MinPlusPos(a::MaxLSE)
    return MinPlusPos(exp(-parent(a)))
end

#
#   eᵃ
#
function MinPlusPos(a::MinLSE)
    return MinPlusPos(exp(parent(a)))
end

#
#   log a
#
function MaxLSE(a::MaxPlusPos)
    return MaxLSE(log(parent(a)))
end

#
#   -log a
#
function MaxLSE(a::MinPlusPos)
    return MaxLSE(-log(parent(a)))
end

#
#   -a
#
function MaxLSE(a::MinLSE)
    return MaxLSE(-parent(a))
end

#
#   -log a
#
function MinLSE(a::MaxPlusPos)
    return MinLSE(-log(parent(a)))
end

#
#   log a
#
function MinLSE(a::MinPlusPos)
    return MinLSE(log(parent(a)))
end

#
#   -a
#
function MinLSE(a::MaxLSE)
    return MinLSE(-parent(a))
end
