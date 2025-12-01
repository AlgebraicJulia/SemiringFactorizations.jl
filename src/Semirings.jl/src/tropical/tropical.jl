include("max_mul.jl")
include("min_mul.jl")
include("max_plus.jl")
include("min_plus.jl")

const TropicalQuantale = Union{MaxMulQuantale, MinMulQuantale, MaxPlusQuantale, MinPlusQuantale}

#
#   1/a
#
function MaxMul(a::MinMul)
    return MaxMul(inv(parent(a)))
end

#
#   eᵃ
#
function MaxMul(a::MaxPlus)
    return MaxMul(exp(parent(a)))
end

#
#   e⁻ᵃ
#
function MaxMul(a::MinPlus)
    return MaxMul(exp(-parent(a)))
end

#
#   1/a
#
function MinMul(a::MaxMul)
    return MinMul(inv(parent(a)))
end

#
#   e⁻ᵃ
#
function MinMul(a::MaxPlus)
    return MinMul(exp(-parent(a)))
end

#
#   eᵃ
#
function MinMul(a::MinPlus)
    return MinMul(exp(parent(a)))
end

#
#   log a
#
function MaxPlus(a::MaxMul)
    return MaxPlus(log(parent(a)))
end

#
#   -log a
#
function MaxPlus(a::MinMul)
    return MaxPlus(-log(parent(a)))
end

#
#   -a
#
function MaxPlus(a::MinPlus)
    return MaxPlus(-parent(a))
end

#
#   -log a
#
function MinPlus(a::MaxMul)
    return MinPlus(-log(parent(a)))
end

#
#   log a
#
function MinPlus(a::MinMul)
    return MinPlus(log(parent(a)))
end

#
#   -a
#
function MinPlus(a::MaxPlus)
    return MinPlus(-parent(a))
end

#
#   { 1 if a ≤ 1
#   { ⊤ if a > 1
#
function star_impl(::Type{A}, a::T) where {A <: TropicalQuantale, T}
    ϵ = one_impl(A, T)
    ⊤ = typemax_impl(A, T)
    return ifelse(le_impl(A, a, ϵ), ϵ, ⊤)
end
