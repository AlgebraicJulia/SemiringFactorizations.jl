include("max_plus_pos.jl")
include("min_plus_pos.jl")
include("max_lse.jl")
include("min_lse.jl")

const LawvereSemiring{P} = Union{MaxPlusPosSemiring{P}, MinPlusPosSemiring{P}, MaxLSESemiring{P}, MinLSESemiring{P}}

#
#   a^(R/P)
#
function MaxPlusPos{P}(a::MaxPlusPos{R}) where {P, R}
    return MaxPlusPos{P}(parent(a)^(R/P))
end

#
#   a^(-R/P)
#
function MaxPlusPos{P}(a::MinPlusPos{R}) where {P, R}
    return MaxPlusPos{P}(inv(parent(a))^(R/P))
end

#
#   exp(R/P a)
#
function MaxPlusPos{P}(a::MaxLSE{R}) where {P, R}
    return MaxPlusPos{P}(exp(parent(a) * R/P))
end

#
#   exp(-R/P a)
#
function MaxPlusPos{P}(a::MinLSE{R}) where {P, R}
    return MaxPlusPos{P}(exp(-parent(a) * R/P))
end

#
#   a^(-R/P)
#
function MinPlusPos{P}(a::MaxPlusPos{R}) where {P, R}
    return MinPlusPos{P}(inv(parent(a))^(R/P))
end

#
#   a^(R/P)
#
function MinPlusPos{P}(a::MinPlusPos{R}) where {P, R}
    return MinPlusPos{P}(parent(a)^(R/P))
end

#
#   exp (-R/P a)
#
function MinPlusPos{P}(a::MaxLSE{R}) where {P, R}
    return MinPlusPos{P}(exp(-parent(a) * R/P))
end

#
#   exp (R/P a)
#
function MinPlusPos{P}(a::MinLSE{R}) where {P, R}
    return MinPlusPos{P}(exp(parent(a) * R/P))
end

#
#   R/P log a
#
function MaxLSE{P}(a::MaxPlusPos{R}) where {P, R}
    return MaxLSE{P}(log(parent(a)) * R/P)
end

#
#   -R/P log a
#
function MaxLSE{P}(a::MinPlusPos{R}) where {P, R}
    return MaxLSE{P}(-log(parent(a)) * R/P)
end

#
#   R/P a
#
function MaxLSE{P}(a::MaxLSE{R}) where {P, R}
    return MaxLSE{P}(parent(a) * (R/P))
end


#
#   -R/P a
#
function MaxLSE{P}(a::MinLSE{R}) where {P, R}
    return MaxLSE{P}(-parent(a) * R/P)
end

#
#   -R/P log a
#
function MinLSE{P}(a::MaxPlusPos{R}) where {P, R}
    return MinLSE{P}(-log(parent(a)) * R/P)
end

#
#   R/P log a
#
function MinLSE{P}(a::MinPlusPos{R}) where {P, R}
    return MinLSE{P}(log(parent(a)) * R/P)
end

#
#   -R/P a
#
function MinLSE{P}(a::MaxLSE{R}) where {P, R}
    return MinLSE{P}(-parent(a) * R/P)
end

#
#   R/P a
#
function MinLSE{P}(a::MinLSE{R}) where {P, R}
    return MinLSE{P}(parent(a) * R/P)
end
