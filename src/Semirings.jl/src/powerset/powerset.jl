include("or_and.jl")
include("and_or.jl")

const PowersetLattice = Union{AndOrLattice, OrAndLattice}

#
#   aᶜ
#
function AndOr(a::OrAnd)
    return AndOr(~parent(a))
end

#
#   aᶜ
#
function OrAnd(a::AndOr)
    return OrAnd(~parent(a))
end
