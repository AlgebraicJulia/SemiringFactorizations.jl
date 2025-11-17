"""
    RegularExpression

A regular expression.
"""
struct RegularExpression
    str::String
    flg::Bool
end

const RE = RegularExpression

function RegularExpression(str::AbstractString)
    return RegularExpression(str, false)
end

function Base.String(a::RegularExpression)
    return a.str
end

function Base.Regex(a::RegularExpression)
    return Regex(a.str)
end

function nmg(str::String)
    return "(?:" * str * ")"
end

function Base.:+(a::RegularExpression, b::RegularExpression)
    if iszero(a)
        c = b
    elseif iszero(b)
        c = a
    else
        c = RegularExpression(a.str * "|" * b.str, true)
    end

    return c
end

function Base.:*(a::RegularExpression, b::RegularExpression)
    if iszero(a) || iszero(b)
        c = zero(RegularExpression)
    elseif !a.flg && !b.flg
        c = RegularExpression(a.str * b.str)
    elseif !a.flg && b.flg
        c = RegularExpression(a.str * nmg(b.str))
    elseif a.flg && !b.flg
        c = RegularExpression(nmg(a.str) * b.str)
    else
        c = RegularExpression(nmg(a.str) * nmg(b.str))
    end

    return c
end

function Base.zero(::Union{RegularExpression, Type{RegularExpression}})
    return RegularExpression("a^")
end

function Base.one(::Union{RegularExpression, Type{RegularExpression}})
    return RegularExpression("")
end

function Base.show(io::IO, a::RegularExpression)
    print(io, a.str)
    return
end

function Base.transpose(a::RegularExpression)
    return a
end

function Base.conj(a::RegularExpression)
    return a
end

function Base.convert(::Type{RegularExpression}, str::AbstractString)
    return RegularExpression(str)
end

function Base.:(==)(a::RegularExpression, b::RegularExpression)
    return a.str == b.str
end
