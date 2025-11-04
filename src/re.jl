"""
    RE

A regular expression.
"""
struct RE
    str::String
    flg::Bool
end

function RE(str::AbstractString)
    return RE(str, false)
end

function Base.String(a::RE)
    return a.str
end

function Base.Regex(a::RE)
    return Regex(a.str)
end

function nmg(str::String)
    return "(?:" * str * ")"
end

function Base.:+(a::RE, b::RE)
    if iszero(a)
        c = b
    elseif iszero(b)
        c = a
    else
        c = RE(a.str * "|" * b.str, true)
    end

    return c
end

function Base.:*(a::RE, b::RE)
    if iszero(a) || iszero(b)
        c = zero(RE)
    elseif !a.flg && !b.flg
        c = RE(a.str * b.str)
    elseif !a.flg && b.flg
        c = RE(a.str * nmg(b.str))
    elseif a.flg && !b.flg
        c = RE(nmg(a.str) * b.str)
    else
        c = RE(nmg(a.str) * nmg(b.str))
    end

    return c
end

function Base.zero(::Union{RE, Type{RE}})
    return RE("a^")
end

function Base.one(::Union{RE, Type{RE}})
    return RE("")
end

function Base.show(io::IO, a::RE)
    print(io, a.str)
    return
end

function Base.transpose(a::RE)
    return a
end

function Base.conj(a::RE)
    return a
end

function Base.convert(::Type{RE}, str::AbstractString)
    return RE(str)
end

function Base.:(==)(a::RE, b::RE)
    return a.str == b.str
end
