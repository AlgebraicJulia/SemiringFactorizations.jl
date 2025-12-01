"""
    RE

A regular expression.
"""
struct RE
    head::Symbol
    char::Char
    flag::Bool
    left::RE
    right::RE

    function RE(char::AbstractChar)
        if char == '∅'
            a = zero(RE)
        elseif char == 'ε'
            a = one(RE)
        else
            a = new(:char, char, false)
        end

        return a
    end

    function RE(head::Symbol, flag::Bool, left, right)
        return new(head, '∅', flag, left, right)
    end

    function RE(head::Symbol, flag::Bool, left)
        return new(head, '∅', flag, left)
    end

    function RE(head::Symbol, flag::Bool)
        return new(head, '∅', flag)
    end
end

function RE(str::AbstractString)
    return mapreduce(RE, *, str; init=one(RE))
end

function Base.string(a::RE)
    if a.head == :char
        str = string(a.char)
    elseif a.head == :zero
        str = "∅"
    elseif a.head == :one
        str = "ε"
    elseif a.head == :top
        str = "⊤"
    else
        l = a.left

        if a.head == :star
            if l.head == :char
                sl = string(l)
            else
                sl = "(" * string(l) * ")"
            end

            str = sl * "*"
        else
            r = a.right

            if a.head == :add
                if l.head in (:char, :zero, :one, :top, :star, :add)
                    sl = string(l)
                else
                    sl = "(" * string(l) * ")"
                end

                if r.head in (:char, :zero, :one, :top, :star, :add)
                    sr = string(r)
                else
                    sr = "(" * string(r) * ")"
                end

                str = sl * "|" * sr
            elseif a.head == :inf
                if l.head in (:char, :zero, :one, :top, :star, :inf)
                    sl = string(l)
                else
                    sl = "(" * string(l) * ")"
                end

                if r.head in (:char, :zero, :one, :top, :star, :inf)
                    sr = string(r)
                else
                    sr = "(" * string(r) * ")"
                end

                str = sl * "∧" * sr
            elseif a.head == :mul
                if l.head in (:char, :zero, :one, :top, :star, :mul)
                    sl = string(l)
                else
                    sl = "(" * string(l) * ")"
                end

                if r.head in (:char, :zero, :one, :top, :star, :mul)
                    sr = string(r)
                else
                    sr = "(" * string(r) * ")"
                end

                str = sl * sr
            else
                error()
            end
        end
    end

    return str
end

function Base.String(a::RE)
    return string(a)
end

function Base.Regex(a::RE)
    return Regex(string(a))
end

function Base.:+(a::RE, b::RE)
    if a.head == :top || b.head == :top
        c = typemax(RE)
    elseif a.head == :zero
        c = b
    elseif b.head == :zero || a.head == b.head == :char && a.char == b.char
        c = a
    else
        c = RE(:add, a.flag || b.flag, a, b)
    end

    return c
end

function TropicalNumbers.inf(a::RE, b::RE)
    if a.head == :zero || b.head == :zero
        c = zero(RE)
    elseif a.head == :one || b.head == :one
        c = one(RE)
    elseif a.head == :top
        c = b
    elseif b.head == :top || a.head == b.head == :char && a.char == b.char
        c = a
    else
        c = RE(:inf, a.flag && b.flag, a, b)
    end

    return c
end

function Base.:*(a::RE, b::RE)
    if a.head == :zero || b.head == :zero
        c = zero(RE)
    elseif a.head == :one
        c = b
    elseif b.head == :one
        c = a
    else
        c = RE(:mul, a.flag && b.flag, a, b)
    end

    return c
end

function lres(a::RE, b::RE)
    if a.head == :char
        c = lres(a.char, b)
    elseif a.head == :zero
        c = typemax(RE)
    elseif a.head == :one
        c = b
    else
        l = a.left        

        if a.head == :top
            if b.head == :top
                c = typemax(RE)
            else
                error()
            end
        elseif a.head == :star
            error()
        else
            r = a.right

            if a.head == :add
                c = lres(l, b) ∧ lres(r, b)
            elseif a.head == :inf
                c = lres(l, b) + lres(r, b)
            elseif a.head == :mul
                c = lres(r, lres(l, b))
            else
                error()
            end
        end
    end

    return c
end

function lres(a::Char, b::RE)
    if b.head == :char && b.char == a
        c = one(RE)        
    elseif b.head in (:char, :zero, :one)
        c = zero(RE)
    elseif b.head == :top
        c = typemax(RegularExpresson)
    else
        l = b.left

        if b.head == :star
            c = lres(a, l) * a
        else
            r = b.right

            if b.head == :add
                c = lres(a, l) + lres(a, r)
            elseif b.head == :inf
                c = lres(a, l) ∧ lres(a, r)
            elseif b.head == :mul
                if l.flag
                    c = lres(a, l) * r + lres(a, r)
                else
                    c = lres(a, l) * r
                end
            else
                error()
            end
        end
    end
end

function rres(b::RE, a::RE)
    if a.head == :char
        c = lres(a.char, b)
    elseif a.head == :zero
        c = typemax(RE)
    elseif a.head == :one
        c = b
    else
        l = a.left        

        if a.head == :top
            if b.head == :top
                c = typemax(RE)
            else
                error()
            end
        elseif a.head == :star
            error()
        else
            r = a.right

            if a.head == :add
                c = rres(b, l) ∧ rres(b, r)
            elseif a.head == :inf
                c = rres(b, l) + rres(b, r)
            elseif a.head == :mul
                c = rres(rres(b, r), l)
            else
                error()
            end
        end
    end

    return c
end

function rres(b::RE, a::Char)
    if b.head == :char && b.char == a
        c = one(RE)
    elseif b.head in (:char, :zero, :one)
        c = zero(RE)
    elseif b.head == :top
        c = typemax(RegularExpresson)
    else
        l = b.left

        if b.head == :star
            c = b * rres(l, a)
        else
            r = b.right

            if b.head == :add
                c = rres(l, a) + rres(r, a)
            elseif b.head == :inf
                c = rres(l, a) ∧ rres(r, a)
            elseif b.head == :mul
                if r.flag
                    c = rres(l, a) + l * rres(r, a)
                else
                    c = l * rres(r, a)
                end
            else
                error()
            end
        end
    end
end

function Base.zero(::Union{RE, Type{RE}})
    return RE(:zero, false)
end

function Base.one(::Union{RE, Type{RE}})
    return RE(:one, true)
end

function Base.typemax(RE)
    return RE(:top, true)
end

function Base.iszero(a::RE)
    return a.head == :zero
end

function Base.isone(a::RE)
    return a.head == :one
end

function Base.show(io::IO, a::RE)
    print(io, string(a))
    return
end

function Base.transpose(a::RE)
    return a
end

function Base.isapprox(a::AbstractVector{RE}, b::AbstractVector{RE}; kw...)
    return a == b
end

function Base.convert(::Type{RE}, char::AbstractChar)
    return RE(char)
end

function Base.convert(::Type{RE}, str::AbstractString)
    return RE(str)
end
