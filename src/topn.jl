struct TopN{N, T} <: AbstractVector{T}
    tup::NTuple{N, T}

    function TopN{N, T}(tup::NTuple{N}) where {N, T}
        return new{N, T}(tup)
    end
end

const Top2 = TopN{2}
const Top3 = TopN{3}
const Top4 = TopN{4}
const Top5 = TopN{5}

function TopN{N, T}(tup...) where {N, T}
    return TopN{N, T}(tup)
end

function TopN{N, T}(tup::NTuple{M}) where {T, N, M}
    new = ntuple(N) do i
        if i <= M
            c = convert(T, tup[i])
        else
            c = zero(T)
        end

        return c    
    end

    return TopN{N, T}(new)
end

function Base.convert(::Type{TopN{N, T}}, tup::TopN{N, T}) where {N, T}
    return tup
end

function Base.convert(::Type{TopN{N, T}}, tup) where {N, T}
    return TopN{N, T}(tup)
end

function Base.:+(a::TopN{N, T}, b::TopN{N, T}) where {N, T}
    tup = a.tup; i = j = 1

    for k in 1:N
        if a[i] >= b[j]
            @reset tup[k] = a[i]
            i += 1
        else
            @reset tup[k] = b[j]
            j += 1
        end
    end

    return TopN{N, T}(tup)
end

function Base.:*(a::TopN{N, T}, b::TopN{N, T}) where {N, T}
    tup = a.tup; j = ntuple(_ -> 1, N)

    for k in 1:N
        imax = 1

        for i in 1:N
            if a[i] * b[j[i]] > a[imax] * b[j[imax]]
                imax = i
            end
        end

        @reset tup[k] = a[imax] * b[j[imax]]
        @reset j[imax] += 1
    end
        
    return TopN{N, T}(tup)
end

function Base.typemax(::Type{TopN{N, T}}) where {N, T}
    tup = ntuple(_ -> typemax(T), N)
    return TopN{N, T}(tup)    
end

function Base.typemax(::T) where {T <: TopN}
    return typemax(T)
end

function Base.typemin(::Type{T}) where {T <: TopN}
    return zero(T)
end

function Base.typemin(::T) where {T <: TopN}
    return typemin(T)
end

function Base.one(::Type{TopN{N, T}}) where {N, T}
    return TopN{N, T}(one(T))
end

function Base.one(::T) where {T <: TopN}
    return one(T)
end

function Base.zero(::Type{TopN{N, T}}) where {N, T}
    return TopN{N, T}(zero(T))
end

function Base.zero(::T) where {T <: TopN}
    return zero(T)
end

# ------------------------ #
# Abstract Array Interface #
# ------------------------ #

function Base.IndexStyle(::Type{<:TopN})
    return IndexLinear()
end

function Base.size(a::TopN)
    return (length(a.tup),)
end

function Base.getindex(a::TopN, i)
    return a.tup[i]
end
