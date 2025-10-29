struct StrictLowerTriangular{T, M <: AbstractMatrix{T}} <: AbstractMatrix{T}
    data::M
end

function Base.parent(matrix::StrictLowerTriangular)
    return matrix.data
end

function Base.copy(matrix::StrictLowerTriangular)
    return StrictLowerTriangular(copy(parent(matrix)))
end

function Base.replace_in_print_matrix(
        matrix::StrictLowerTriangular,
        i::Integer,
        j::Integer,
        string::AbstractString,
    )

    if i <= j
        string = Base.replace_with_centered_mark(string)
    end

    return string
end

# ------------------------ #
# Abstract Array Interface #
# ------------------------ #

function Base.size(matrix::StrictLowerTriangular)
    return size(parent(matrix))
end

@propagate_inbounds function Base.getindex(matrix::StrictLowerTriangular{T}, i::Integer, j::Integer) where {T}
    @boundscheck checkbounds(matrix, i, j)

    if i <= j
        v = zero(T)
    else
        v = parent(matrix)[i, j]     
    end

    return v
end
