struct SemiringSampler{A <: AbstractSemiring, T, S <: Sampler{T}} <: Sampler{SemiringNumber{A, T}}
    smp::S
end

function SemiringSampler{A, T}(smp::S) where {A <: AbstractSemiring, T, S <: Sampler{T}}
    return SemiringSampler{A, T, S}(smp)
end

function Random.Sampler(::Type{R}, ::Type{SemiringNumber{A, T}}, rep::Repetition) where {R <: AbstractRNG, A <: AbstractSemiring, T}
    smp = Sampler(R, T, rep)
    return SemiringSampler{A, T}(smp)
end

function Base.rand(rng::AbstractRNG, smp::SemiringSampler{A}) where {A <: AbstractSemiring}
    num = rand(rng, smp.smp)
    return SemiringNumber{A}(num)
end
