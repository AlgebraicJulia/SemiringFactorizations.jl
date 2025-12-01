using LinearAlgebra, VectorizationBase, LoopVectorization
using VectorizationBase: OffsetPrecalc, StaticBool, Bit, static, NativeTypes, Index, gep_quote, VectorIndex,
    AbstractMask, NativeTypesExceptBit, AbstractSIMDVector, IndexNoUnroll, AbstractStridedPointer, AbstractSIMD
using VectorizationBase: contiguous_batch_size, contiguous_axis, val_stride_rank, bytestrides, offsets, memory_reference,
    vmaximum, fmap, FloatingTypes, IntegerIndex, LazyMulAdd, _vstore!, __vstore!, _vload, __vload, _vbroadcast, OffsetPrecalc, StridedPointer, _gep
using Base: @ntuple
using Octavian: zstridedpointer, preserve_buffer, matmul_sizes, matmul_params, dontpack, matmul_st_pack_dispatcher!,
    loopmul!, inlineloopmul!, maybeinline, matmul_only_β!, One, Zero, ArrayInterface, block_sizes, __matmul!
using Octavian.ArrayInterface: is_column_major
using LoopVectorization: check_type

const MulMatrix_1{T} = Union{
    Matrix{T},
    FixedSizeMatrix{T},
}

const MulMatrix_2{T} = Union{
    MulMatrix_1{T},
    SubArray{T, 2, <:MulMatrix_1{T}},
}

const MulMatrix_3{T} = Union{
    MulMatrix_2{T},
    Transpose{T, <:MulMatrix_2{T}},
    Adjoint{T, <:MulMatrix_2{T}},
}

const MulMatrix = MulMatrix_3

#=
collapse_add_alg(::Type{<:Union{MaxPlusQuantale, MaxMinLattice}}, args...) = VectorizationBase.collapse_max(args...)
collapse_add_alg(::Type{<:Union{MinPlusQuantale, MinMaxLattice}}, args...) = VectorizationBase.collapse_min(args...)
collapse_add_alg(::Type{<:AndOrLattice}, args...) = VectorizationBase.collapse_or(args...)

contract_add_alg(::Type{<:Union{MaxPlusQuantale, MaxMinLattice}}, args...) = VectorizationBase.contract_max(args...)
contract_add_alg(::Type{<:Union{MinPlusQuantale, MinMaxLattice}}, args...) = VectorizationBase.contract_min(args...)
contract_add_alg(::Type{<:AndOrLattice}, args...) = VectorizationBase.contract_or(args...)

reduced_add_alg(::Type{<:Union{MaxPlusQuantale, MaxMinLattice}}, args...) = VectorizationBase.reduced_max(args...)
reduced_add_alg(::Type{<:Union{MinPlusQuantale, MinMaxLattice}}, args...) = VectorizationBase.reduced_min(args...)
reduced_add_alg(::Type{<:AndOrLattice}, args...) = VectorizationBase.reduced_or(args...)

vsum_alg(::Type{<:Union{MaxPlusQuantale, MaxMinLattice}}, args...) = VectorizationBase.vmaximum(args...)
vsum_alg(::Type{<:Union{MinPlusQuantale, MinMaxLattice}}, args...) = VectorizationBase.vminimum(args...)
vsum_alg(::Type{<:AndOrLattice}, args...) = VectorizationBase.vany(args...)

@inline function VectorizationBase.collapse_add(vu::SemiringNumber{Q, VecUnroll{N,W,T,V}}) where {N,W,T,V,Q<:AbstractSemiring}
    SemiringNumber{Q}(collapse_add_alg(Q, parent(vu)))
end

@inline function VectorizationBase.contract_add(vu::SemiringNumber{Q, VecUnroll{N,W,T,V}}, ::StaticInt{K}) where {N,W,T,V,K,Q<:AbstractSemiring}
    SemiringNumber{Q}(contract_add_alg(Q, parent(vu), StaticInt{K}()))
end

@inline function VectorizationBase.reduced_add(x::SemiringNumber{Q}, y::SemiringNumber{Q}) where {Q <: AbstractSemiring}
    SemiringNumber{Q}(reduced_add_alg(Q, parent(x), parent(y)))
end

@inline VectorizationBase.vsum(x::SemiringNumber{Q, <:AbstractSIMD}) where {Q <: AbstractSemiring} = SemiringNumber{Q}(vsum_alg(Q, parent(x)))
=#

function LoopVectorization.check_args(
        ::Type{SemiringNumber{A, T}},
        ::Type{SemiringNumber{A, T}},
    ) where {A <: AbstractSemiring, T}
    return true
end

function LoopVectorization.check_type(::Type{SemiringNumber{A, T}}) where {A <: AbstractSemiring, T}
    return check_type(T)
end

function VectorizationBase._vstore!(
        ptr::AbstractStridedPointer,
        vu::AbstractSemiringNumber{<:VecUnroll{<:Any, W}},
        u::Unroll{<:Any, <:Any, <:Any, <:Any, W},
        a::StaticBool,
        s::StaticBool,
        nt::StaticBool,
        si::StaticInt,
    ) where {W}
    return _vstore!(nosemiring(ptr), parent(vu), u, a, s, nt, si)
end

function VectorizationBase._vstore!(
        ptr::AbstractStridedPointer,
        vu::AbstractSemiringNumber{<:VecUnroll{<:Any, W}},
        u::Unroll{<:Any, <:Any, <:Any, <:Any, W},
        m::AbstractMask{W},
        a::StaticBool,
        s::StaticBool,
        nt::StaticBool,
        si::StaticInt,
    ) where {W}
    return _vstore!(nosemiring(ptr), parent(vu), u, m, a, s, nt, si)
end

#=
@inline function VectorizationBase._vstore!(
    g::G, ptr::AbstractStridedPointer{T,D,C}, vu::SemiringNumber{Q, <:VecUnroll{U,W}}, u::Unroll{AU,F,N,AV,1,M,X,I}, a::A, s::S, nt::NT, si::StaticInt{RS}
) where {T,D,C,U,AU,F,N,W,M,I,G<:Function,AV,A<:StaticBool, S<:StaticBool, NT<:StaticBool, RS,X,Q<:AbstractSemiring}
    VectorizationBase._vstore!(g, nosemiring(ptr), parent(vu), u, a, s, nt, si)
end
=#

#=
@inline function VectorizationBase.__vstore!(
    f::F, ptr::Ptr{SemiringNumber{Q, T}}, v::SemiringNumber{Q, T}, i::IntegerIndex, a::A, s::S, nt::NT, si::StaticInt{RS}
) where {T<:NativeTypesExceptBit, F<:Function,A<:StaticBool,S<:StaticBool,NT<:StaticBool,RS,Q<:AbstractSemiring}
    VectorizationBase.__vstore!(f, Ptr{T}(ptr), parent(v), i, a, s, nt, si)
end
=#

function VectorizationBase.__vstore!(
        ptr::Ptr{SemiringNumber{A, T}},
        v::SemiringNumber{A, <:Vec},
        i::Index,
        m::AbstractMask,
        a::StaticBool,
        s::StaticBool,
        nt::StaticBool,
        si::StaticInt,
    ) where {A <: AbstractSemiring, T <: NativeTypesExceptBit}
    return __vstore!(Ptr{T}(ptr), parent(v), i, m, a, s, nt, si)
end

function VectorizationBase.__vstore!(
        ptr::Ptr{SemiringNumber{A, T}},
        v::SemiringNumber{A, <:Vec},
        i::VectorIndex,
        a::StaticBool,
        s::StaticBool,
        nt::StaticBool,
        si::StaticInt
    ) where {A <: AbstractSemiring, T}
    return __vstore!(Ptr{T}(ptr), parent(v), i, a, s, nt, si)
end

function VectorizationBase.__vload(
        ptr::Ptr{SemiringNumber{A, T}},
        i::Index,
        m::AbstractMask,
        a::StaticBool,
        si::StaticInt,
    )  where {A <: AbstractSemiring, T <: NativeTypes}
    num = __vload(Ptr{T}(ptr), i, m, a, si)
    return SemiringNumber{A}(num)
end

function VectorizationBase.__vload(
        ptr::Ptr{SemiringNumber{A, T}},
        i::Index,
        a::StaticBool,
        si::StaticInt,
    ) where {A <: AbstractSemiring, T <: NativeTypes}
    num = __vload(Ptr{T}(ptr), i, a, si)
    return SemiringNumber{A}(num)
end

function VectorizationBase._vbroadcast(
        a::Union{Val, StaticInt},
        s::SemiringNumber{A},
        si::StaticInt,
    ) where {A <: AbstractSemiring}
    num = _vbroadcast(a, parent(s), si)
    return SemiringNumber{A}(num)
end

function VectorizationBase._vload(
        ptr::AbstractStridedPointer{<:SemiringNumber{A}},
        u::Unroll,
        a::StaticBool,
        si::StaticInt,
    ) where {A <: AbstractSemiring}
    num = _vload(nosemiring(ptr), u, a, si)
    return SemiringNumber{A}(num)
end

function VectorizationBase._vload(
        ptr::AbstractStridedPointer{<:SemiringNumber{A}},
        u::Unroll,
        m::AbstractMask,
        a::StaticBool,
        si::StaticInt,
    ) where {A <: AbstractSemiring}
    num = _vload(nosemiring(ptr), u, m, a, si)
    SemiringNumber{A}(num)
end

function nosemiring(ptr::StridedPointer{<:AbstractSemiringNumber{T}, <:Any, <:Any, B}) where {T, B}
    return stridedpointer(Ptr{T}(ptr.p), ptr.si, StaticInt{B}())
end

function nosemiring(ptr::OffsetPrecalc{<:SemiringNumber})
    return OffsetPrecalc(nosemiring(ptr.ptr), ptr.precalc)
end

@generated function VectorizationBase.zero_vecunroll(
        ::StaticInt{N},
        w::StaticInt,
        ::Type{SemiringNumber{A, T}},
        si::StaticInt,
    ) where {A <: AbstractSemiring, T, N}
    quote
        val = _vbroadcast(w, parent(zero(SemiringNumber{A, T})), si)
        tup = @ntuple $N _ -> val
        num = VecUnroll(tup)
        return SemiringNumber{A}(num)
    end
end

function VectorizationBase._vzero(
        w::StaticInt,
        ::Type{SemiringNumber{A, T}},
        si::StaticInt,
    ) where {A <: AbstractSemiring, T}
    num = _vbroadcast(w, parent(zero(SemiringNumber{A, T})), si)
    return SemiringNumber{A}(num)
end

#=
function VectorizationBase._gep(
        ptr::Ptr{SemiringNumber{A, T}},
        i::StaticInt,
        si::StaticInt,
    ) where {A <: AbstractSemiring, T <: NativeTypes}
    return Ptr{SemiringNumber{A, T}}(_gep(Ptr{T}(ptr), i, si))
end
=#

function VectorizationBase._gep(
        ptr::Ptr{SemiringNumber{A, T}},
        i::IntegerIndex,
        si::StaticInt,
    ) where {A <: AbstractSemiring, T <: NativeTypes}
    return Ptr{SemiringNumber{A, T}}(_gep(Ptr{T}(ptr), i, si))
end

function VectorizationBase._gep(
        ptr::Ptr{SemiringNumber{A, T}},
        i::LazyMulAdd{<:Any, <:Any, <:Integer},
        si::StaticInt,
    ) where {A <: AbstractSemiring, T <: NativeTypes}
    return Ptr{SemiringNumber{A, T}}(_gep(Ptr{T}(ptr), i, si))
end

function VectorizationBase.VecUnroll(data::Tuple{SemiringNumber{A, T}, Vararg{SemiringNumber{A, T}}}) where {A <: AbstractSemiring, T}
    num = VecUnroll(map(parent, data))
    return SemiringNumber{A}(num)
end

#@inline LoopVectorization.vecmemaybe(x::AbstractSemiringNumber) = x

#=
@inline function VectorizationBase.ifelse(f::F, m::AbstractMask, v1::SemiringNumber{Q}, v2::SemiringNumber{Q}, v3::SemiringNumber{Q}) where {F<:Function,Q<:AbstractSemiring}
    SemiringNumber{Q}(VectorizationBase.ifelse(m, parent(f(v1, v2, v3)), parent(v3)))
end
=#

#=
@inline function VectorizationBase.vifelse(f::F, m::AbstractMask, a::SemiringNumber{Q}, b::SemiringNumber{Q}, c::SemiringNumber{Q}) where {F<:Function,Q<:AbstractSemiring}
    VectorizationBase.vifelse(m, f(a, b, c), c)
end
=#

#=
@inline function VectorizationBase.vifelse(m::AbstractMask, a::SemiringNumber{Q}, b::SemiringNumber{Q}) where {Q<:AbstractSemiring}
    SemiringNumber{Q}(VectorizationBase.vifelse(m, parent(a), parent(b)))
end
=#

# Overwrite the `mul!` in LinearAlgebra (also changes the behavior of `*` in Base)!

function LinearAlgebra.mul!(
        C::MulMatrix{SemiringNumber{Q, T}},
        A::MulMatrix{SemiringNumber{Q, T}},
        B::MulMatrix{SemiringNumber{Q, T}},
        α::Number,
        β::Number,
    ) where {Q <: AbstractSemiring, T <: NativeTypes}
    α = convert_or_static(Q, T, α)
    β = convert_or_static(Q, T, β)
    return matmul!(C, A, B, α, β)
end

function convert_or_static(::Type{A}, ::Type{T}, a::SemiringNumber{A}) where {A <: AbstractSemiring, T}
    return convert(SemiringNumber{A, T}, a)
end

function convert_or_static(::Type, ::Type, a::Union{Bool, StaticInt})
    return ifelse(isone(a), StaticInt{1}(), StaticInt{0}())
end

function Octavian._matmul!(C::AbstractMatrix{T}, A, B, α, β, nthread, MKN) where {T <: AbstractSemiringNumber{<:NativeTypes}}
    M, K, N = isnothing(MKN) ? matmul_sizes(C, A, B) : MKN

    if iszero(M * N)
        return
    elseif iszero(K)
        matmul_only_β!(C, β)
        return
    end

    W = pick_vector_width(T)
    pA = zstridedpointer(A); pB = zstridedpointer(B); pC = zstridedpointer(C);
    Cb = preserve_buffer(C); Ab = preserve_buffer(A); Bb = preserve_buffer(B);
    m, n = matmul_params(Val(T))

    GC.@preserve Cb Ab Bb begin
        if maybeinline(M, N, T, is_column_major(A))
            inlineloopmul!(pC, pA, pB, One(), Zero(), M, K, N)
            return
        else
            (n ≥ N) && @goto LOOPMUL

            if (Sys.ARCH === :x86_64) || (Sys.ARCH === :i686)
                (M * K * N < (StaticInt{4_096}() * W)) && @goto LOOPMUL
            else
                (M * K * N < (StaticInt{32_000}() * W)) && @goto LOOPMUL
            end

            __matmul!(pC, pA, pB, α, β, M, K, N, nthread)
            return

            @label LOOPMUL
            loopmul!(pC, pA, pB, α, β, M, K, N)
            return
        end
    end
end

function Octavian._matmul_serial!(C::AbstractMatrix{T}, A::AbstractMatrix, B::AbstractMatrix, α, β, MKN) where {T <: SemiringNumber}
    M, K, N = isnothing(MKN) ? matmul_sizes(C, A, B) : MKN

    if iszero(M * N)
        return
    elseif iszero(K)
        matmul_only_β!(C, β)
        return
    end

    pA = zstridedpointer(A); pB = zstridedpointer(B); pC = zstridedpointer(C);
    Cb = preserve_buffer(C); Ab = preserve_buffer(A); Bb = preserve_buffer(B);
    Mc, Kc, Nc = block_sizes(Val(T))
    m, n = matmul_params(Val(T))

    GC.@preserve Cb Ab Bb begin
        if maybeinline(M, N, T, is_column_major(A))
            inlineloopmul!(pC, pA, pB, One(), Zero(), M, K, N)
            return
        elseif (n >= N) || dontpack(pA, M, K, Mc, Kc, T)
            loopmul!(pC, pA, pB, α, β, M, K, N)
            return
        else
            matmul_st_pack_dispatcher!(pC, pA, pB, α, β, M, K, N)
            return
        end
    end
end

function Octavian._matmul!(c::AbstractVector{T}, A::AbstractMatrix, b::AbstractVector, α, β, MKN, contig_axis) where {T <: SemiringNumber}
    @tturbo for m in indices((A, c), 1)
        cm = zero(T)

        for n in indices((A, b), (2, 1))
            cm += A[m, n] * b[n]
        end

        c[m] = α * cm + β * c[m]
    end

    return c
end

function Octavian._matmul_serial!(y::AbstractVector{T}, A::AbstractMatrix, x::AbstractVector, α, β, MKN) where {T <: SemiringNumber}
    @turbo for m in indices((A, c), 1)
        cm = zero(T)

        for n in indices((A, b), (2, 1))
            cm += A[m, n] * b[n]
        end

        c[m] = α * cm + β * c[m]
    end

    return c
end

#Octavian.matmul_params(::Val{T}) where {T <: AbstractSemiringNumber} = LoopVectorization.matmul_params()
#@inline Octavian.incrementp(A::AbstractStridedPointer{<:AbstractSemiringNumber,3}, a::Ptr) = VectorizationBase.increment_ptr(A, a, (Zero(), Zero(), One()))
#@inline Octavian.increment2(B::AbstractStridedPointer{<:AbstractSemiringNumber,2}, b::Ptr, ::StaticInt{nᵣ}) where {nᵣ} = VectorizationBase.increment_ptr(B, b, (Zero(), StaticInt{nᵣ}()))
#@inline Octavian.increment1(C::AbstractStridedPointer{<:AbstractSemiringNumber,2}, c::Ptr, ::StaticInt{mᵣW}) where {mᵣW} = VectorizationBase.increment_ptr(C, c, (StaticInt{mᵣW}(), Zero()))
