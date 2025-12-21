function Base.:≈(a::AbstractMatrix, b::AbstractMatrix)
    return all(splat(≈), zip(a, b))
end

function Base.:<(a::AbstractMatrix, b::AbstractMatrix)
    return all(splat(<), zip(a, b))
end

function ⪯(a::AbstractMatrix, b::AbstractMatrix)
    return all(splat(⪯), zip(a, b))
end

function ⪯(a, b)
    return (a < b) || (a ≈ b)
end

function test_semiring(a, b, c, ∅, ϵ)
    # 1 is the multiplicative identity
    @test ϵ * a ≈ a * ϵ ≈ a

    # 0 is the additive identity
    @test ∅ + a ≈ a + ∅ ≈ a

    # 0 is absorbing
    @test ∅ * a ≈ a * ∅ ≈ ∅

    # multiplication associates
    @test (a * b) * c ≈ a * (b * c)

    # addition associates
    @test (a + b) + c ≈ a + (b + c)

    # addition commutes
    @test a + b ≈ b + a

    # multiplication right-distributes over addition
    @test (a + b) * c ≈ a * c + b * c

    # multiplication left-distributes over addition
    @test a * (b + c) ≈ a * b + a * c

    # star is a quasi-invere
    @test a * star(a) + ϵ ≈ star(a) * a + ϵ ≈ star(a)
end

function test_quantale(a, b, c, ∅, ϵ, ⊤)
    test_semiring(a, b, c, ∅, ϵ)

    # ⊤ is the identity for infimum
    @test ⊤ & a ≈ a & ⊤ ≈ a

    # addition and infimum are connected by the absorbtion law
    @test a + (a & b) ≈ a & (a + b) ≈ a

    # lattice is residuated
    @test (a * b ⪯ c) == (b ⪯ a \ c) == (a ⪯ c / b)

    # partial order is reflexive
    @test a <= a

    # strict order is irreflexive
    @test !(a < a)

    # partial order is transitive
    @test (a ⪯ b && b ⪯ c) <= (a ⪯ c)

    # strict order is transitive
    @test (a < b && b < c) <= (a < c)

    # partial ordering agrees with lattice structure
    @test (a ⪯ b) == (a ≈ a & b) == (b ≈ a + b)

    # strict ordering agrees with partial ordering
    @test (a < b) == (a != b && a ⪯ b)

    # star is a Kleene star
    @test (a * b ⪯ b) <= (star(a) * b ⪯ b)
    @test (a * b ⪯ a) <= (a * star(b) ⪯ a)

    # Quantales and their Applications
    # Rosenthal
    # Proposition 2.1.1
    @test a * (a \ b) ⪯ b
    @test (b / a) * a ⪯ b
    @test (b \ (a \ c)) ≈ (a * b) \ c
    @test (a \ (c / b)) ≈ (a \ c) / b

    # Generic Inference
    # Pouly and Kohlas
    # Lemma 6.8
    @test ϵ ⪯ star(a)
    @test a ⪯ star(a)
    @test star(∅) ≈ ϵ ≈ star(ϵ)
    @test star(a) * star(a) ≈ star(a)
    @test star(star(a)) ≈ star(a)
    @test (a ⪯ b) <= (star(a) ⪯ star(b))

    # Lemma 6.10
    @test star(a + b) ≈ star(a) * star(b * star(a))

    # Lemma 6.11
    @test star(a + b) ≈ star(star(a) + b) ≈ star(a + star(b)) ≈ star(star(a) + star(b))
end

function test_girard(a, b, c, ∅, ϵ, ⊥, ⊤)
    test_quantale(a, b, c, ∅, ϵ, ⊤)

    # dual operations
    @test a'' ≈ a

    @test a * b ≈ (b' ⅋ a')'
    @test a ⅋ b ≈ (b' * a')'    

    @test a + b ≈ (a' & b')'
    @test a & b ≈ (a' + b')'

    @test ϵ' ≈ ⊥
    @test ⊥' ≈ ϵ

    @test ∅' ≈ ⊤
    @test ⊤' ≈ ∅

    @test a \ b ≈ a' ⅋ b
    @test a / b ≈ a ⅋ b' 

    # cyclicity
    ⊥ / a ≈ a \ ⊥ ≈ a'

    # distributivity
    @test a ⅋ (b & c) ≈ (a ⅋ b) & (a ⅋ c)
    @test (a & b) ⅋ c ≈ (a ⅋ c) & (b ⅋ c)
    @test a \ (b & c) ≈ (a \ b) & (a \ c)
    @test (a + b) \ c ≈ (a \ c) & (b \ c)
end

@testset "interface" begin
    # nonnegative real numbers
    a = rand()
    b = rand()
    c = rand()
    test_semiring(a, b, c, 0.0, 1.0)

    a = rand(10, 10) * 0.1
    b = rand(10, 10) * 0.1
    c = rand(10, 10) * 0.1
    test_semiring(a, b, c, zero(a), one(a))

    # power set lattices (boolean)
    for T in (OrAnd, AndOr)
        a = rand(AndOr{Bool}) |> T
        b = rand(AndOr{Bool}) |> T
        c = rand(AndOr{Bool}) |> T
        test_girard(a, b, c, zero(a), one(a), one(a)', typemax(a))

        a = rand(AndOr{Bool}, 10, 10) .|> T
        b = rand(AndOr{Bool}, 10, 10) .|> T
        c = rand(AndOr{Bool}, 10, 10) .|> T
        test_girard(a, b, c, zero(a), one(a), one(a)', typemax.(a))
    end

    # power set lattices (unsigned)
    for T in (OrAnd, AndOr)
        a = rand(AndOr{UInt64}) |> T
        b = rand(AndOr{UInt64}) |> T
        c = rand(AndOr{UInt64}) |> T
        test_girard(a, b, c, zero(a), one(a), one(a)', typemax(a))

        a = rand(AndOr{UInt64}, 10, 10) .|> T
        b = rand(AndOr{UInt64}, 10, 10) .|> T
        c = rand(AndOr{UInt64}, 10, 10) .|> T
        test_girard(a, b, c, zero(a), one(a), one(a)', typemax.(a))
    end

    # relation quantales
    for T in (OrAndRel, AndOrRel)
        a = rand(AndOrRel) |> T
        b = rand(AndOrRel) |> T
        c = rand(AndOrRel) |> T
        test_girard(a, b, c, zero(a), one(a), one(a)', typemax(a))

        a = rand(AndOrRel, 10, 10) .|> T
        b = rand(AndOrRel, 10, 10) .|> T
        c = rand(AndOrRel, 10, 10) .|> T
        test_girard(a, b, c, zero(a), one(a), one(a)', typemax.(a))
    end

    # bottleneck lattices
    for T in (MaxMin, MinMax)
        a = rand(MinMax{Int}) |> T
        b = rand(MinMax{Int}) |> T
        c = rand(MinMax{Int}) |> T
        test_quantale(a, b, c, zero(a), one(a), typemax(a))
    
        a = rand(MinMax{Int}, 10, 10) .|> T
        b = rand(MinMax{Int}, 10, 10) .|> T
        c = rand(MinMax{Int}, 10, 10) .|> T
        test_quantale(a, b, c, zero(a), one(a), typemax.(a))
    end

    # tropical quantales
    for T in (MaxMul, MinMul, MaxPlus, MinPlus)
        a = rand(MinPlus{Float64}) |> T
        b = rand(MinPlus{Float64}) |> T
        c = rand(MinPlus{Float64}) |> T
        test_girard(a, b, c, zero(a), one(a), one(a)', typemax(a))
    
        a = rand(MinPlus{Float64}, 10, 10) .|> T
        b = rand(MinPlus{Float64}, 10, 10) .|> T
        c = rand(MinPlus{Float64}, 10, 10) .|> T
        test_girard(a, b, c, zero(a), one(a), one(a)', typemax.(a))
    end

    # lawvere quantales
    for T in (
            MaxPlusPos,
            MinPlusPos,
            MaxLSE,
            MinLSE,
        )
        a = rand(MinPlusPos{Float64}) |> T
        b = rand(MinPlusPos{Float64}) |> T
        c = rand(MinPlusPos{Float64}) |> T
        test_quantale(a, b, c, zero(a), one(a), typemax(a))
    
        a = rand(MinPlusPos{Float64}, 10, 10) .|> T
        b = rand(MinPlusPos{Float64}, 10, 10) .|> T
        c = rand(MinPlusPos{Float64}, 10, 10) .|> T
        test_quantale(a, b, c, zero(a), one(a), typemax.(a))
    end

    for T in (MaxGod, MinGod, MaxGog, MinGog)
        a = rand() |> T
        b = rand() |> T
        c = rand() |> T
        test_quantale(a, b, c, zero(a), one(a), typemax(a))
    
        a = rand(10, 10) .|> T
        b = rand(10, 10) .|> T
        c = rand(10, 10) .|> T
        test_quantale(a, b, c, zero(a), one(a), typemax.(a))
    end

    for T in (MaxLuk, MinLuk, MaxFod, MinFod)
        a = rand() |> T
        b = rand() |> T
        c = rand() |> T
        test_girard(a, b, c, zero(a), one(a), one(a)', typemax(a))
    
        a = rand(10, 10) .|> T
        b = rand(10, 10) .|> T
        c = rand(10, 10) .|> T
        test_girard(a, b, c, zero(a), one(a), one(a)', typemax.(a))
    end

    for T in (GCDMulPos,)
        a = rand(0:9) |> T
        b = rand(0:9) |> T
        c = rand(0:9) |> T
        test_quantale(a, b, c, zero(a), one(a), typemax(a))

        a = rand(0:9, 10, 10) .|> T
        b = rand(0:9, 10, 10) .|> T
        c = rand(0:9, 10, 10) .|> T
        test_quantale(a, b, c, zero(a), one(a), typemax.(a))
    end

    for T in (LCMMul, GCDMul)
        a = rand(0:9) // rand(1:9) |> T
        b = rand(0:9) // rand(1:9) |> T
        c = rand(0:9) // rand(1:9) |> T
        test_girard(a, b, c, zero(a), one(a), one(a)', typemax(a))

        a = rand(0:9, 10, 10) .// rand(1:9, 10, 10) .|> T
        b = rand(0:9, 10, 10) .// rand(1:9, 10, 10) .|> T
        c = rand(0:9, 10, 10) .// rand(1:9, 10, 10) .|> T
        test_girard(a, b, c, zero(a), one(a), one(a)', typemax.(a))

    end
end
