function ⪯(a, b)
    return a <= b || a ≈ b
end

function ⪯(a::AbstractMatrix, b::AbstractMatrix)
    return all(a .⪯ b)
end

function Base.:<(a::AbstractMatrix, b::AbstractMatrix)
    return all(a .< b)
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

    # starred operations
    @test star(a) * b ≈ slmul(a, b)
    @test a * star(b) ≈ srmul(a, b)

    # ternary operations
    @test (a * b) + c ≈ fma(a, b, c)
end

function test_quantale(a, b, c, ∅, ϵ, ⊤)
    test_semiring(a, b, c, ∅, ϵ)

    # ⊤ is the identity for infimum
    @test ⊤ ∧ a ≈ a ∧ ⊤ ≈ a

    # addition and infimum are connected by the absorbtion law
    @test a + (a ∧ b) ≈ a ∧ (a + b) ≈ a

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
    @test (a ⪯ b) == (a ≈ a ∧ b) == (b ≈ a + b)

    # strict ordering agrees with partial ordering
    @test (a < b) == (a != b && a ⪯ b)

    # star is a Kleene star
    @test (a * b ⪯ b) <= (star(a) * b ⪯ b)
    @test (a * b ⪯ a) <= (a * star(b) ⪯ a)

    # starred operations
    @test sldiv(a, b) ≈ star(a) \ b
    @test srdiv(a, b) ≈ a / star(b)

    # ternary operations
    @test (a \ b) ∧ c ≈ fli(a, b, c)
    @test (a / b) ∧ c ≈ fri(a, b, c)

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

@testset "interface" begin
    # semirings
    types = (
        Float64,
        ComplexF64,
    )

    for T in types
        a = rand(T)
        b = rand(T)
        c = rand(T)
        test_semiring(a, b, c, zero(T), one(T))
    end

    # power set lattices (boolean)
    for T in (OrAnd, AndOr)
        a = rand(AndOr{Bool}) |> T
        b = rand(AndOr{Bool}) |> T
        c = rand(AndOr{Bool}) |> T
        test_quantale(a, b, c, zero(a), one(a), typemax(a))

        a = rand(AndOr{Bool}, 10, 10) .|> T
        b = rand(AndOr{Bool}, 10, 10) .|> T
        c = rand(AndOr{Bool}, 10, 10) .|> T
        test_quantale(a, b, c, zero(a), one(a), typemax.(a))
    end

    # power set lattices (unsigned)
    for T in (OrAnd, AndOr)
        a = rand(AndOr{UInt64}) |> T
        b = rand(AndOr{UInt64}) |> T
        c = rand(AndOr{UInt64}) |> T
        test_quantale(a, b, c, zero(a), one(a), typemax(a))

        a = rand(AndOr{UInt64}, 10, 10) .|> T
        b = rand(AndOr{UInt64}, 10, 10) .|> T
        c = rand(AndOr{UInt64}, 10, 10) .|> T
        test_quantale(a, b, c, zero(a), one(a), typemax.(a))
    end

    # relation quantales
    for T in (OrAndRel, AndOrRel)
        a = rand(AndOrRel) |> T
        b = rand(AndOrRel) |> T
        c = rand(AndOrRel) |> T
        test_quantale(a, b, c, zero(a), one(a), typemax(a))

        a = rand(AndOrRel, 10, 10) .|> T
        b = rand(AndOrRel, 10, 10) .|> T
        c = rand(AndOrRel, 10, 10) .|> T
        test_quantale(a, b, c, zero(a), one(a), typemax.(a))
    end

    # chains
    for T in (OrAnd{UInt64}, OrAndRel, MinMax{Float64}, MinPlus{Float64})
        a1 = rand(T); a2 = rand(T) ∧ a1; a3 = rand(T) ∧ a2
        b1 = rand(T); b2 = rand(T) ∧ b1; b3 = rand(T) ∧ b2
        c1 = rand(T); c2 = rand(T) ∧ c1; c3 = rand(T) ∧ c2

        a = Chain((a1, a2, a3))
        b = Chain((b1, b2, b3))
        c = Chain((c1, c2, c3))

        test_quantale(a, b, c, zero(a), one(a), typemax(a))
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
        test_quantale(a, b, c, zero(a), one(a), typemax(a))
    
        a = rand(MinPlus{Float64}, 10, 10) .|> T
        b = rand(MinPlus{Float64}, 10, 10) .|> T
        c = rand(MinPlus{Float64}, 10, 10) .|> T
        test_quantale(a, b, c, zero(a), one(a), typemax.(a))
    end

    # lawvere quantales
    for T in (
            MaxPlusPos{1},
            MaxPlusPos{2},
            MaxPlusPos{3},
            MaxPlusPos{4},
            MinPlusPos{1},
            MinPlusPos{2},
            MinPlusPos{3},
            MinPlusPos{4},
            MaxLSE{1},
            MaxLSE{2},
            MaxLSE{3},
            MaxLSE{4},
            MinLSE{1},
            MinLSE{2},
            MinLSE{3},
            MinLSE{4},
        )
        a = rand(MinPlusPos{1, Float64}) |> T
        b = rand(MinPlusPos{1, Float64}) |> T
        c = rand(MinPlusPos{1, Float64}) |> T
        test_quantale(a, b, c, zero(a), one(a), typemax(a))
    
        a = rand(MinPlusPos{1, Float64}, 10, 10) .|> T
        b = rand(MinPlusPos{1, Float64}, 10, 10) .|> T
        c = rand(MinPlusPos{1, Float64}, 10, 10) .|> T
        test_quantale(a, b, c, zero(a), one(a), typemax.(a))
    end

    for T in (MaxGod, MinGod, MaxGog, MinGog, MaxLuk, MinLuk, MaxFod, MinFod)
        a = rand() |> T
        b = rand() |> T
        c = rand() |> T
        test_quantale(a, b, c, zero(a), one(a), typemax(a))
    
        a = rand(10, 10) .|> T
        b = rand(10, 10) .|> T
        c = rand(10, 10) .|> T
        test_quantale(a, b, c, zero(a), one(a), typemax.(a))
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
        test_quantale(a, b, c, zero(a), one(a), typemax(a))

        a = rand(0:9, 10, 10) .// rand(1:9, 10, 10) .|> T
        b = rand(0:9, 10, 10) .// rand(1:9, 10, 10) .|> T
        c = rand(0:9, 10, 10) .// rand(1:9, 10, 10) .|> T
        test_quantale(a, b, c, zero(a), one(a), typemax.(a))

    end
end
