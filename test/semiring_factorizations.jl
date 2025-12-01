@testset "sparse" begin
    n = 1000

    for T in (OrAndRel, MinMax{Float64})
        AS = sprand(T, n, n, .005)
        BS = sprand(T, n, n, .005)

        AD = Matrix(AS)
        BD = Matrix(BS)

        @test AS * BS == AD * BD
        @test AS \ BD == AD \ BD
        @test BD / AS == BD / AD
        @test AS + BS == AD + BD
        @test AS ∧ BS == AD ∧ BD

        FS = slu(AS)
        FD = slu(AD)

        @test Matrix(FS) == Matrix(FD)
        @test FS * BD == FD * BD
        @test FS \ BD == FD \ BD
        @test BD / FS == BD / FD
    end
end
