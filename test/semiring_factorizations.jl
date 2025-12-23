@testset "sparse" begin
    n = 100

    for T in (OrAndRel, MinMax{Float64})
        AS = sprand(T, n, n, 0.1)
        BS = sprand(T, n, n, 0.1)

        AD = Matrix(AS)
        BD = Matrix(BS)

        @test AS * BS == AD * BD
        @test AS \ BD == AD \ BD
        @test BD / AS == BD / AD
        @test AS + BS == AD + BD
        @test AS & BS == AD & BD

        FS = slu(AS)
        FD = slu(AD)

        @test Matrix(FS) == Matrix(FD)
        @test FS * BD == FD * BD
        @test BD * FS == BD * FD
        @test FS \ BD == FD \ BD
        @test BD / FS == BD / FD

        # Jacobi (dense)
        @test jacobi(AD, BD, Val(:N), Val(:L)) == FD * BD
        @test jacobi(AD, BD, Val(:N), Val(:R)) == BD * FD
        @test jacobi(AD, BD, Val(:C), Val(:L)) == FD \ BD
        @test jacobi(AD, BD, Val(:C), Val(:R)) == BD / FD

        # Jacobi (sparse)
        @test jacobi(AS, BS, Val(:N), Val(:L)) == FS * BS
        @test jacobi(AS, BS, Val(:N), Val(:R)) == BS * FS
        @test jacobi(AS, BS, Val(:C), Val(:L)) == FS \ BS
        @test jacobi(AS, BS, Val(:C), Val(:R)) == BS / FS
    end
end

@testset "Newton" begin
    n = 10

    for T in (AndOr{UInt64}, MinMax{Float64})
        A = rand(T, n)
        B = rand(T, n, n)
        C = rand(T, n, n, n)
        D = rand(T, n, n, n, n)
        #
        #   x = A
        #
        x = newton(A)
        @test x == first(horner(x, A))
        #
        #   x = A + Bx
        #
        x = newton(A, B)
        @test x == first(horner(x, A, B)) 
        #
        #   x = A + Bx + Cxx
        #
        x = newton(A, B, C)
        @test x == first(horner(x, A, B, C)) 
        #
        #   x = A + Bx + Cxx + Dxxx
        #
        x = newton(A, B, C, D)
        @test x == first(horner(x, A, B, C, D))
    end
end
