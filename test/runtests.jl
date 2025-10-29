using Graphs
using LinearAlgebra
using MatrixMarket
using SemiringFactorizations
using SparseArrays
using SuiteSparseMatrixCollection
using Test
using TropicalNumbers

function readmatrix(name::String)
    ssmc = ssmc_db()
    path = joinpath(fetch_ssmc(ssmc[ssmc.name .== name, :]; format="MM")[1], "$(name).mtx")
    return mmread(path)
end

@testset "(ℝ ∪ {+∞}, +, ×, 0, 1)" begin
    for name in ("swang1", "wang1", "wang3", "ex1", "ex10", "ex10hs")
        A = readmatrix(name)
        b = rand(size(A, 1))
        @test sldiv!(A, copy(b)) ≈ (I - A) \ b
    end
end

@testset "(ℝ ∪ {-∞, +∞}, ∧, +, +∞, 0)" begin
    A = TropicalMinPlusF64[
        Inf 9.0 8.0 Inf
        Inf Inf 6.0 Inf
        Inf Inf Inf 7.0
        5.0 Inf Inf Inf
    ]

    @test sinv(A) == TropicalMinPlusF64[
         0.0  9.0  8.0 15.0
        18.0  0.0  6.0 13.0
        12.0 21.0  0.0  7.0
         5.0 14.0 13.0  0.0
    ]

    A = TropicalMinPlusF64[
        Inf 7.0 1.0
        4.0 Inf Inf
        Inf 2.0 Inf
    ]

    @test sinv(A) == TropicalMinPlusF64[
        0.0 3.0 1.0
        4.0 0.0 5.0
        6.0 2.0 0.0     
    ]

    for name in ("saylr1", "08blocks", "GD01_a", "cage7", "gre_343", "CAG_mat364")
        A = readmatrix(name)
        A[diagind(A)] .= 0

        B = SparseMatrixCSC(
            size(A)...,
            A.colptr,
            A.rowval,
            TropicalMinPlusF64.(A.nzval),
        )

        g = DiGraph(A)

        D1 = sinv(B)
        D2 = floyd_warshall_shortest_paths(g, A).dists

        for j in size(A, 2), i in size(A, 1)
            D1ij = D1[i, j]
            D2ij = D2[i, j]

            if D1ij.n == Inf
                @test D2ij > 1e10
            elseif D1ij.n == -Inf
                @test D2ij < -1e10
            else
                @test D2ij ≈ D1ij.n
            end
        end
    end
end 

@testset "([0, 1], ∨, ×, 0, 1)" begin
    A = TropicalMaxMulF64[
        0.0 0.6 0.4
        0.0 0.1 0.9
        0.0 1.0 0.0
    ]

    @test sinv(A) == TropicalMaxMulF64[
        1.0 0.6 0.54
        0.0 1.0 0.9
        0.0 1.0 1.0     
    ]
end
