using SemiringFactorizations
using Documenter

makedocs(;
    modules = [SemiringFactorizations],
    format = Documenter.HTML(),
    sitename = "SemiringFactorizations.jl",
    checkdocs = :none,
    pages = ["SemiringFactorizations.jl" => "index.md", "Library Reference" => "api.md"],
)

deploydocs(;
    target = "build", repo = "github.com/AlgebraicJulia/SemiringFactorizations.jl.git", branch = "gh-pages"
)
