using Documenter
using Literate: markdown
using SemiringFactorizations

markdown("src/examples/scheduling.jl", "src")

makedocs(;
    modules = [SemiringFactorizations],
    format = Documenter.HTML(),
    sitename = "SemiringFactorizations.jl",
    checkdocs = :none,
    pages = ["SemiringFactorizations.jl" => "index.md", "Library Reference" => "api.md", "Examples" => "scheduling.md"],
)

deploydocs(;
    target = "build", repo = "github.com/AlgebraicJulia/SemiringFactorizations.jl.git", branch = "gh-pages"
)
