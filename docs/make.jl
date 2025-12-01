using SemiringFactorizations
using Documenter

using Literate
Literate.markdown("docs/src/examples_lit.jl", "docs/src")


makedocs(;
    modules = [SemiringFactorizations],
    format = Documenter.HTML(),
    sitename = "SemiringFactorizations.jl",
    checkdocs = :none,
    pages = ["SemiringFactorizations.jl" => "index.md", "Library Reference" => "api.md", "Examples" => "examples_lit.md"],
)

deploydocs(;
    target = "build", repo = "github.com/AlgebraicJulia/SemiringFactorizations.jl.git", branch = "gh-pages"
)
