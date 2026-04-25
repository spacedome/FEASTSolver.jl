using Documenter, FEASTSolver

makedocs(;
    modules=[FEASTSolver],
    format=Documenter.HTML(),
    pages=[
        "Home" => "index.md",
    ],
    remotes=nothing,
    sitename="FEASTSolver.jl",
    authors="spacedome",
)

deploydocs(;
    repo="github.com/spacedome/FEASTSolver.jl",
)
