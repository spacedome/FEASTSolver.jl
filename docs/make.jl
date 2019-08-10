using Documenter, FEASTSolver

makedocs(;
    modules=[FEASTSolver],
    format=Documenter.HTML(),
    pages=[
        "Home" => "index.md",
    ],
    repo="https://github.com/spacedome/FEASTSolver.jl/blob/{commit}{path}#L{line}",
    sitename="FEASTSolver.jl",
    authors="spacedome",
    assets=String[],
)

deploydocs(;
    repo="github.com/spacedome/FEASTSolver.jl",
)
