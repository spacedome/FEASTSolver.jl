using LinearAlgebra
using Printf

include(joinpath(@__DIR__, "src", "ProjectorMomentFEAST.jl"))
using .ProjectorMomentFEAST

problem = contact_mean_field_1d(
    points=512,
    half_length=10.0,
    coupling=0.0,
    occupied=8,
    sparse=true,
)
initial = initial_orbitals(problem; seed=407)
H = hamiltonian(problem, density(problem, initial))
chart = CircularChart(4.0, 3.9, 64)
factors = ProjectorMomentFEAST.contour_factorizations(H, chart)

println("single-cache moment-width sweep")
println("depth × width   RHS columns   singular ratio   invariant residual")
for depth in (1, 2, 4, 8)
    width = problem.occupied ÷ depth
    tangent = ProjectorMomentFEAST.moment_tangent(problem.occupied, width)
    step = moment_projector_step(
        H,
        chart,
        initial * tangent,
        depth,
        problem.occupied;
        factorizations=factors,
    )
    residual = invariant_residual(problem, density(problem, step.orbitals), step.orbitals)
    @printf(
        "%5d × %-5d %12d   %14.3e   %.3e\n",
        depth,
        width,
        step.rhs_count,
        step.singular_values[problem.occupied] / step.singular_values[1],
        residual,
    )
end

