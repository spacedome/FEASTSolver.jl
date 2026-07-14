using LinearAlgebra
using Printf

include(joinpath(@__DIR__, "src", "ProjectorMomentFEAST.jl"))
using .ProjectorMomentFEAST

println("dual accumulated conditioning sweep")
println("ν     cond(S)   ok  refresh   q   factors   RHS     residual     biorth")
for nonnormality in (0.1, 0.3, 0.5, 0.65)
    problem = similarity_mean_field_1d(
        points=40,
        half_length=6.0,
        coupling=1.0,
        occupied=3,
        nonnormality=nonnormality,
    )
    right, left = initial_biorthogonal_orbitals(problem; seed=33)
    rho = oblique_density(problem, right, left)
    reference = reference_scf(problem.base, rho; mixing=0.5)
    initial_values = eigvals(hamiltonian(problem.base, rho))
    final_values = eigvals(hamiltonian(problem.base, reference.density))
    lower = min(initial_values[1], final_values[1])
    upper = max(initial_values[3], final_values[3])
    exterior = min(initial_values[4], final_values[4])
    chart = CircularChart(
        (lower + upper) / 2,
        (upper - lower) / 2 + 0.25 * (exterior - upper),
        64,
    )
    result = solve_dual_windowed_nlfeast(
        problem,
        chart,
        right,
        left;
        config=DualWindowConfig(
            moment_depth=3,
            probe_width=1,
            window_blocks=3,
            refresh_iterations=20,
            inner_iterations=3,
            inner_mixing=0.6,
            density_tolerance=1e-7,
            residual_tolerance=1e-7,
        ),
    )
    @printf(
        "%.2f  %9.2e  %-5s %7d %3d %9d %5d   %.2e   %.2e\n",
        nonnormality,
        maximum(problem.similarity) / minimum(problem.similarity),
        string(result.converged),
        result.refresh_count,
        size(result.right_basis, 2),
        result.solve_count,
        result.rhs_count,
        result.residual,
        norm(adjoint(result.left) * result.right - I),
    )
end
