using LinearAlgebra
using Printf

include(joinpath(@__DIR__, "src", "CacheNativeCorrectedMoments.jl"))
using .CacheNativeCorrectedMoments

const PMF = CacheNativeCorrectedMoments.ProjectorMomentFEAST

println("simultaneous spectral/state correction reuse")
base = PMF.contact_mean_field_1d(
    points=32,
    half_length=6.0,
    coupling=1.0,
    occupied=3,
)
combined_problem = PMF.quadratic_density_nep(base; strength=0.02)
combined_initial = PMF.initial_orbitals(base; seed=33)
combined_chart = PMF.CircularChart(1.8, 1.5, 64)
println("policy       ok  refresh  factors  base RHS  correction RHS  repairs")
for (name, corrections, tolerance) in (
    ("single", 1, 0.0),
    ("forced 1e-6", 3, 1e-6),
    ("forced 1e-10", 3, 1e-10),
    ("fixed three", 3, 0.0),
)
    result = solve_cache_native_combined(
        combined_problem,
        combined_chart,
        combined_initial;
        config=CombinedCacheConfig(
            moment_depth=3,
            probe_width=1,
            spectral_corrections=corrections,
            spectral_tolerance=tolerance,
            window_blocks=4,
            refresh_iterations=15,
            reduced_inner_iterations=2,
            reduced_mixing=0.6,
            density_tolerance=1e-7,
            residual_tolerance=1e-7,
        ),
    )
    @printf(
        "%-12s %-5s %7d %8d %9d %15d %8d\n",
        name,
        string(result.converged),
        result.refresh_count,
        result.factorization_count,
        result.base_rhs_count,
        result.correction_rhs_count,
        sum(length(record.spectral_residuals) for record in result.history),
    )
end

println("\nnon-Hermitian constrained-memory response reuse")
dual_problem = PMF.similarity_mean_field_1d(
    points=40,
    half_length=6.0,
    coupling=1.0,
    occupied=3,
    nonnormality=0.65,
)
right, left = PMF.initial_biorthogonal_orbitals(dual_problem; seed=33)
rho = PMF.oblique_density(dual_problem, right, left)
reference = PMF.reference_scf(dual_problem.base, rho; mixing=0.5)
initial_values = sort(real.(eigvals(PMF.hamiltonian(dual_problem, rho))))
final_values = sort(real.(eigvals(PMF.hamiltonian(dual_problem, reference.density))))
lower = min(initial_values[1], final_values[1])
upper = max(initial_values[3], final_values[3])
exterior = min(initial_values[4], final_values[4])
dual_chart = PMF.CircularChart(
    (lower + upper) / 2,
    (upper - lower) / 2 + 0.25 * (exterior - upper),
    64,
)
println("policy         ok  refresh  factors  base RHS  response RHS  q")
for (name, blocks, inner, response, period) in (
    ("h2 inner1", 2, 1, false, 1),
    ("h2 cache/2", 2, 1, true, 2),
    ("h2 adaptive", 2, 1, :adaptive, 3),
    ("h3 inner2", 3, 2, false, 1),
    ("h3 cache/2", 3, 2, true, 2),
    ("h3 adaptive", 3, 2, :adaptive, 3),
)
    result = solve_cache_native_dual(
        dual_problem,
        dual_chart,
        right,
        left;
        config=DualCacheConfig(
            moment_depth=3,
            probe_width=1,
            window_blocks=blocks,
            refresh_iterations=30,
            pre_update=response === :adaptive ? :adaptive_response :
                response ? :response : :none,
            response_period=period,
            response_warmup_refreshes=2,
            response_warmup_mixing=0.6,
            reduced_inner_iterations=inner,
            reduced_mixing=0.6,
            density_tolerance=1e-7,
            residual_tolerance=1e-7,
        ),
    )
    @printf(
        "%-14s %-5s %7d %8d %9d %13d %2d\n",
        name,
        string(result.converged),
        result.refresh_count,
        result.factorization_count,
        result.right_rhs_count + result.left_rhs_count,
        result.response_rhs_count,
        size(result.right_basis, 2),
    )
end
