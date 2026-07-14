using LinearAlgebra
using Printf

include(joinpath(@__DIR__, "src", "ProjectorMomentFEAST.jl"))
using .ProjectorMomentFEAST

function run_case(; coupling, q, depth, width, tolerance, mixing)
    problem = contact_mean_field_1d(
        points=48,
        half_length=6.0,
        coupling=coupling,
        occupied=3,
    )
    initial = initial_orbitals(problem; seed=33)
    reference = reference_scf(
        problem,
        density(problem, initial);
        mixing=coupling <= 1 ? 0.5 : 0.2,
    )
    chart = OccupiedChartPolicy(lower_bound=0.0, upper_bound=15.0, node_count=64)
    config = TwoTimescaleConfig(
        moment_depth=depth,
        probe_width=width,
        subspace_dimension=q,
        refresh_iterations=35,
        maximum_inner_iterations=50,
        forcing_ratio=1.0,
        inner_mixing=mixing,
        density_tolerance=tolerance,
        residual_tolerance=tolerance,
    )
    result = solve_two_timescale_nlfeast(problem, chart, initial; config=config)
    (
        coupling=coupling,
        q=q,
        depth=depth,
        width=width,
        result=result,
        gap=subspace_gap(result.orbitals, reference.orbitals),
    )
end

cases = [
    run_case(coupling=1.0, q=5, depth=d, width=w, tolerance=1e-8, mixing=0.5)
    for (d, w) in ((1, 3), (2, 2), (3, 1))
]
append!(cases, [
    run_case(coupling=5.0, q=q, depth=3, width=1, tolerance=1e-8, mixing=0.3)
    for q in (3, 5, 8, 10)
])

println("Two-timescale occupied-enrichment variants")
println("  g    q   d×ℓ   ok   refresh   inner      closure      leakage          gap     RHS")
for case in cases
    result = case.result
    @printf(
        "%3.0f  %3d   %d×%d   %-5s %7d %7d   %10.3e   %10.3e   %10.3e %7d\n",
        case.coupling,
        case.q,
        case.depth,
        case.width,
        string(result.converged),
        result.refresh_count,
        result.inner_iterations,
        result.closure_defect,
        result.residual,
        case.gap,
        result.rhs_count,
    )
end
