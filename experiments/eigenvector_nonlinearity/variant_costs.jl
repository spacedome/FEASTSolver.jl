using LinearAlgebra
using Printf

include(joinpath(@__DIR__, "src", "ProjectorMomentFEAST.jl"))
using .ProjectorMomentFEAST

problem = contact_mean_field_1d(points=72, half_length=7.0, coupling=5.0, occupied=4)
initial = initial_orbitals(problem; seed=903)
reference = reference_scf(problem, density(problem, initial); mixing=0.2)
spectrum = eigen(hamiltonian(problem, reference.density)).values
gap = spectrum[problem.occupied + 1] - spectrum[problem.occupied]
chart = CircularChart(
    (spectrum[1] + spectrum[problem.occupied]) / 2,
    (spectrum[problem.occupied] - spectrum[1]) / 2 + 0.30gap,
    96,
)

common_response = (
    moment_depth=2,
    probe_width=2,
    iterations=30,
    warmup_iterations=2,
    warmup_method=:anderson,
    warmup_mixing=0.6,
    warmup_history_depth=10,
    krylov_method=:cg,
    krylov_tolerance=1e-9,
    density_tolerance=1e-10,
    residual_tolerance=1e-10,
)

results = [
    (
        name="mixed projector",
        result=solve_moment_scf(
            problem,
            chart,
            initial;
            config=MomentSCFConfig(
                moment_depth=2,
                probe_width=2,
                iterations=150,
                mixing=0.2,
                density_tolerance=1e-10,
                residual_tolerance=1e-10,
            ),
        ),
    ),
    (
        name="Anderson",
        result=solve_anderson_scf(
            problem,
            chart,
            initial;
            config=AndersonConfig(
                moment_depth=2,
                probe_width=2,
                iterations=60,
                history_depth=10,
                mixing=0.6,
                density_tolerance=1e-10,
                residual_tolerance=1e-10,
            ),
        ),
    ),
    (
        name="inexact response",
        result=solve_response_newton(
            problem,
            chart,
            initial;
            config=ResponseNewtonConfig(; common_response..., adaptive_krylov=true),
        ),
    ),
    (
        name="tight response",
        result=solve_response_newton(
            problem,
            chart,
            initial;
            config=ResponseNewtonConfig(; common_response..., adaptive_krylov=false),
        ),
    ),
]

println("outer-update cost envelope")
println("method                 outer  factorizations  RHS columns   residual")
for row in results
    result = row.result
    @printf(
        "%-22s %5d %15d %12d   %.2e\n",
        row.name,
        length(result.history),
        result.solve_count,
        result.rhs_count,
        result.residual,
    )
end

println("\nnormalized work = r × factorizations + RHS columns")
for ratio in (1, 10, 25, 100, 250)
    work = [ratio * row.result.solve_count + row.result.rhs_count for row in results]
    winner = results[argmin(work)].name
    @printf("r=%3d  winner=%-22s work=%d\n", ratio, winner, minimum(work))
end

anderson = results[2].result
inexact = results[3].result
tight = results[4].result
anderson_crossover = (inexact.rhs_count - anderson.rhs_count) /
    (anderson.solve_count - inexact.solve_count)
tight_crossover = (tight.rhs_count - inexact.rhs_count) /
    (inexact.solve_count - tight.solve_count)
@printf("\nAnderson/inexact crossover: %.2f RHS-column solves per factorization\n", anderson_crossover)
@printf("inexact/tight crossover:    %.2f RHS-column solves per factorization\n", tight_crossover)

