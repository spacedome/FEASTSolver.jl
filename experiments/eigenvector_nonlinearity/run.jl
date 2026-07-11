using LinearAlgebra
using Printf

include(joinpath(@__DIR__, "src", "ProjectorMomentFEAST.jl"))
using .ProjectorMomentFEAST

problem = contact_mean_field_1d(points=96, half_length=8.0, coupling=5.0, occupied=4)
initial = initial_orbitals(problem; seed=2401)
initial_density = density(problem, initial)
reference = reference_scf(problem, initial_density; mixing=0.2)
reference.converged || error("reference SCF did not converge")

chart_policy = OccupiedChartPolicy(
    lower_bound=0.0,
    upper_bound=10.0,
    node_count=64,
)
reference_chart = occupied_chart(
    chart_policy,
    hamiltonian(problem, reference.density),
    problem.occupied,
).chart

config = ResponseNewtonConfig(
    moment_depth=2,
    probe_width=2,
    iterations=30,
    warmup_iterations=2,
    warmup_method=:anderson,
    warmup_mixing=0.6,
    warmup_history_depth=10,
    krylov_method=:cg,
    density_tolerance=1e-10,
    residual_tolerance=1e-10,
    krylov_tolerance=1e-9,
)
result = solve_response_newton(problem, chart_policy, initial; config=config)
anderson = solve_anderson_scf(
    problem,
    chart_policy,
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
)
control = solve_moment_scf(
    problem,
    chart_policy,
    initial;
    config=MomentSCFConfig(
        moment_depth=2,
        probe_width=2,
        iterations=150,
        mixing=0.2,
        density_tolerance=1e-10,
        residual_tolerance=1e-10,
    ),
)

exact_jacobian = projector_density_jacobian(problem, reference.density)
contour_jacobian = contour_density_jacobian(problem, reference.density, reference_chart)
plain_contraction = maximum(abs, eigvals(exact_jacobian))

println("projector moment FEAST for a 1D contact mean-field Hamiltonian")
@printf("grid points:                 %d\n", length(problem.grid))
@printf("occupied orbitals:           %d\n", problem.occupied)
@printf("moment depth × probe width:  %d × %d\n", config.moment_depth, config.probe_width)
@printf("contour nodes:               %d\n", length(reference_chart.nodes))
@printf("converged:                   %s in %d iterations\n", result.converged, length(result.history))
@printf("nonlinear invariant residual %.3e\n", result.residual)
@printf("density error vs reference:  %.3e\n", norm(result.density - reference.density) / norm(reference.density))
@printf("occupied subspace gap:       %.3e\n", subspace_gap(result.orbitals, reference.orbitals))
@printf("energy error:                %.3e\n", abs(result.energy - reference.energy))
@printf("contour Jacobian error:      %.3e\n", norm(contour_jacobian - exact_jacobian) / norm(exact_jacobian))
@printf("plain SCF Jacobian radius:   %.3e\n", plain_contraction)
@printf("response-Newton iterations:  %d\n", count(row -> row.mode === :response_newton, result.history))
@printf("response Krylov method:       %s\n", string(first(row.krylov_method for row in result.history if row.krylov_method !== :none)))
@printf("factorizations:               %d\n", result.solve_count)
@printf("inertia factorizations:       %d\n", result.chart_factorization_count)
@printf("node solve applications:     %d\n", result.solve_application_count)
@printf("RHS columns:                 %d\n", result.rhs_count)
@printf("damped projector iterations: %d\n", length(control.history))
@printf("damped factorizations:       %d\n", control.solve_count)
@printf("Anderson iterations:         %d\n", length(anderson.history))
@printf("Anderson factorizations:     %d\n", anderson.solve_count)
@printf("Anderson inertia factors:    %d\n", anderson.chart_factorization_count)
@printf("Anderson RHS columns:        %d\n", anderson.rhs_count)
