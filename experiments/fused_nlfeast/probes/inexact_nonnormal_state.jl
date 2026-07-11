using LinearAlgebra
using Printf
using Random

include(joinpath(@__DIR__, "..", "src", "FusedNLFEAST.jl"))
using .FusedNLFEAST

const EXPECTED = ComplexF64[-pi, 0, pi, -pi / 2, pi / 2, 0]

function bottleneck_distance(actual, expected)
    length(actual) == length(expected) || return Inf
    _, errors = FusedNLFEAST.bottleneck_match(actual, expected)
    maximum(errors)
end

function residual_controlled_solve(matrix, block, relative_residual, observed)
    if iszero(relative_residual)
        solution = matrix \ block
    else
        decomposition = svd(matrix)
        worst_direction = decomposition.U[:, end]
        row_direction = fill(inv(sqrt(size(block, 2))), 1, size(block, 2))
        residual = relative_residual * norm(block) .* (worst_direction * row_direction)
        solution = matrix \ (block - residual)
    end
    ratio = norm(block - matrix * solution) / norm(block)
    observed[] = max(observed[], ratio)
    solution
end

function run_case(coupling, relative_residual)
    case = nonnormal_analytic_case(coupling=coupling)
    chart = CircularChart(0.0, 4.0, 48)
    observed = Ref(0.0)
    right_solve = (z, block) -> residual_controlled_solve(
        case.T(z),
        block,
        relative_residual,
        observed,
    )
    left_solve = (z, block) -> residual_controlled_solve(
        adjoint(case.T(z)),
        block,
        relative_residual,
        observed,
    )
    rng = MersenneTwister(7103)
    result = structured_fused_state_nlfeast(
        case.structured_coefficients,
        case.structured_functions,
        right_solve,
        left_solve,
        chart,
        randn(rng, ComplexF64, case.dimension, 2),
        randn(rng, ComplexF64, case.dimension, 2);
        config=StateIterationConfig(
            moment_count=5,
            iterations=12,
            ranktol=1e-10,
            residual_ranktol=1e-13,
            residual_tol=1e-10,
            maxrank=6,
            target_count=6,
            common_ranktol=1e-10,
            rollback_ratio=1.05,
            max_stagnation=3,
        ),
    )
    modal = state_modal_output(result.state)
    components = state_components(result.state)
    values = eigvals(components.right_state)
    modal_conditions = modal.available ? structured_eigenvalue_condition_numbers(
        case.structured_coefficients,
        (cos, value -> -sin(value), exp),
        modal.values,
        modal.right,
        modal.left,
    ) : Float64[Inf]
    (
        condition=maximum(cond(case.T(z)) for z in chart.nodes),
        observed=observed[],
        value_error=bottleneck_distance(values, EXPECTED),
        final_error=result.history[end].error,
        modal_condition=maximum(modal_conditions),
        updates=length(result.history) - 1,
        converged=result.converged,
        termination=result.termination_reason,
    )
end

println("nonnormal residual-controlled contour solves")
println("coupling  κΓ           η requested  η observed   converged  updates  λ error       pair error     max κλ        termination")
for coupling in (8.0, 1e2, 1e4)
    for relative_residual in (0.0, 1e-8, 1e-6, 1e-4, 1e-2)
        outcome = try
            run_case(coupling, relative_residual)
        catch error
            error
        end
        if outcome isa Exception
            @printf(
                "%-9.1e %-12s %-12.1e %-12s %-10s %-8s %-13s %-14s %-13s %s\n",
                coupling,
                "-",
                relative_residual,
                "-",
                "false",
                "-",
                "-",
                "-",
                "-",
                sprint(showerror, outcome),
            )
            continue
        end
        @printf(
            "%-9.1e %-12.3e %-12.1e %-12.3e %-10s %-8d %-13.3e %-14.3e %-13.3e %s\n",
            coupling,
            outcome.condition,
            relative_residual,
            outcome.observed,
            string(outcome.converged),
            outcome.updates,
            outcome.value_error,
            outcome.final_error,
            outcome.modal_condition,
            string(outcome.termination),
        )
    end
end
