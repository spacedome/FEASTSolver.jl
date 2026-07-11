using LinearAlgebra
using Printf
using Random

include(joinpath(@__DIR__, "..", "src", "FusedNLFEAST.jl"))
using .FusedNLFEAST

function sorted_distance(actual, expected)
    length(actual) == length(expected) || return Inf
    order(values) = sort(ComplexF64.(values); by=value -> (real(value), imag(value)))
    maximum(abs.(order(actual) .- order(expected)))
end

function state_sweep(delta; nodes=32, updates=3)
    coefficients = Matrix{ComplexF64}[
        fill(ComplexF64(-delta^2), 1, 1),
        zeros(ComplexF64, 1, 1),
        ones(ComplexF64, 1, 1),
    ]
    case = polynomial_case(coefficients)
    chart = CircularChart(0.0 + 0.0im, 1.0, nodes)
    cache = ContourSampleCache(chart, case.right_solve, case.left_solve)
    right_initial = add_right_probe!(cache, :initial, ones(ComplexF64, 1, 1))
    left_initial = add_left_probe!(cache, :initial, ones(ComplexF64, 1, 1))
    right_moments = probe_moments(cache, :initial, :right, 4)
    left_moments = probe_moments(cache, :initial, :left, 4)
    state = common_polynomial_realization(
        coefficients,
        chart,
        right_moments,
        left_moments,
        left_initial.probe,
        right_initial.probe,
        2;
        ranktol=1e-14,
        maxrank=2,
        fixed_rank=2,
        overlap_ranktol=1e-14,
    )
    for iteration in 1:updates
        factors = state_residual_factors(
            state.right_residual,
            state.left_residual;
            ranktol=1e-14,
        )
        right_id = Symbol(:right_, iteration)
        left_id = Symbol(:left_, iteration)
        add_right_probe!(cache, right_id, factors.right_basis; role=:residual)
        add_left_probe!(cache, left_id, factors.left_basis; role=:residual)
        right_moments, left_moments = state_corrected_moments(
            cache,
            state.state,
            state.state,
            state.right,
            state.left,
            right_id,
            left_id,
            factors.right_coefficients,
            factors.left_coefficients,
            4,
        )
        state = common_polynomial_realization(
            coefficients,
            chart,
            right_moments,
            left_moments,
            left_initial.probe,
            right_initial.probe,
            2;
            ranktol=1e-14,
            maxrank=2,
            fixed_rank=2,
            overlap_ranktol=1e-14,
        )
    end
    target = ComplexF64[-delta, delta]
    singular_ratio = state.right_singular_values[2] / state.right_singular_values[1]
    pair_residual = max(norm(state.right_residual), norm(state.left_residual))
    (
        error=sorted_distance(eigvals(state.state), target),
        pair_residual=pair_residual,
        singular_ratio=singular_ratio,
        lifted_rank=rank(vcat(state.right, state.right * state.state); rtol=1e-12),
    )
end

function modal_sweep(delta; nodes=32)
    coefficients = Matrix{ComplexF64}[
        fill(ComplexF64(-delta^2), 1, 1),
        zeros(ComplexF64, 1, 1),
        ones(ComplexF64, 1, 1),
    ]
    case = polynomial_case(coefficients)
    chart = CircularChart(0.0 + 0.0im, 1.0, nodes)
    result = fused_nlfeast(
        case.T,
        case.right_solve,
        case.left_solve,
        chart,
        ones(ComplexF64, 1, 1),
        ones(ComplexF64, 1, 1);
        config=FusedConfig(
            moment_count=2,
            iterations=3,
            ranktol=1e-14,
            residual_tol=1e-12,
            target_count=2,
            coupling=:independent,
        ),
    )
    target = ComplexF64[-delta, delta]
    (
        error=sorted_distance(result.extraction.values, target),
        residual=isempty(result.extraction.residuals) ? Inf : maximum(result.extraction.residuals),
        count=length(result.extraction.values),
        reason=result.termination_reason,
    )
end

println("δ          state λ error  pair residual  σ₂/σ₁       lifted rank  modal λ error  modal residual  count  reason")
for delta in (1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 0.0)
    state = try
        state_sweep(delta)
    catch error
        error
    end
    modal = try
        modal_sweep(delta)
    catch error
        error
    end
    if state isa Exception || modal isa Exception
        @printf("%-10.1e state=%s modal=%s\n", delta, sprint(showerror, state), sprint(showerror, modal))
        continue
    end
    @printf(
        "%-10.1e %-14.3e %-14.3e %-12.3e %-12d %-14.3e %-15.3e %-6d %s\n",
        delta,
        state.error,
        state.pair_residual,
        state.singular_ratio,
        state.lifted_rank,
        modal.error,
        modal.residual,
        modal.count,
        string(modal.reason),
    )
end
