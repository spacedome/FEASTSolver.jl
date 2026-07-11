using LinearAlgebra
using Printf
using Random

include(joinpath(@__DIR__, "..", "src", "FusedNLFEAST.jl"))
using .FusedNLFEAST

const EXPECTED = sort(
    ComplexF64[-pi, 0, pi, -pi / 2, pi / 2, 0];
    by=value -> (real(value), imag(value)),
)
const UPDATE_COUNT = parse(Int, get(ENV, "FUSED_STATE_UPDATES", "5"))

function multiset_distance(actual, expected)
    length(actual) == length(expected) || return Inf
    order(values) = sort(ComplexF64.(values); by=value -> (real(value), imag(value)))
    maximum(abs.(order(actual) .- order(expected)))
end

function modal_residual(T, state, vectors; side)
    decomposition = eigen(state)
    modes = side === :right ?
        vectors * decomposition.vectors :
        vectors * adjoint(inv(decomposition.vectors))
    maximum([
        norm(
            side === :right ?
                T(decomposition.values[j]) * modes[:, j] :
                adjoint(T(decomposition.values[j])) * modes[:, j],
        ) / norm(modes[:, j])
        for j in eachindex(decomposition.values)
    ])
end

function controllability_ratio(state, tangent, moment_count; side)
    dynamics = side === :right ? state : adjoint(state)
    blocks = Matrix{ComplexF64}[]
    block = Matrix{ComplexF64}(tangent)
    for _ in 1:moment_count
        push!(blocks, block)
        block = dynamics * block
    end
    singulars = svdvals(reduce(hcat, blocks))
    singulars[end] / singulars[1]
end

function run_gauge(nodes, seed, gauge; updates=5, tangent_width=2)
    case = nonnormal_analytic_case(coupling=8.0)
    chart = CircularChart(0.0 + 0.0im, 4.0, nodes)
    rng = MersenneTwister(seed)
    cache = ContourSampleCache(chart, case.right_solve, case.left_solve)
    right_initial = add_right_probe!(cache, :initial, randn(rng, ComplexF64, 3, 2))
    left_initial = add_left_probe!(cache, :initial, randn(rng, ComplexF64, 3, 2))
    right_moments = probe_moments(cache, :initial, :right, 10)
    left_moments = probe_moments(cache, :initial, :left, 10)

    realize = gauge === :common ? common_structured_realization : independent_structured_realization
    state = realize(
        case.structured_coefficients,
        case.structured_functions,
        chart,
        right_moments,
        left_moments,
        left_initial.probe,
        right_initial.probe,
        5;
        ranktol=1e-10,
        maxrank=6,
        fixed_rank=6,
        target_count=6,
    )
    minimum_tangent_ratio = Inf
    for iteration in 1:updates
        right_state = gauge === :common ? state.state : state.right_state
        left_state = gauge === :common ? state.state : state.left_state
        identity_state = Matrix{ComplexF64}(I, size(right_state, 1), size(right_state, 2))
        factors = state_residual_factors(
            state.right_residual,
            state.left_residual;
            ranktol=1e-13,
        )
        right_id = Symbol(:right_, iteration)
        left_id = Symbol(:left_, iteration)
        add_right_probe!(cache, right_id, factors.right_basis; role=:residual)
        add_left_probe!(cache, left_id, factors.left_basis; role=:residual)
        right_tangent = state_moment_tangent(
            (right_state .- chart.center .* identity_state) ./ chart.radius,
            5,
            tangent_width;
            side=:right,
            ranktol=1e-11,
        )
        left_tangent = state_moment_tangent(
            (left_state .- chart.center .* identity_state) ./ chart.radius,
            5,
            tangent_width;
            side=:left,
            ranktol=1e-11,
        )
        minimum_tangent_ratio = min(
            minimum_tangent_ratio,
            controllability_ratio(right_state, right_tangent, 5; side=:right),
            controllability_ratio(left_state, left_tangent, 5; side=:left),
        )
        right_moments, left_moments = state_corrected_moments(
            cache,
            right_state,
            left_state,
            state.right,
            state.left,
            right_id,
            left_id,
            factors.right_coefficients,
            factors.left_coefficients,
            10;
            right_tangent=right_tangent,
            left_tangent=left_tangent,
        )
        state = realize(
            case.structured_coefficients,
            case.structured_functions,
            chart,
            right_moments,
            left_moments,
            left_initial.probe,
            right_initial.probe,
            5;
            ranktol=1e-10,
            maxrank=6,
            fixed_rank=6,
            restrict_to_chart=false,
        )
    end
    if gauge === :common
        values = eigvals(state.state)
        residual = max(
            modal_residual(case.T, state.state, state.right; side=:right),
            modal_residual(case.T, state.state, state.left; side=:left),
        )
        detected = (state.rank, state.rank)
        overlap_ratio = state.overlap_singular_values[end] / state.overlap_singular_values[1]
    else
        right_values = eigvals(state.right_state)
        left_values = eigvals(state.left_state)
        values = right_values
        residual = max(
            modal_residual(case.T, state.right_state, state.right; side=:right),
            modal_residual(case.T, state.left_state, state.left; side=:left),
            multiset_distance(right_values, left_values),
        )
        detected = (state.right_detected_rank, state.left_detected_rank)
        overlap_ratio = NaN
    end
    (
        error=multiset_distance(values, EXPECTED),
        residual=residual,
        detected=detected,
        overlap_ratio=overlap_ratio,
        tangent_ratio=minimum_tangent_ratio,
    )
end

println("nodes width gauge       successes  worst λ error  worst residual  min overlap  min tangent")
for nodes in (24, 32, 48), tangent_width in (2, 6), gauge in (:common, :independent)
    rows = Any[]
    for seed in 1:10
        push!(rows, try
            run_gauge(nodes, seed, gauge; updates=UPDATE_COUNT, tangent_width=tangent_width)
        catch error
            error
        end)
    end
    valid = filter(row -> !(row isa Exception), rows)
    successes = count(row -> row.error <= 1e-8 && row.residual <= 1e-8, valid)
    worst_error = isempty(valid) ? Inf : maximum(row.error for row in valid)
    worst_residual = isempty(valid) ? Inf : maximum(row.residual for row in valid)
    overlap_ratios = [row.overlap_ratio for row in valid if isfinite(row.overlap_ratio)]
    minimum_overlap = isempty(overlap_ratios) ? NaN : minimum(overlap_ratios)
    minimum_tangent = isempty(valid) ? NaN : minimum(row.tangent_ratio for row in valid)
    @printf(
        "%-5d %-5d %-11s %-10d %-14.3e %-15.3e %-12.3e %.3e\n",
        nodes,
        tangent_width,
        string(gauge),
        successes,
        worst_error,
        worst_residual,
        minimum_overlap,
        minimum_tangent,
    )
    failures = [(seed=index, result=row) for (index, row) in enumerate(rows) if row isa Exception]
    isempty(failures) || println("  exceptions: ", failures)
end
