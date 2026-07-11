using FEASTSolver
using LinearAlgebra
using NonlinearEigenproblems
using Printf
using Random

include(joinpath(@__DIR__, "..", "src", "FusedNLFEAST.jl"))
using .FusedNLFEAST

operator = FEASTSolver.feast_gallery("nlevp_native_loaded_string", 20, 1, 1)
A = Matrix{ComplexF64}(operator.A0 + operator.A1)
B = Matrix{ComplexF64}(operator.B0 + operator.B1)
C = Matrix{ComplexF64}(operator.C)
σ = operator.sigma
T = z -> Matrix{ComplexF64}(FEASTSolver.operator_matrix(operator, z))
right_solve = (z, rhs) -> T(z) \ rhs
left_solve = (z, rhs) -> adjoint(T(z)) \ rhs

companion_values, _ = polyeig(PEP([-σ .* A, A .+ σ .* B .+ C, -B]))
center = 200.0 + 0.0im
radius = 190.0
expected = ComplexF64[
    value for value in companion_values
    if abs(value - center) < radius && abs(value - σ) > 1e-6
]
coefficients = (A, -B, C)
functions = (one, identity, value -> value / (value - σ * one(value)))

function bottleneck_distance(actual, reference)
    length(actual) == length(reference) || return Inf
    _, errors = FusedNLFEAST.bottleneck_match(actual, reference)
    isempty(errors) ? 0.0 : maximum(errors)
end

function backward_error(value, vector; left=false)
    scale = norm(A) + abs(value) * norm(B) + abs(value / (value - σ)) * norm(C)
    action = left ? adjoint(T(value)) * vector : T(value) * vector
    norm(action) / (scale * norm(vector))
end

state_parts(state::CommonStateRealization) = (
    state.state,
    state.state,
    state.right,
    state.left,
    state.right_residual,
    state.left_residual,
)

state_parts(state::IndependentStateRealization) = (
    state.right_state,
    state.left_state,
    state.right,
    state.left,
    state.right_residual,
    state.left_residual,
)

state_parts(state::TwoSidedHankelRealization) = (
    state.state,
    state.state,
    state.right,
    state.left,
    state.right_residual,
    state.left_residual,
)

function realize(kind, chart, right_moments, left_moments, right_observer, left_observer; kwargs...)
    if kind === :single
        two_sided_structured_realization(
            coefficients,
            functions,
            chart,
            right_moments,
            left_moments,
            right_observer,
            3;
            ranktol=1e-10,
            maxrank=6,
            kwargs...,
        )
    elseif kind === :common
        common_structured_realization(
            coefficients,
            functions,
            chart,
            right_moments,
            left_moments,
            right_observer,
            left_observer,
            3;
            ranktol=1e-10,
            maxrank=6,
            kwargs...,
        )
    else
        independent_structured_realization(
            coefficients,
            functions,
            chart,
            right_moments,
            left_moments,
            right_observer,
            left_observer,
            3;
            ranktol=1e-10,
            maxrank=6,
            kwargs...,
        )
    end
end

function state_metrics(state)
    right_state, left_state, right, left, _, _ = state_parts(state)
    right_decomposition = eigen(right_state)
    left_decomposition = eigen(left_state)
    right_vectors = right * right_decomposition.vectors
    left_vectors = left * adjoint(inv(left_decomposition.vectors))
    right_errors = [
        backward_error(right_decomposition.values[j], right_vectors[:, j])
        for j in eachindex(right_decomposition.values)
    ]
    left_errors = [
        backward_error(left_decomposition.values[j], left_vectors[:, j]; left=true)
        for j in eachindex(left_decomposition.values)
    ]
    (
        values=right_decomposition.values,
        left_values=left_decomposition.values,
        right_backward_error=maximum(right_errors),
        left_backward_error=maximum(left_errors),
        backward_error=maximum(vcat(right_errors, left_errors)),
        right_condition=cond(right_decomposition.vectors),
        left_condition=cond(left_decomposition.vectors),
    )
end

function policy_candidate(
    kind,
    selection,
    chart,
    right_moments,
    left_moments,
    left_observer,
    right_observer,
    right_state,
    left_state,
)
    if selection === :fixed
        realize(
            kind,
            chart,
            right_moments,
            left_moments,
            left_observer,
            right_observer;
            fixed_rank=length(expected),
            restrict_to_chart=false,
        )
    elseif selection === :chart
        realize(
            kind,
            chart,
            right_moments,
            left_moments,
            left_observer,
            right_observer;
            target_count=length(expected),
        )
    elseif selection === :track
        realize(
            kind,
            chart,
            right_moments,
            left_moments,
            left_observer,
            right_observer;
            restrict_to_chart=false,
            right_reference_values=eigvals(right_state),
            left_reference_values=eigvals(left_state),
        )
    else
        anchor = selection === :right_anchor ? :right : :left
        realize(
            kind,
            chart,
            right_moments,
            left_moments,
            left_observer,
            right_observer;
            target_count=length(expected),
            selection_anchor=anchor,
        )
    end
end

function run_policy(
    kind,
    selection,
    nodes,
    seed;
    updates=12,
    augment=false,
    max_initial_width=4,
    tangent_width=nothing,
    compress_after=0,
    compression_gate=true,
)
    rng = MersenneTwister(seed)
    chart = CircularChart(center, radius, nodes)
    cache = ContourSampleCache(chart, right_solve, left_solve)
    add_right_probe!(cache, :initial, randn(rng, ComplexF64, 20, 2))
    add_left_probe!(cache, :initial, randn(rng, ComplexF64, 20, 2))
    right_ids = Symbol[:initial]
    left_ids = Symbol[:initial]
    right_moments = nothing
    left_moments = nothing
    state = nothing
    while state === nothing
        right_initial_probe = combined_probe(cache, right_ids, :right)
        left_initial_probe = combined_probe(cache, left_ids, :left)
        right_moments = combined_probe_moments(cache, right_ids, :right, 6)
        left_moments = combined_probe_moments(cache, left_ids, :left, 6)
        state = try
            realize(
                kind,
                chart,
                right_moments,
                left_moments,
                left_initial_probe,
                right_initial_probe;
                target_count=length(expected),
            )
        catch error
            current_width = size(right_initial_probe, 2)
            if !augment || current_width >= max_initial_width
                rethrow(error)
            end
            next_width = current_width + 1
            right_id = Symbol(:initial_right_, next_width)
            left_id = Symbol(:initial_left_, next_width)
            add_right_probe!(cache, right_id, randn(rng, ComplexF64, 20, 1))
            add_left_probe!(cache, left_id, randn(rng, ComplexF64, 20, 1))
            push!(right_ids, right_id)
            push!(left_ids, left_id)
            nothing
        end
    end
    right_initial_probe = combined_probe(cache, right_ids, :right)
    left_initial_probe = combined_probe(cache, left_ids, :left)
    compression_rejections = 0

    for iteration in 1:updates
        right_state, left_state, right, left, right_residual, left_residual = state_parts(state)
        factors = state_residual_factors(right_residual, left_residual; ranktol=1e-13)
        right_id = Symbol(:right_, iteration)
        left_id = Symbol(:left_, iteration)
        add_right_probe!(cache, right_id, factors.right_basis; role=:residual)
        add_left_probe!(cache, left_id, factors.left_basis; role=:residual)
        active_tangent_width = iteration > compress_after ? tangent_width : nothing
        state_count = size(right_state, 1)
        identity_state = Matrix{ComplexF64}(I, state_count, state_count)
        right_chart_state = (right_state .- chart.center .* identity_state) ./ chart.radius
        left_chart_state = (left_state .- chart.center .* identity_state) ./ chart.radius
        right_tangent = if kind === :single
            adjoint(left) * right_initial_probe
        elseif active_tangent_width === nothing
            nothing
        else
            state_moment_tangent(
                right_chart_state,
                3,
                active_tangent_width;
                side=:right,
                ranktol=1e-10,
            )
        end
        left_tangent = if kind === :single
            left_filter = zeros(ComplexF64, state_count, state_count)
            for (z, weight) in zip(chart.nodes, chart.weights)
                left_filter .+= conj(weight) .* inv(conj(z) .* identity_state .- adjoint(left_state))
            end
            left_filter \ (adjoint(right) * left_initial_probe)
        elseif active_tangent_width === nothing
            nothing
        else
            state_moment_tangent(
                left_chart_state,
                3,
                active_tangent_width;
                side=:left,
                ranktol=1e-10,
            )
        end
        right_moments, left_moments = state_corrected_moments(
            cache,
            right_state,
            left_state,
            right,
            left,
            right_id,
            left_id,
            factors.right_coefficients,
            factors.left_coefficients,
            6,
            right_tangent=right_tangent,
            left_tangent=left_tangent,
        )
        candidate = policy_candidate(
            kind,
            selection,
            chart,
            right_moments,
            left_moments,
            left_initial_probe,
            right_initial_probe,
            right_state,
            left_state,
        )
        if compression_gate && active_tangent_width !== nothing &&
           state_metrics(candidate).backward_error > state_metrics(state).backward_error
            compression_rejections += 1
            right_moments, left_moments = state_corrected_moments(
                cache,
                right_state,
                left_state,
                right,
                left,
                right_id,
                left_id,
                factors.right_coefficients,
                factors.left_coefficients,
                6,
            )
            candidate = policy_candidate(
                kind,
                selection,
                chart,
                right_moments,
                left_moments,
                left_initial_probe,
                right_initial_probe,
                right_state,
                left_state,
            )
        end
        state = candidate
        drop_right_probe!(cache, right_id)
        drop_left_probe!(cache, left_id)
    end

    metrics = state_metrics(state)
    (
        λ_error=bottleneck_distance(metrics.values, expected),
        pair_error=bottleneck_distance(metrics.values, metrics.left_values),
        right_backward_error=metrics.right_backward_error,
        left_backward_error=metrics.left_backward_error,
        backward_error=metrics.backward_error,
        right_condition=metrics.right_condition,
        left_condition=metrics.left_condition,
        initial_width=size(right_initial_probe, 2),
        compression_rejections=compression_rejections,
    )
end

function run_driver(nodes, seed; updates=8, residual_tol=1e-10, coupling=:common)
    rng = MersenneTwister(seed)
    chart = CircularChart(center, radius, nodes)
    result = structured_fused_state_nlfeast(
        coefficients,
        functions,
        right_solve,
        left_solve,
        chart,
        randn(rng, ComplexF64, 20, 2),
        randn(rng, ComplexF64, 20, 2);
        config=StateIterationConfig(
            moment_count=3,
            iterations=updates,
            ranktol=1e-10,
            residual_ranktol=1e-13,
            residual_tol=residual_tol,
            target_count=length(expected),
            maxrank=6,
            max_initial_probe_width=3,
            rollback_ratio=1.05,
        ),
        augment_probe=(side, width, attempt) -> randn(rng, ComplexF64, 20, 1),
        coupling=coupling,
    )
    metrics = state_metrics(result.state)
    (
        result=result,
        λ_error=bottleneck_distance(metrics.values, expected),
        backward_error=metrics.backward_error,
        initial_width=result.history[1].right_probe_width,
        common_steps=count(record -> record.representation === :common, result.history),
        rollbacks=count(record -> record.selection === :rollback, result.history),
    )
end

if abspath(PROGRAM_FILE) == @__FILE__
    println("state policy matrix, loaded string, 12 updates")
    println("nodes  realization  selection  successes  worst λ error  worst pair error  worst backward error")
    for nodes in (16, 24, 32), kind in (:independent, :common, :single)
        policies = kind === :single ? (:fixed, :chart) :
            (:fixed, :chart, :track, :right_anchor, :left_anchor)
        for selection in policies
        rows = Any[
            try
                run_policy(kind, selection, nodes, seed)
            catch error
                error
            end
            for seed in 1:10
        ]
        valid = filter(row -> !(row isa Exception), rows)
        successes = count(
            row -> row.λ_error <= 1e-7 && row.backward_error <= 1e-10,
            valid,
        )
        @printf(
            "%-6d %-12s %-10s %-10d %-14.3e %-16.3e %.3e\n",
            nodes,
            String(kind),
            String(selection),
            successes,
            isempty(valid) ? Inf : maximum(row.λ_error for row in valid),
            isempty(valid) ? Inf : maximum(row.pair_error for row in valid),
            isempty(valid) ? Inf : maximum(row.backward_error for row in valid),
        )
        failures = [sprint(showerror, row) for row in rows if row isa Exception]
        isempty(failures) || println("  failures: ", unique(failures))
        end
    end
end
