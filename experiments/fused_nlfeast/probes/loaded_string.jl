using FEASTSolver
using LinearAlgebra
using NonlinearEigenproblems
using Printf
using Random

include(joinpath(@__DIR__, "..", "src", "FusedNLFEAST.jl"))
using .FusedNLFEAST

function multiset_distance(actual, expected)
    length(actual) == length(expected) || return Inf
    order(values) = sort(ComplexF64.(values); by=value -> (real(value), imag(value)))
    maximum(abs.(order(actual) .- order(expected)))
end

operator = FEASTSolver.feast_gallery("nlevp_native_loaded_string", 20, 1, 1)
A = Matrix{ComplexF64}(operator.A0 + operator.A1)
B = Matrix{ComplexF64}(operator.B0 + operator.B1)
C = Matrix{ComplexF64}(operator.C)
sigma = operator.sigma
T = z -> Matrix{ComplexF64}(FEASTSolver.operator_matrix(operator, z))
right_solve = (z, rhs) -> T(z) \ rhs
left_solve = (z, rhs) -> adjoint(T(z)) \ rhs

companion_values, _ = polyeig(PEP([-sigma .* A, A .+ sigma .* B .+ C, -B]))
center = 200.0 + 0.0im
radius = 190.0
expected = ComplexF64[
    value for value in companion_values
    if abs(value - center) < radius && abs(value - sigma) > 1e-6
]

function divided_overlap(left_values, left, right_values, right)
    linear = -(adjoint(left) * B * right)
    rational = adjoint(left) * C * right
    for i in eachindex(left_values), j in eachindex(right_values)
        rational[i, j] *= -sigma /
            ((left_values[i] - sigma) * (right_values[j] - sigma))
    end
    linear + rational
end

function backward_error(value, vector)
    scale = norm(A) + abs(value) * norm(B) + abs(value / (value - sigma)) * norm(C)
    norm(T(value) * vector) / (scale * norm(vector))
end

println("nodes  successes  worst λ error  worst backward error  common accepted")
for nodes in (16, 24, 32, 48, 64)
    rows = Any[]
    for seed in 1:10
        rng = MersenneTwister(seed)
        chart = CircularChart(center, radius, nodes)
        result = try
            fused_nlfeast(
                T,
                right_solve,
                left_solve,
                chart,
                randn(rng, ComplexF64, 20, 2),
                randn(rng, ComplexF64, 20, 2);
                config=FusedConfig(
                    moment_count=3,
                    iterations=8,
                    ranktol=1e-10,
                    residual_ranktol=1e-13,
                    residual_tol=1e-10,
                    residual_scale=1.0,
                    target_count=length(expected),
                    coupling=:hybrid,
                    maxrank=6,
                ),
                divided_overlap=divided_overlap,
            )
        catch error
            error
        end
        if result isa Exception
            push!(rows, result)
            continue
        end
        errors = [
            backward_error(result.extraction.values[j], result.extraction.right[:, j])
            for j in eachindex(result.extraction.values)
        ]
        push!(rows, (
            value_error=multiset_distance(result.extraction.values, expected),
            backward_error=isempty(errors) ? Inf : maximum(errors),
            accepted=any(record.coupling_accepted for record in result.history),
        ))
    end
    valid = filter(row -> !(row isa Exception), rows)
    successes = count(
        row -> row.value_error <= 1e-7 && row.backward_error <= 1e-10,
        valid,
    )
    @printf(
        "%-6d %-10d %-14.3e %-21.3e %d/%d\n",
        nodes,
        successes,
        isempty(valid) ? Inf : maximum(row.value_error for row in valid),
        isempty(valid) ? Inf : maximum(row.backward_error for row in valid),
        count(row -> row.accepted, valid),
        length(valid),
    )
    exceptions = count(row -> row isa Exception, rows)
    if exceptions > 0
        messages = Dict{String,Int}()
        for row in rows
            row isa Exception || continue
            message = sprint(showerror, row)
            messages[message] = get(messages, message, 0) + 1
        end
        println("  exceptions: ", messages)
    end
end

function rational_function(value)
    value / (value - sigma * one(value))
end

structured_coefficients = (A, -B, C)
structured_functions = (one, identity, rational_function)

function state_run(nodes, seed; updates=12)
    trace = get(ENV, "FUSED_LOADED_TRACE", "false") == "true" && nodes == 24 && seed == 1
    rng = MersenneTwister(seed)
    chart = CircularChart(center, radius, nodes)
    cache = ContourSampleCache(chart, right_solve, left_solve)
    right_initial = add_right_probe!(cache, :initial, randn(rng, ComplexF64, 20, 2))
    left_initial = add_left_probe!(cache, :initial, randn(rng, ComplexF64, 20, 2))
    right_moments = probe_moments(cache, :initial, :right, 6)
    left_moments = probe_moments(cache, :initial, :left, 6)
    state = common_structured_realization(
        structured_coefficients,
        structured_functions,
        chart,
        right_moments,
        left_moments,
        left_initial.probe,
        right_initial.probe,
        3;
        ranktol=1e-10,
        maxrank=6,
        target_count=length(expected),
    )
    selections = Symbol[]
    function pair_score(candidate)
        decomposition = eigen(candidate.state)
        cond(decomposition.vectors) <= 1e8 || return Inf
        right_modes = candidate.right * decomposition.vectors
        left_modes = candidate.left * adjoint(inv(decomposition.vectors))
        maximum(vcat(
            [
                backward_error(decomposition.values[j], right_modes[:, j])
                for j in eachindex(decomposition.values)
            ],
            [
                begin
                    value = decomposition.values[j]
                    vector = left_modes[:, j]
                    scale = norm(A) + abs(value) * norm(B) +
                        abs(value / (value - sigma)) * norm(C)
                    norm(adjoint(T(value)) * vector) / (scale * norm(vector))
                end
                for j in eachindex(decomposition.values)
            ],
        ))
    end
    inside_count(candidate) = count(
        value -> FusedNLFEAST.in_chart(chart, value),
        eigvals(candidate.state),
    )
    for iteration in 1:updates
        factors = state_residual_factors(
            state.right_residual,
            state.left_residual;
            ranktol=1e-13,
        )
        right_id = Symbol(:state_right_, iteration)
        left_id = Symbol(:state_left_, iteration)
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
            6,
        )
        candidates = [(state=state, mode=:previous)]
        fixed = try
            common_structured_realization(
                structured_coefficients,
                structured_functions,
                chart,
                right_moments,
                left_moments,
                left_initial.probe,
                right_initial.probe,
                3;
                ranktol=1e-10,
                maxrank=6,
                fixed_rank=state.rank,
                restrict_to_chart=false,
            )
        catch
            nothing
        end
        fixed === nothing || push!(candidates, (state=fixed, mode=:fixed))
        broad = try
            common_structured_realization(
                structured_coefficients,
                structured_functions,
                chart,
                right_moments,
                left_moments,
                left_initial.probe,
                right_initial.probe,
                3;
                ranktol=1e-10,
                maxrank=6,
                target_count=state.rank,
            )
        catch
            nothing
        end
        broad === nothing || push!(candidates, (state=broad, mode=:broad))
        scores = [
            (
                state.rank - inside_count(candidate.state),
                pair_score(candidate.state),
            )
            for candidate in candidates
        ]
        if trace
            println(
                "iteration ",
                iteration,
                ": ",
                [
                    (
                        mode=candidate.mode,
                        inside=inside_count(candidate.state),
                        score=pair_score(candidate.state),
                        values=sort(eigvals(candidate.state.state); by=real),
                    )
                    for candidate in candidates
                ],
            )
        end
        best = candidates[argmin(scores)]
        push!(selections, best.mode)
        best.mode === :previous && break
        state = best.state
    end
    decomposition = eigen(state.state)
    right_modes = state.right * decomposition.vectors
    errors = [backward_error(decomposition.values[j], right_modes[:, j]) for j in eachindex(decomposition.values)]
    (
        value_error=multiset_distance(decomposition.values, expected),
        backward_error=maximum(errors),
        overlap_ratio=state.overlap_singular_values[end] / state.overlap_singular_values[1],
        selections=selections,
    )
end

println("state realization:")
println("nodes  successes  worst λ error  worst backward error  min overlap ratio")
for nodes in (16, 24, 32)
    rows = Any[
        try
            state_run(nodes, seed)
        catch error
            error
        end
        for seed in 1:10
    ]
    valid = filter(row -> !(row isa Exception), rows)
    successes = count(
        row -> row.value_error <= 1e-7 && row.backward_error <= 1e-10,
        valid,
    )
    @printf(
        "%-6d %-10d %-14.3e %-21.3e %.3e\n",
        nodes,
        successes,
        isempty(valid) ? Inf : maximum(row.value_error for row in valid),
        isempty(valid) ? Inf : maximum(row.backward_error for row in valid),
        isempty(valid) ? NaN : minimum(row.overlap_ratio for row in valid),
    )
    exceptions = [sprint(showerror, row) for row in rows if row isa Exception]
    isempty(exceptions) || println("  exceptions: ", exceptions)
end
