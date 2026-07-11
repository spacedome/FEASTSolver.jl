using FEASTSolver
using LinearAlgebra
using Printf
using Random
using SparseArrays

include(joinpath(@__DIR__, "..", "src", "FusedNLFEAST.jl"))
using .FusedNLFEAST

operator = FEASTSolver.feast_gallery("nlevp_native_gun")
center = 140000.0 + 0.0im
radius = 30000.0
nodes = parse(Int, get(ENV, "FUSED_GUN_NODES", "8"))
updates = parse(Int, get(ENV, "FUSED_GUN_UPDATES", "6"))
probe_width = parse(Int, get(ENV, "FUSED_GUN_WIDTH", "8"))
chart = CircularChart(center, radius, nodes)

factors = Dict{ComplexF64,Any}()
factor_seconds = @elapsed for z in chart.nodes
    matrix = FEASTSolver.operator_prototype(operator)
    FEASTSolver.materialize!(matrix, operator, z)
    factors[z] = lu(matrix)
end
right_solve = (z, block) -> factors[z] \ block
left_solve = (z, block) -> adjoint(factors[z]) \ block

branch_point = 108.8774^2
coefficients = (operator.K, -operator.M, operator.W1, operator.W2)
functions = (
    one,
    identity,
    value -> im * sqrt(value),
    value -> im * sqrt(value - branch_point * one(value)),
)
coefficient_norms = norm.(coefficients)

function pair_error(state)
    right_scale = sum(
        coefficient_norms[k] * norm(state.right * functions[k](state.state))
        for k in eachindex(functions)
    )
    left_scale = sum(
        coefficient_norms[k] * norm(state.left * adjoint(functions[k](state.state)))
        for k in eachindex(functions)
    )
    max(
        norm(state.right_residual) / right_scale,
        norm(state.left_residual) / left_scale,
    )
end

function modal_error(state)
    decomposition = eigen(state.state)
    right_vectors = state.right * decomposition.vectors
    left_vectors = state.left * adjoint(inv(decomposition.vectors))
    worst = 0.0
    for j in eachindex(decomposition.values)
        value = decomposition.values[j]
        right_vector = view(right_vectors, :, j)
        left_vector = view(left_vectors, :, j)
        right_residual = zeros(ComplexF64, size(operator, 1))
        left_residual = zeros(ComplexF64, size(operator, 1))
        workspace = similar(right_residual)
        for k in eachindex(functions)
            scalar = functions[k](value)
            mul!(workspace, coefficients[k], right_vector)
            axpy!(scalar, workspace, right_residual)
            mul!(workspace, adjoint(coefficients[k]), left_vector)
            axpy!(conj(scalar), workspace, left_residual)
        end
        right_scale = sum(
            coefficient_norms[k] * abs(functions[k](value)) * norm(right_vector)
            for k in eachindex(functions)
        )
        left_scale = sum(
            coefficient_norms[k] * abs(functions[k](value)) * norm(left_vector)
            for k in eachindex(functions)
        )
        worst = max(
            worst,
            norm(right_residual) / right_scale,
            norm(left_residual) / left_scale,
        )
    end
    worst, cond(decomposition.vectors)
end

rng = MersenneTwister(9901)
cache = nothing
right_initial = nothing
left_initial = nothing
state = nothing
initial_seconds = @elapsed begin
    cache = ContourSampleCache(chart, right_solve, left_solve)
    right_initial = add_right_probe!(
        cache,
        :initial,
        randn(rng, ComplexF64, size(operator, 1), probe_width),
    )
    left_initial = add_left_probe!(
        cache,
        :initial,
        randn(rng, ComplexF64, size(operator, 1), probe_width),
    )
    right_moments = probe_moments(cache, :initial, :right, 6)
    left_moments = probe_moments(cache, :initial, :left, 6)
    state = common_structured_realization(
        coefficients,
        functions,
        chart,
        right_moments,
        left_moments,
        left_initial.probe,
        right_initial.probe,
        3;
        ranktol=1e-10,
        maxrank=3 * probe_width,
        target_count=17,
    )
end

println("gun common-state iteration")
@printf("n=%d, nodes=%d, probe width=%d, factorization=%.2f s\n", size(operator, 1), nodes, probe_width, factor_seconds)
@printf("initial filter and extraction: %.2f s\n", initial_seconds)
println("iteration  residual       residual rank R/L  overlap ratio")
@printf(
    "%-10d %.3e      %-7s            %.3e\n",
    0,
    pair_error(state),
    "-/-",
    state.overlap_singular_values[end] / state.overlap_singular_values[1],
)

function run_updates(state, cache, updates)
    iteration_seconds = Float64[]
    for iteration in 1:updates
        factors_residual = nothing
        elapsed = @elapsed begin
            factors_residual = state_residual_factors(
                state.right_residual,
                state.left_residual;
                ranktol=1e-13,
            )
            right_id = Symbol(:right_, iteration)
            left_id = Symbol(:left_, iteration)
            add_right_probe!(cache, right_id, factors_residual.right_basis; role=:residual)
            add_left_probe!(cache, left_id, factors_residual.left_basis; role=:residual)
            right_moments, left_moments = state_corrected_moments(
                cache,
                state.state,
                state.state,
                state.right,
                state.left,
                right_id,
                left_id,
                factors_residual.right_coefficients,
                factors_residual.left_coefficients,
                6,
            )
            state = common_structured_realization(
                coefficients,
                functions,
                chart,
                right_moments,
                left_moments,
                left_initial.probe,
                right_initial.probe,
                3;
                ranktol=1e-10,
                maxrank=3 * probe_width,
                target_count=17,
            )
            drop_right_probe!(cache, right_id)
            drop_left_probe!(cache, left_id)
        end
        push!(iteration_seconds, elapsed)
        @printf(
            "%-10d %.3e      %-2d/%-4d            %.3e\n",
            iteration,
            pair_error(state),
            size(factors_residual.right_basis, 2),
            size(factors_residual.left_basis, 2),
            state.overlap_singular_values[end] / state.overlap_singular_values[1],
        )
    end
    state, iteration_seconds
end

state, iteration_seconds = run_updates(state, cache, updates)

values = eigvals(state.state)
worst_modal_error, modal_condition = modal_error(state)
println("interior states: ", length(values))
@printf("worst modal backward error: %.3e\n", worst_modal_error)
@printf("state eigenvector condition: %.3e\n", modal_condition)
@printf(
    "worst chart coordinate: %.6f\n",
    maximum(abs.(FusedNLFEAST.chart_coordinate.(Ref(chart), values))),
)
isempty(iteration_seconds) || @printf(
    "mean update time: %.2f s\n",
    sum(iteration_seconds) / length(iteration_seconds),
)
println("solve calls: ", solve_counts(cache))

if get(ENV, "FUSED_GUN_DRIVER", "0") == "1"
    driver_result = nothing
    driver_seconds = @elapsed begin
        driver_result = structured_fused_state_nlfeast(
            coefficients,
            functions,
            right_solve,
            left_solve,
            chart,
            right_initial.probe,
            left_initial.probe;
            config=StateIterationConfig(
                moment_count=3,
                iterations=updates,
                ranktol=1e-10,
                residual_ranktol=1e-13,
                residual_tol=1e-11,
                maxrank=3 * probe_width,
                target_count=17,
                common_ranktol=1e-10,
                rollback_ratio=1.05,
            ),
        )
    end
    println("\nintegrated state driver")
    @printf("driver time: %.2f s\n", driver_seconds)
    @printf("final lift-normalized error: %.3e\n", driver_result.history[end].error)
    println("representations: ", [record.representation for record in driver_result.history])
    println("selections: ", [record.selection for record in driver_result.history])
    println("solve calls: ", solve_counts(driver_result.cache))
end
