using FEASTSolver
using LinearAlgebra
using Printf
using Random
using SparseArrays

include(joinpath(@__DIR__, "..", "src", "FusedNLFEAST.jl"))
using .FusedNLFEAST

operator = FEASTSolver.feast_gallery("schrodinger_movebc", 128)
prototype = FEASTSolver.operator_prototype(operator)

function Tmatrix(z)
    matrix = similar(prototype)
    FEASTSolver.materialize!(matrix, operator, z)
    matrix
end

center = -35.0 + 0.0im
radius = 4.2
count_estimate = determinant_winding_count(
    Tmatrix,
    CircularChart(center, radius, 64);
    phase_resolution_certified=true,
)
target_count = count_estimate.count

span = operator.L1 - operator.L0
function shifted_root(value)
    identity_value = one(value)
    sqrt(value + operator.V0 * identity_value)
end
g(value) = cosh(span * shifted_root(value))
h(value) = sinh(span * shifted_root(value)) / shifted_root(value)

coefficients = (operator.A0, -operator.I0, operator.G, operator.F)
functions = (one, identity, g, h)
right_solve = (z, rhs) -> Tmatrix(z) \ rhs
left_solve = (z, rhs) -> adjoint(Tmatrix(z)) \ rhs

coefficient_norms = norm.(coefficients)
function backward_scale(value)
    abs(one(value)) * coefficient_norms[1] +
        abs(value) * coefficient_norms[2] +
        abs(g(value)) * coefficient_norms[3] +
        abs(h(value)) * coefficient_norms[4]
end

function state_run(seed; nodes=24, updates=12)
    rng = MersenneTwister(seed)
    chart = CircularChart(center, radius, nodes)
    cache = ContourSampleCache(chart, right_solve, left_solve)
    right_initial = add_right_probe!(cache, :initial, randn(rng, ComplexF64, 128, 2))
    left_initial = add_left_probe!(cache, :initial, randn(rng, ComplexF64, 128, 2))
    right_moments = probe_moments(cache, :initial, :right, 4)
    left_moments = probe_moments(cache, :initial, :left, 4)
    state = common_structured_realization(
        coefficients,
        functions,
        chart,
        right_moments,
        left_moments,
        left_initial.probe,
        right_initial.probe,
        2;
        ranktol=1e-10,
        maxrank=4,
        target_count=target_count,
    )
    for iteration in 1:updates
        factors = state_residual_factors(
            state.right_residual,
            state.left_residual;
            ranktol=1e-12,
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
        drop_right_probe!(cache, right_id)
        drop_left_probe!(cache, left_id)
        state = common_structured_realization(
            coefficients,
            functions,
            chart,
            right_moments,
            left_moments,
            left_initial.probe,
            right_initial.probe,
            2;
            ranktol=1e-10,
            maxrank=4,
            fixed_rank=state.rank,
            restrict_to_chart=false,
        )
    end
    decomposition = eigen(state.state)
    right_modes = state.right * decomposition.vectors
    residuals = [
        norm(Tmatrix(decomposition.values[j]) * right_modes[:, j]) /
        (backward_scale(decomposition.values[j]) * norm(right_modes[:, j]))
        for j in eachindex(decomposition.values)
    ]
    (
        values=decomposition.values,
        residual=maximum(residuals),
        cache_bytes=Base.summarysize(cache),
        solves=solve_counts(cache),
    )
end

println(
    "determinant winding: coarse=",
    count_estimate.coarse_value,
    ", refined=",
    count_estimate.value,
    ", target=",
    target_count,
)
println("sparse prototype: ", prototype isa AbstractSparseMatrix)
for seed in 1:3
    result = state_run(seed)
    @printf(
        "seed=%d count=%d residual=%.3e cache=%.2f MiB solves=(%d,%d) values=%s\n",
        seed,
        length(result.values),
        result.residual,
        result.cache_bytes / 2.0^20,
        result.solves.right,
        result.solves.left,
        repr(result.values),
    )
end
