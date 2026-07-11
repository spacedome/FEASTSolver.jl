using LinearAlgebra
using Printf

include(joinpath(@__DIR__, "..", "src", "FusedNLFEAST.jl"))
using .FusedNLFEAST

function multiset_distance(actual, expected)
    length(actual) == length(expected) || return Inf
    order(values) = sort(ComplexF64.(values); by=value -> (real(value), imag(value)))
    maximum(abs.(order(actual) .- order(expected)))
end

function separated_pole_control()
    pole = 1.3 + 0.0im
    T = z -> fill(ComplexF64(sin(z) / (z - pole)), 1, 1)
    right_solve = (z, B) -> ((z - pole) / sin(z)) .* B
    left_solve = (z, B) -> conj((z - pole) / sin(z)) .* B
    chart = CircularChart(0.0 + 0.0im, 4.0, 128)
    result = fused_nlfeast(
        T,
        right_solve,
        left_solve,
        chart,
        ones(ComplexF64, 1, 1),
        ones(ComplexF64, 1, 1);
        config=FusedConfig(
            moment_count=3,
            iterations=3,
            ranktol=1e-11,
            residual_tol=1e-11,
            target_count=3,
            coupling=:independent,
        ),
    )
    index = argument_principle_count(
        z -> cos(z) / sin(z) - inv(z - pole),
        chart;
        integrality_tolerance=1e-8,
        refinement_tolerance=1e-8,
    )
    expected = ComplexF64[-pi, 0, pi]
    (
        extracted=length(result.extraction.values),
        error=multiset_distance(result.extraction.values, expected),
        residual=maximum(result.extraction.residuals),
        winding=index.count,
        winding_reliable=index.reliable,
        termination=result.termination_reason,
    )
end

function separated_pole_state_control()
    pole = 1.3 + 0.0im
    chart = CircularChart(0.0 + 0.0im, 4.0, 128)
    scalar_function = value -> sin(value) / (value - pole * one(value))
    right_solve = (z, block) -> ((z - pole) / sin(z)) .* block
    left_solve = (z, block) -> conj((z - pole) / sin(z)) .* block
    count = meromorphic_eigenvalue_count(
        z -> cos(z) / sin(z) - inv(z - pole),
        chart;
        pole_multiplicity=1,
        pole_multiplicity_certified=true,
        resolution_certified=true,
        integrality_tolerance=1e-8,
        refinement_tolerance=1e-8,
    )
    error_metric = function (state)
        components = state_components(state)
        lifted_residual_error(
            components.right_state,
            components.right,
            components.right_residual,
            components.left_state,
            components.left,
            components.left_residual;
            lift_depth=3,
        )
    end
    result = structured_fused_state_nlfeast(
        (ones(ComplexF64, 1, 1),),
        (scalar_function,),
        right_solve,
        left_solve,
        chart,
        ones(ComplexF64, 1, 1),
        ones(ComplexF64, 1, 1);
        config=StateIterationConfig(
            moment_count=3,
            iterations=6,
            ranktol=1e-11,
            residual_ranktol=1e-14,
            residual_tol=1e-11,
            target_count=count.count,
            target_count_certified=count.reliable,
            maxrank=count.count,
            rollback_ratio=1.05,
        ),
        state_error=error_metric,
    )
    values = eigvals(state_components(result.state).right_state)
    expected = ComplexF64[-pi, 0, pi]
    (
        count=count,
        error=multiset_distance(values, expected),
        residual=result.history[end].error,
        representation=result.history[end].representation,
        certified=result.certified,
        termination=result.termination_reason,
    )
end

function coincident_zero_pole_control()
    point = 0.2 + 0.0im
    chart = CircularChart(0.0 + 0.0im, 1.0, 32)
    inverse_operator = z -> Diagonal(ComplexF64[z - point, inv(z - point)])
    cache = ContourSampleCache(
        chart,
        (z, B) -> inverse_operator(z) * B,
        (z, B) -> adjoint(inverse_operator(z)) * B,
    )
    probe = Matrix{ComplexF64}(I, 2, 2)
    right = add_right_probe!(cache, :initial, probe)
    left = add_left_probe!(cache, :initial, probe)
    right_moments = probe_moments(cache, :initial, :right, 2)
    left_moments = probe_moments(cache, :initial, :left, 2)
    right_realization = FusedNLFEAST.hankel_realization(
        right_moments,
        left.probe,
        1;
        ranktol=1e-12,
        maxrank=2,
    )
    left_realization = FusedNLFEAST.hankel_realization(
        left_moments,
        right.probe,
        1;
        ranktol=1e-12,
        maxrank=2,
    )
    right_value = chart.center + chart.radius * only(eigvals(right_realization.state))
    left_value = chart.center + chart.radius * conj(only(eigvals(left_realization.state)))
    index = argument_principle_count(
        _ -> zero(ComplexF64),
        chart;
        integrality_tolerance=1e-12,
        refinement_tolerance=1e-12,
    )
    (
        right_rank=right_realization.rank,
        left_rank=left_realization.rank,
        right_value=right_value,
        left_value=left_value,
        winding=index.count,
        winding_reliable=index.reliable,
    )
end

separated = separated_pole_control()
separated_state = separated_pole_state_control()
coincident = coincident_zero_pole_control()

println("separated scalar pole:")
println(separated)
println("separated scalar pole, invariant-pair iteration:")
println(separated_state)
println("coincident matrix zero/pole partial multiplicities:")
println(coincident)
