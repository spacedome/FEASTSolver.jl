@testset "state driver preserves a shared-eigenvector invariant pair" begin
    case = shared_eigenvector_case()
    contour = case.contours[1]
    chart = CircularChart(contour.center, contour.radius, 16)
    rng = MersenneTwister(812)
    result = structured_fused_state_nlfeast(
        case.coefficients,
        (one, identity, value -> value^2),
        case.right_solve,
        case.left_solve,
        chart,
        randn(rng, ComplexF64, case.dimension, 1),
        randn(rng, ComplexF64, case.dimension, 1);
        config=StateIterationConfig(
            moment_count=2,
            iterations=8,
            residual_tol=1e-9,
            target_count=2,
            target_count_certified=true,
            maxrank=2,
        ),
    )
    components = state_components(result.state)
    @test multiset_distance(eigvals(components.right_state), contour.expected) <= 1e-8
    @test result.residual_converged
    @test result.count_matched
    @test result.certified
    @test result.termination_reason === :converged_certified
    @test all(isempty(block.responses) for block in values(result.cache.right_blocks))
    @test all(isempty(block.responses) for block in values(result.cache.left_blocks))
    @test rank(
        vcat(
            components.left,
            components.left * adjoint(components.left_state),
        );
        rtol=1e-8,
    ) == 2
end

@testset "initial count deficiency is a typed candidate failure" begin
    case = scalar_sine_case()
    error = try
        structured_fused_state_nlfeast(
            (ones(ComplexF64, 1, 1),),
            (sin,),
            case.right_solve,
            case.left_solve,
            CircularChart(pi / 2, 2.0, 16),
            ones(ComplexF64, 1, 1),
            ones(ComplexF64, 1, 1);
            config=StateIterationConfig(
                moment_count=1,
                iterations=0,
                target_count=2,
                maxrank=2,
            ),
        )
        nothing
    catch caught
        caught
    end
    @test error isa StateCandidateFailure
    @test error.reason === :insufficient_states
end

@testset "state driver can retain independent left and right realizations" begin
    root = 0.2 + 0.0im
    solve = (z, block) -> block ./ (z - root)
    result = structured_fused_state_nlfeast(
        (fill(-root, 1, 1), ones(ComplexF64, 1, 1)),
        (one, identity),
        solve,
        (z, block) -> block ./ conj(z - root),
        CircularChart(0.0, 1.0, 16),
        ones(ComplexF64, 1, 1),
        ones(ComplexF64, 1, 1);
        config=StateIterationConfig(moment_count=1, iterations=0, target_count=1),
        coupling=:independent,
    )

    @test result.history[1].representation === :independent
    @test_throws ArgumentError structured_fused_state_nlfeast(
        (fill(-root, 1, 1), ones(ComplexF64, 1, 1)),
        (one, identity),
        solve,
        solve,
        CircularChart(0.0, 1.0, 16),
        ones(ComplexF64, 1, 1),
        ones(ComplexF64, 1, 1);
        config=StateIterationConfig(moment_count=1, iterations=0, target_count=1),
        coupling=:unsupported,
    )
end

@testset "state driver enforces reported contour solve accuracy" begin
    root = 0.2 + 0.0im
    right_solve = (z, block) -> ContourSolveResult(
        block ./ (z - root);
        relative_residual=1e-14,
        iterations=2,
    )
    left_solve = (z, block) -> ContourSolveResult(
        block ./ conj(z - root);
        relative_residual=1e-14,
        iterations=2,
    )
    result = structured_fused_state_nlfeast(
        (fill(-root, 1, 1), ones(ComplexF64, 1, 1)),
        (one, identity),
        right_solve,
        left_solve,
        CircularChart(0.0, 1.0, 16),
        ones(ComplexF64, 1, 1),
        ones(ComplexF64, 1, 1);
        config=StateIterationConfig(
            moment_count=1,
            iterations=2,
            residual_tol=1e-11,
            target_count=1,
            target_count_certified=true,
            maxrank=1,
            require_solve_diagnostics=true,
            maximum_solve_residual=1e-12,
        ),
    )
    summary = solve_diagnostic_summary(result.cache)
    @test result.certified
    @test summary.unreported == 0
    @test summary.maximum_relative_residual == 1e-14
    @test summary.maximum_iterations == 2
end

@testset "invalid analytic chart is rejected before contour solves" begin
    solve_calls = Ref(0)
    solve = function (z, block)
        solve_calls[] += 1
        block
    end
    error = try
        structured_fused_state_nlfeast(
            (ones(ComplexF64, 1, 1),),
            (sqrt,),
            solve,
            solve,
            CircularChart(0.0, 1.0, 16),
            ones(ComplexF64, 1, 1),
            ones(ComplexF64, 1, 1);
            config=StateIterationConfig(moment_count=1, target_count=1),
            analytic_domain=_ -> (
                valid=false,
                reason="principal square-root branch cut crosses the chart",
            ),
        )
        nothing
    catch caught
        caught
    end
    @test error isa InvalidAnalyticDomainError
    @test occursin("branch cut", sprint(showerror, error))
    @test solve_calls[] == 0
end

@testset "an exterior continuation cannot certify a chart leaf" begin
    low = -37.526
    high = -24.588
    chart = RectangularChart((low + high) / 2, (high - low) / 2, 10.0, 128)
    expected = ComplexF64[-11pi, -10pi, -9pi, -8pi]
    case = scalar_sine_case()
    error_metric = function (state)
        components = state_components(state)
        lifted_residual_error(
            components.right_state,
            components.right,
            components.right_residual,
            components.left_state,
            components.left,
            components.left_residual;
            lift_depth=4,
        )
    end
    result = structured_fused_state_nlfeast(
        (ones(ComplexF64, 1, 1),),
        (sin,),
        case.right_solve,
        case.left_solve,
        chart,
        ones(ComplexF64, 1, 1),
        ones(ComplexF64, 1, 1);
        config=StateIterationConfig(
            moment_count=4,
            iterations=6,
            residual_tol=1e-10,
            target_count=4,
            target_count_certified=true,
            maxrank=4,
            rollback_ratio=1.05,
        ),
        state_error=error_metric,
    )
    values = eigvals(state_components(result.state).right_state)
    @test all(value -> FusedNLFEAST.in_chart(chart, value), values)
    @test multiset_distance(values, expected) > 1e-3
    @test result.history[end].boundary_margin > 0
    @test result.count_matched
    @test !result.residual_converged
    @test !result.certified
end

@testset "state driver retains an exact double root" begin
    square = value -> value^2
    error_metric = function (state)
        components = state_components(state)
        lifted_residual_error(
            components.right_state,
            components.right,
            components.right_residual,
            components.left_state,
            components.left,
            components.left_residual;
            lift_depth=2,
        )
    end
    result = structured_fused_state_nlfeast(
        (ones(ComplexF64, 1, 1),),
        (square,),
        (z, block) -> block ./ z^2,
        (z, block) -> block ./ conj(z)^2,
        CircularChart(0.0, 1.0, 16),
        ones(ComplexF64, 1, 1),
        ones(ComplexF64, 1, 1);
        config=StateIterationConfig(
            moment_count=2,
            iterations=3,
            residual_tol=1e-10,
            target_count=2,
            target_count_certified=true,
            maxrank=2,
        ),
        state_error=error_metric,
    )
    components = state_components(result.state)
    right_lift = vcat(
        components.right,
        components.right * components.right_state,
    )
    @test size(components.right_state, 1) == 2
    @test rank(right_lift; rtol=1e-10) == 2
    @test result.history[end].error <= 1e-10
    @test result.certified
    @test !state_modal_output(result.state).available
end

@testset "state driver augments a capacity-deficient initial probe" begin
    values = ComplexF64[-0.5, 0.2, 0.8]
    operator = Diagonal(values)
    identity_map = Matrix{ComplexF64}(I, 3, 3)
    T = z -> z .* identity_map .- operator
    right_solve = (z, block) -> T(z) \ block
    left_solve = (z, block) -> adjoint(T(z)) \ block
    extra = reshape(ComplexF64[1.0, im, 2.0], :, 1)
    result = structured_fused_state_nlfeast(
        (-Matrix(operator), identity_map),
        (one, identity),
        right_solve,
        left_solve,
        CircularChart(0.0, 1.2, 24),
        ones(ComplexF64, 3, 1),
        ones(ComplexF64, 3, 1);
        config=StateIterationConfig(
            moment_count=2,
            iterations=4,
            residual_tol=1e-10,
            target_count=3,
            target_count_certified=true,
            maxrank=4,
            max_initial_probe_width=2,
        ),
        augment_probe=(side, width, attempt) -> copy(extra),
    )
    components = state_components(result.state)
    @test multiset_distance(eigvals(components.right_state), values) <= 1e-9
    @test result.history[1].selection === :augmented_initial
    @test result.history[1].right_probe_width == 2
    @test result.history[1].left_probe_width == 2
    @test result.certified
end

@testset "black-box holomorphic state driver uses operator actions" begin
    values = ComplexF64[-0.6, 0.15, 0.7]
    operator = Diagonal(values)
    identity_map = Matrix{ComplexF64}(I, 3, 3)
    T = z -> z .* identity_map .- operator
    chart = CircularChart(0.0, 1.0, 64)
    rng = MersenneTwister(813)
    result = holomorphic_fused_state_nlfeast(
        (z, block) -> T(z) \ block,
        (z, block) -> adjoint(T(z)) \ block,
        (z, block) -> T(z) * block,
        (z, block) -> adjoint(T(z)) * block,
        chart,
        randn(rng, ComplexF64, 3, 2),
        randn(rng, ComplexF64, 3, 2);
        config=StateIterationConfig(
            moment_count=2,
            iterations=4,
            residual_tol=1e-10,
            target_count=3,
            target_count_certified=true,
            maxrank=4,
        ),
        operator_scale=2.0,
    )
    components = state_components(result.state)
    @test multiset_distance(eigvals(components.right_state), values) <= 1e-9
    @test result.history[end].error <= 1e-10
    @test result.converged
    @test !result.residual_certified
    @test !result.certified
end
