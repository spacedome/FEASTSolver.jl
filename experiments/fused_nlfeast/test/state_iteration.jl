@testset "common-state iteration with shared eigenvectors" begin
    case = shared_eigenvector_case()
    for contour_index in (1, 3)
        contour = case.contours[contour_index]
        chart = CircularChart(contour.center, contour.radius, 16)
        rng = MersenneTwister(900 + contour_index)
        cache = ContourSampleCache(chart, case.right_solve, case.left_solve)
        right_initial = add_right_probe!(cache, :initial, rand(rng, ComplexF64, 3, 1))
        left_initial = add_left_probe!(cache, :initial, rand(rng, ComplexF64, 3, 1))
        right_moments = probe_moments(cache, :initial, :right, 4)
        left_moments = probe_moments(cache, :initial, :left, 4)
        common = common_polynomial_realization(
            case.coefficients,
            chart,
            right_moments,
            left_moments,
            left_initial.probe,
            right_initial.probe,
            2;
            ranktol=1e-9,
            maxrank=2,
        )
        expected_state = Diagonal(contour.expected)
        expected_right_lift = vcat(
            contour.expected_right,
            contour.expected_right * expected_state,
        )
        expected_left_lift = vcat(
            contour.expected_left,
            contour.expected_left * adjoint(expected_state),
        )
        initial_right_gap = realization_subspace_gap(
            vcat(common.right, common.right * common.state),
            expected_right_lift,
        )
        initial_left_gap = realization_subspace_gap(
            vcat(common.left, common.left * adjoint(common.state)),
            expected_left_lift,
        )
        residual_history = Float64[state_modal_data(case.T, common).residual]
        for iteration in 1:7
            factors = state_residual_factors(
                common.right_residual,
                common.left_residual;
                ranktol=1e-13,
            )
            right_id = Symbol("state_right_", contour_index, "_", iteration)
            left_id = Symbol("state_left_", contour_index, "_", iteration)
            add_right_probe!(cache, right_id, factors.right_basis; role=:residual)
            add_left_probe!(cache, left_id, factors.left_basis; role=:residual)
            right_moments, left_moments = state_corrected_moments(
                cache,
                common.state,
                common.state,
                common.right,
                common.left,
                right_id,
                left_id,
                factors.right_coefficients,
                factors.left_coefficients,
                4,
            )
            common = common_polynomial_realization(
                case.coefficients,
                chart,
                right_moments,
                left_moments,
                left_initial.probe,
                right_initial.probe,
                2;
                ranktol=1e-9,
                maxrank=2,
            )
            push!(residual_history, state_modal_data(case.T, common).residual)
        end
        modal = state_modal_data(case.T, common)
        @test multiset_distance(modal.values, contour.expected) <= 1e-9
        @test residual_history[end] <= 1e-9
        @test residual_history[end] <= 1e-7 * residual_history[1]
        @test common.rank == 2
        @test istriu(common.state)
        final_right_gap = realization_subspace_gap(
            vcat(common.right, common.right * common.state),
            expected_right_lift,
        )
        final_left_gap = realization_subspace_gap(
            vcat(common.left, common.left * adjoint(common.state)),
            expected_left_lift,
        )
        @test final_right_gap <= 1e-8
        @test final_left_gap <= 1e-8
        @test final_right_gap <= 1e-4 * initial_right_gap
        @test final_left_gap <= 1e-4 * initial_left_gap
        if contour.shared === :left
            @test rank(modal.left; rtol=1e-8) == 1
            @test rank(vcat(common.left, common.left * adjoint(common.state)); rtol=1e-8) == 2
        else
            @test rank(modal.right; rtol=1e-8) == 1
            @test rank(vcat(common.right, common.right * common.state); rtol=1e-8) == 2
        end
    end
end
