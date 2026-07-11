function structured_fused_state_nlfeast(
    coefficients,
    functions,
    right_solve,
    left_solve,
    chart,
    right_probe,
    left_probe;
    config::StateIterationConfig,
    augment_probe=nothing,
    state_error=nothing,
    analytic_domain=nothing,
    coupling=:common,
)
    coupling in (:common, :independent) || throw(ArgumentError(
        "coupling must be :common or :independent",
    ))
    right_residual = (right, state) -> structured_invariant_residual(
        coefficients,
        functions,
        right,
        state,
    )
    left_residual = (left, state) -> structured_left_invariant_residual(
        coefficients,
        functions,
        left,
        state,
    )
    state_overlap = coupling === :common ?
        (
            (left_state, left, right_state, right) ->
                structured_state_divided_overlap(
                    coefficients,
                    functions,
                    left_state,
                    left,
                    right_state,
                    right,
                )
        ) : nothing
    error_metric = if state_error === nothing
        function (state)
            components = state_components(state)
            structured_pair_backward_error(
                coefficients,
                functions,
                components.right_state,
                components.right,
                components.left_state,
                components.left;
                lift_depth=config.moment_count,
                lift_center=chart.center,
                lift_radius=chart.radius,
            )
        end
    else
        state_error
    end
    fused_state_nlfeast(
        right_solve,
        left_solve,
        chart,
        right_probe,
        left_probe;
        config=config,
        right_residual=right_residual,
        left_residual=left_residual,
        state_overlap=state_overlap,
        state_error=error_metric,
        augment_probe=augment_probe,
        analytic_domain=analytic_domain,
        residual_certified=true,
    )
end

function holomorphic_fused_state_nlfeast(
    right_solve,
    left_solve,
    right_action,
    left_action,
    chart,
    right_probe,
    left_probe;
    config::StateIterationConfig,
    operator_scale,
    augment_probe=nothing,
    analytic_domain=nothing,
    action_refinement_factor=2,
    action_resolution_certified=false,
    coupling=:common,
)
    coupling in (:common, :independent) || throw(ArgumentError(
        "coupling must be :common or :independent",
    ))
    action_refinement_factor >= 1 || throw(ArgumentError(
        "action_refinement_factor must be at least one",
    ))
    action_chart = action_refinement_factor == 1 ? chart :
        refine_chart(chart, action_refinement_factor * length(chart.nodes))
    right_residual = (right, state) -> cauchy_right_invariant_residual(
        action_chart,
        state,
        right,
        right_action,
    )
    left_residual = (left, state) -> cauchy_left_invariant_residual(
        action_chart,
        state,
        left,
        left_action,
    )
    state_overlap = coupling === :common ?
        (
            (left_state, left, right_state, right) ->
                cauchy_state_divided_overlap(
                    action_chart,
                    left_state,
                    left,
                    right_state,
                    right,
                    right_action,
                )
        ) : nothing
    state_error = function (state)
        components = state_components(state)
        lifted_residual_error(
            components.right_state,
            components.right,
            components.right_residual,
            components.left_state,
            components.left,
            components.left_residual;
            lift_depth=config.moment_count,
            operator_scale=operator_scale,
            lift_center=chart.center,
            lift_radius=chart.radius,
        )
    end
    fused_state_nlfeast(
        right_solve,
        left_solve,
        chart,
        right_probe,
        left_probe;
        config=config,
        right_residual=right_residual,
        left_residual=left_residual,
        state_overlap=state_overlap,
        state_error=state_error,
        augment_probe=augment_probe,
        analytic_domain=analytic_domain,
        residual_certified=action_resolution_certified,
    )
end
