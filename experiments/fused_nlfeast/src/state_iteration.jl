function fused_state_nlfeast(
    right_solve,
    left_solve,
    chart::AbstractContourChart,
    right_probe,
    left_probe;
    config::StateIterationConfig,
    right_residual,
    left_residual,
    state_error,
    state_overlap=nothing,
    augment_probe=nothing,
    analytic_domain=nothing,
    residual_certified=false,
)
    validate_state_config(config)
    validate_analytic_chart(chart, analytic_domain)
    2 * config.moment_count <= length(chart.nodes) || throw(ArgumentError(
        "state Hankel sequences require at least 2 * moment_count contour nodes",
    ))
    size(right_probe, 1) == size(left_probe, 1) || throw(DimensionMismatch(
        "right and left probes must have the same physical dimension",
    ))
    cache = ContourSampleCache(chart, right_solve, left_solve)
    add_right_probe!(
        cache,
        :initial,
        right_probe;
        role=:realization,
        require_diagnostics=config.require_solve_diagnostics,
        maximum_relative_residual=config.maximum_solve_residual,
    )
    add_left_probe!(
        cache,
        :initial,
        left_probe;
        role=:realization,
        require_diagnostics=config.require_solve_diagnostics,
        maximum_relative_residual=config.maximum_solve_residual,
    )
    right_ids = Symbol[:initial]
    left_ids = Symbol[:initial]
    augmentation = 0
    candidate = nothing

    while candidate === nothing
        right_observer = combined_probe(cache, left_ids, :left)
        left_observer = combined_probe(cache, right_ids, :right)
        right_moments = combined_probe_moments(
            cache,
            right_ids,
            :right,
            2 * config.moment_count,
        )
        left_moments = combined_probe_moments(
            cache,
            left_ids,
            :left,
            2 * config.moment_count,
        )
        candidate = try
            state_candidate(
                chart,
                right_moments,
                left_moments,
                right_observer,
                left_observer,
                config,
                state_error,
                right_residual,
                left_residual,
                state_overlap,
            )
        catch error
            current_width = max(size(right_observer, 2), size(left_observer, 2))
            maximum_width = something(config.max_initial_probe_width, current_width)
            if !recoverable_state_candidate_error(error) ||
               augment_probe === nothing || current_width >= maximum_width
                rethrow(error)
            end
            augmentation += 1
            right_extra = augment_probe(:right, size(left_observer, 2), augmentation)
            left_extra = augment_probe(:left, size(right_observer, 2), augmentation)
            right_id = Symbol(:initial_right_, augmentation)
            left_id = Symbol(:initial_left_, augmentation)
            add_right_probe!(
                cache,
                right_id,
                right_extra;
                role=:realization,
                require_diagnostics=config.require_solve_diagnostics,
                maximum_relative_residual=config.maximum_solve_residual,
            )
            add_left_probe!(
                cache,
                left_id,
                left_extra;
                role=:realization,
                require_diagnostics=config.require_solve_diagnostics,
                maximum_relative_residual=config.maximum_solve_residual,
            )
            push!(right_ids, right_id)
            push!(left_ids, left_id)
            nothing
        end
    end

    state = candidate.state
    errors = candidate.errors
    right_observer = combined_probe(cache, left_ids, :left)
    left_observer = combined_probe(cache, right_ids, :right)
    foreach(id -> release_right_responses!(cache, id), right_ids)
    foreach(id -> release_left_responses!(cache, id), left_ids)
    history = StateIterationRecord[state_iteration_record(
        0,
        state,
        errors,
        augmentation > 0 ? :augmented_initial : :initial,
        0,
        0,
        size(left_observer, 2),
        size(right_observer, 2),
        chart,
        false,
    )]
    stagnation = 0
    termination_reason = :iteration_limit

    for iteration in 1:config.iterations
        errors.maximum <= config.residual_tol && (termination_reason = :residual_tolerance; break)
        raw_components = state_components(state)
        components = lift_normalized_components(
            raw_components.right_state,
            raw_components.right,
            raw_components.right_residual,
            raw_components.left_state,
            raw_components.left,
            raw_components.left_residual;
            lift_depth=config.moment_count,
            lift_center=chart.center,
            lift_radius=chart.radius,
        )
        factors = state_residual_factors(
            components.right_residual,
            components.left_residual;
            ranktol=config.residual_ranktol,
        )
        right_id = Symbol(:state_right_, iteration)
        left_id = Symbol(:state_left_, iteration)
        add_right_probe!(
            cache,
            right_id,
            factors.right_basis;
            role=:residual,
            require_diagnostics=config.require_solve_diagnostics,
            maximum_relative_residual=config.maximum_solve_residual,
        )
        add_left_probe!(
            cache,
            left_id,
            factors.left_basis;
            role=:residual,
            require_diagnostics=config.require_solve_diagnostics,
            maximum_relative_residual=config.maximum_solve_residual,
        )
        active_width = iteration > config.compress_after ? config.tangent_width : nothing
        right_tangent, left_tangent = if active_width === nothing
            nothing, nothing
        else
            state_count = size(components.right_state, 1)
            identity_state = Matrix{ComplexF64}(I, state_count, state_count)
            right_chart_state = (
                components.right_state .- chart.center .* identity_state
            ) ./ chart.radius
            left_chart_state = (
                components.left_state .- chart.center .* identity_state
            ) ./ chart.radius
            try
                (
                    state_moment_tangent(
                        right_chart_state,
                        config.moment_count,
                        active_width;
                        side=:right,
                        ranktol=config.ranktol,
                    ),
                    state_moment_tangent(
                        left_chart_state,
                        config.moment_count,
                        active_width;
                        side=:left,
                        ranktol=config.ranktol,
                    ),
                )
            catch error
                error isa ArgumentError || rethrow(error)
                nothing, nothing
            end
        end
        right_moments, left_moments = state_corrected_moments(
            cache,
            components.right_state,
            components.left_state,
            components.right,
            components.left,
            right_id,
            left_id,
            factors.right_coefficients,
            factors.left_coefficients,
            2 * config.moment_count;
            right_tangent=right_tangent,
            left_tangent=left_tangent,
        )
        selection = :chart
        compression_rejected = false
        next_candidate = try
            state_candidate(
                chart,
                right_moments,
                left_moments,
                right_observer,
                left_observer,
                config,
                state_error,
                right_residual,
                left_residual,
                state_overlap,
            )
        catch error
            recoverable_state_candidate_error(error) || rethrow(error)
            nothing
        end

        if right_tangent !== nothing &&
           (next_candidate === nothing || next_candidate.errors.maximum > errors.maximum)
            compression_rejected = true
            right_moments, left_moments = state_corrected_moments(
                cache,
                components.right_state,
                components.left_state,
                components.right,
                components.left,
                right_id,
                left_id,
                factors.right_coefficients,
                factors.left_coefficients,
                2 * config.moment_count,
            )
            next_candidate = try
                state_candidate(
                    chart,
                    right_moments,
                    left_moments,
                    right_observer,
                    left_observer,
                    config,
                    state_error,
                    right_residual,
                    left_residual,
                    state_overlap,
                )
            catch error
                recoverable_state_candidate_error(error) || rethrow(error)
                nothing
            end
        end

        fixed_candidate = try
            state_candidate(
                chart,
                right_moments,
                left_moments,
                right_observer,
                left_observer,
                config,
                state_error,
                right_residual,
                left_residual,
                state_overlap;
                fixed_rank=state_rank(state),
                restrict_to_chart=false,
            )
        catch error
            recoverable_state_candidate_error(error) || rethrow(error)
            nothing
        end
        fixed_separated = fixed_candidate !== nothing && (
            config.minimum_contour_separation == 0 ||
            state_contour_separation(chart, fixed_candidate.state).minimum >=
                config.minimum_contour_separation
        )
        if fixed_separated && state_inside_chart(chart, fixed_candidate.state) &&
           (next_candidate === nothing || fixed_candidate.errors.maximum < next_candidate.errors.maximum)
            next_candidate = fixed_candidate
            selection = :fixed
        end
        if next_candidate === nothing ||
           next_candidate.errors.maximum > config.rollback_ratio * errors.maximum
            next_candidate = (state=state, errors=errors)
            selection = :rollback
            stagnation += 1
        else
            stagnation = next_candidate.errors.maximum < errors.maximum ? 0 : stagnation + 1
        end
        state = next_candidate.state
        errors = next_candidate.errors
        drop_right_probe!(cache, right_id)
        drop_left_probe!(cache, left_id)
        push!(history, state_iteration_record(
            iteration,
            state,
            errors,
            selection,
            size(factors.right_basis, 2),
            size(factors.left_basis, 2),
            size(left_observer, 2),
            size(right_observer, 2),
            chart,
            compression_rejected,
        ))
        if stagnation >= config.max_stagnation
            termination_reason = :stagnation
            break
        end
    end

    residual_converged = errors.maximum <= config.residual_tol
    count_matched = config.target_count !== nothing &&
        state_rank(state) == config.target_count && state_inside_chart(chart, state)
    count_certified = count_matched && config.target_count_certified
    converged = residual_converged && count_matched
    certified = converged && count_certified && residual_certified
    termination_reason = if certified
        :converged_certified
    elseif converged
        :converged_uncertified
    elseif residual_converged
        :residual_converged_uncertified
    else
        termination_reason
    end
    StateIterationResult(
        state,
        history,
        cache,
        config,
        residual_converged,
        count_matched,
        count_certified,
        residual_certified,
        converged,
        certified,
        termination_reason,
    )
end
