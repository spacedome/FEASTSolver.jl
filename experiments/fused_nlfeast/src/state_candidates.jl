function validate_state_config(config)
    config.moment_count > 0 || throw(ArgumentError("moment_count must be positive"))
    config.iterations >= 0 || throw(ArgumentError("iterations must be nonnegative"))
    config.ranktol > 0 || throw(ArgumentError("ranktol must be positive"))
    config.residual_ranktol >= 0 || throw(ArgumentError("residual_ranktol must be nonnegative"))
    config.residual_tol >= 0 || throw(ArgumentError("residual_tol must be nonnegative"))
    config.maxrank > 0 || throw(ArgumentError("maxrank must be positive"))
    config.target_count === nothing || config.target_count > 0 || throw(ArgumentError(
        "target_count must be positive when supplied",
    ))
    !config.target_count_certified || config.target_count !== nothing || throw(ArgumentError(
        "target_count_certified requires target_count",
    ))
    config.common_ranktol > 0 || throw(ArgumentError("common_ranktol must be positive"))
    config.common_acceptance_ratio > 0 || throw(ArgumentError(
        "common_acceptance_ratio must be positive",
    ))
    config.tangent_width === nothing || config.tangent_width > 0 || throw(ArgumentError(
        "tangent_width must be positive when supplied",
    ))
    config.compress_after >= 0 || throw(ArgumentError("compress_after must be nonnegative"))
    config.max_initial_probe_width === nothing || config.max_initial_probe_width > 0 ||
        throw(ArgumentError("max_initial_probe_width must be positive when supplied"))
    config.rollback_ratio >= 1 || throw(ArgumentError("rollback_ratio must be at least one"))
    config.max_stagnation > 0 || throw(ArgumentError("max_stagnation must be positive"))
    config.minimum_contour_separation >= 0 || throw(ArgumentError(
        "minimum_contour_separation must be nonnegative",
    ))
    config.maximum_solve_residual >= 0 || throw(ArgumentError(
        "maximum_solve_residual must be nonnegative",
    ))
end

function recoverable_state_candidate_error(error)
    error isa StateCandidateFailure && return true
    message = sprint(showerror, error)
    if error isa ArgumentError
        return occursin("state must lie inside the Cauchy chart", message)
    end
    if error isa DimensionMismatch
        return occursin("interior state", message) ||
            occursin("interior states", message) ||
            occursin("interior state counts", message) ||
            occursin("candidate states", message)
    end
    error isa ErrorException || return false
    occursin("numerical rank zero", message) ||
        occursin("no states inside", message) ||
        occursin("overlap is numerically rank deficient", message) ||
        occursin("state contour separation", message)
end

function classify_state_candidate_failure(error)
    message = sprint(showerror, error)
    if error isa ArgumentError
        occursin("state must lie inside the Cauchy chart", message) && return :outside_chart
        occursin("fixed realization rank", message) && return :insufficient_capacity
    elseif error isa DimensionMismatch
        if occursin("interior state", message) || occursin("interior states", message) ||
           occursin("interior state counts", message) || occursin("candidate states", message)
            return :insufficient_states
        end
    elseif error isa LinearAlgebra.SingularException
        return :nonminimal_lift
    elseif error isa ErrorException
        occursin("numerical rank zero", message) && return :rank_deficient
        occursin("no states inside", message) && return :empty_chart
        occursin("not minimal", message) && return :nonminimal_lift
        occursin("state contour separation", message) && return :contour_separation
    end
    nothing
end

function state_candidate(
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
    fixed_rank=nothing,
    restrict_to_chart=true,
)
    target_count = fixed_rank === nothing ? config.target_count : nothing
    independent = try
        extracted = independent_state_realization(
            chart,
            right_moments,
            left_moments,
            right_observer,
            left_observer,
            config.moment_count;
            ranktol=config.ranktol,
            maxrank=config.maxrank,
            fixed_rank=fixed_rank,
            restrict_to_chart=restrict_to_chart,
            target_count=target_count,
            right_residual=right_residual,
            left_residual=left_residual,
        )
        lift_normalized_independent_state(
            extracted,
            config.moment_count;
            center=chart.center,
            radius=chart.radius,
        )
    catch error
        reason = classify_state_candidate_failure(error)
        reason === nothing && rethrow(error)
        throw(StateCandidateFailure(reason, sprint(showerror, error)))
    end
    independent_errors = state_error_data(state_error, independent)
    selected = if state_overlap === nothing
        (state=independent, errors=independent_errors)
    else
        common = try
            common_state_from_independent(
                independent;
                overlap_ranktol=config.common_ranktol,
                right_residual=right_residual,
                left_residual=left_residual,
                state_overlap=state_overlap,
            )
        catch error
            recoverable_state_candidate_error(error) || rethrow(error)
            nothing
        end
        if common === nothing
            (state=independent, errors=independent_errors)
        else
            common_errors = state_error_data(state_error, common)
            common_allowed = !restrict_to_chart || state_inside_chart(chart, common)
            common_allowed && common_errors.maximum <=
                config.common_acceptance_ratio * independent_errors.maximum ?
                (state=common, errors=common_errors) :
                (state=independent, errors=independent_errors)
        end
    end
    if restrict_to_chart && config.minimum_contour_separation > 0
        separation = state_contour_separation(chart, selected.state)
        if separation.minimum < config.minimum_contour_separation
            throw(StateCandidateFailure(
                :contour_separation,
                "state contour separation is below the configured minimum",
            ))
        end
    end
    selected
end
