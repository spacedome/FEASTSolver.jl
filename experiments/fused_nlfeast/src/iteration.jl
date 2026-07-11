Base.@kwdef struct FusedConfig
    moment_count::Int
    iterations::Int = 4
    ranktol::Float64 = 1e-10
    residual_ranktol::Float64 = 1e-12
    residual_tol::Float64 = 1e-10
    residual_scale::Union{Nothing,Float64} = nothing
    residual_metric::Union{Nothing,Function} = nothing
    visibilitytol::Float64 = 1e-10
    matchtol::Float64 = 1e-8
    maxrank::Int = typemax(Int)
    coupling::Symbol = :hybrid
    common_ranktol::Float64 = 1e-10
    common_acceptance_ratio::Float64 = 0.9
    right_moment_width::Union{Nothing,Int} = nothing
    left_moment_width::Union{Nothing,Int} = nothing
    moment_cluster_tol::Float64 = 1e-6
    target_count::Union{Nothing,Int} = nothing
    target_count_certified::Bool = false
end

struct IterationRecord
    iteration::Int
    rank::Int
    count::Int
    max_residual::Float64
    max_match_error::Float64
    right_residual_rank::Int
    left_residual_rank::Int
    coupling_accepted::Bool
    loewner_condition::Float64
    intertwining_defect::Float64
    overlap_rank::Int
    selection_mode::Symbol
    right_moment_width::Int
    left_moment_width::Int
end

struct FusedResult{C}
    extraction::FusedExtraction
    history::Vector{IterationRecord}
    cache::C
    config::FusedConfig
    residual_scale::Union{Nothing,Float64}
    pairing_accepted::Bool
    residual_converged::Bool
    count_matched::Bool
    count_certified::Bool
    converged::Bool
    certified::Bool
    termination_reason::Symbol
end

function iteration_accepted(record, chart, config)
    pairing_accepted = record.selection_mode === :common ||
        record.max_match_error <= config.matchtol * chart.radius
    count_accepted = config.target_count === nothing || record.count == config.target_count
    record.max_residual <= config.residual_tol && pairing_accepted && count_accepted
end

function iteration_record(
    iteration,
    extraction,
    right_residual_rank=0,
    left_residual_rank=0,
    right_moment_width=0,
    left_moment_width=0,
)
    IterationRecord(
        iteration,
        extraction.rank,
        length(extraction.values),
        isempty(extraction.residuals) ? Inf : maximum(extraction.residuals),
        isempty(extraction.match_errors) ? Inf : maximum(extraction.match_errors),
        right_residual_rank,
        left_residual_rank,
        extraction.coupling.accepted,
        extraction.coupling.condition_number,
        extraction.coupling.intertwining_defect,
        extraction.coupling.rank,
        extraction.selection_mode,
        right_moment_width,
        left_moment_width,
    )
end

function fused_nlfeast(
    T,
    right_solve,
    left_solve,
    chart::AbstractContourChart,
    right_probe,
    left_probe;
    config::FusedConfig,
    divided_difference=nothing,
    divided_overlap=nothing,
    analytic_domain=nothing,
)
    validate_analytic_chart(chart, analytic_domain)
    config.moment_count > 0 || throw(ArgumentError("moment_count must be positive"))
    config.iterations >= 0 || throw(ArgumentError("iterations must be nonnegative"))
    config.ranktol > 0 || throw(ArgumentError("ranktol must be positive"))
    config.residual_ranktol >= 0 || throw(ArgumentError("residual_ranktol must be nonnegative"))
    config.residual_tol >= 0 || throw(ArgumentError("residual_tol must be nonnegative"))
    config.residual_scale === nothing || config.residual_scale > 0 ||
        throw(ArgumentError("residual_scale must be positive when supplied"))
    config.residual_scale === nothing || config.residual_metric === nothing ||
        throw(ArgumentError("pass either residual_scale or residual_metric, not both"))
    config.visibilitytol >= 0 || throw(ArgumentError("visibilitytol must be nonnegative"))
    config.matchtol >= 0 || throw(ArgumentError("matchtol must be nonnegative"))
    config.maxrank > 0 || throw(ArgumentError("maxrank must be positive"))
    config.coupling in (:independent, :hybrid) || throw(ArgumentError(
        "coupling must be :independent or :hybrid",
    ))
    config.common_ranktol > 0 || throw(ArgumentError("common_ranktol must be positive"))
    0 < config.common_acceptance_ratio <= 1 || throw(ArgumentError(
        "common_acceptance_ratio must lie in (0, 1]",
    ))
    config.right_moment_width === nothing || config.right_moment_width > 0 ||
        throw(ArgumentError("right_moment_width must be positive when supplied"))
    config.left_moment_width === nothing || config.left_moment_width > 0 ||
        throw(ArgumentError("left_moment_width must be positive when supplied"))
    config.moment_cluster_tol >= 0 || throw(ArgumentError("moment_cluster_tol must be nonnegative"))
    config.target_count === nothing || config.target_count > 0 ||
        throw(ArgumentError("target_count must be positive when supplied"))
    !config.target_count_certified || config.target_count !== nothing ||
        throw(ArgumentError("target_count_certified requires target_count"))
    2 * config.moment_count <= length(chart.nodes) || throw(ArgumentError(
        "circular moment sequences require at least 2 * moment_count contour nodes to avoid power aliasing",
    ))
    size(right_probe, 1) == size(left_probe, 1) || throw(DimensionMismatch(
        "right and left probes must have the same physical dimension",
    ))
    size(right_probe, 2) > 0 || throw(ArgumentError("right probe must have at least one column"))
    size(left_probe, 2) > 0 || throw(ArgumentError("left probe must have at least one column"))
    cache = ContourSampleCache(chart, right_solve, left_solve)
    initial_right = add_right_probe!(cache, :initial, right_probe; role=:realization)
    initial_left = add_left_probe!(cache, :initial, left_probe; role=:realization)
    right_observer = initial_left.probe
    left_observer = initial_right.probe
    residual_scale = if config.residual_metric !== nothing
        nothing
    elseif config.residual_scale === nothing
        maximum(opnorm(Matrix{ComplexF64}(T(z))) for z in chart.nodes)
    else
        config.residual_scale
    end
    residual_scale === nothing || residual_scale > 0 || error("operator scale is zero on every contour node")
    right_moments = probe_moments(cache, :initial, :right, 2 * config.moment_count)
    left_moments = probe_moments(cache, :initial, :left, 2 * config.moment_count)
    extraction = extract_two_sided(
        T,
        chart,
        right_moments,
        left_moments,
        right_observer,
        left_observer,
        config.moment_count;
        ranktol=config.ranktol,
        maxrank=config.maxrank,
        residual_scale=residual_scale,
        residual_metric=config.residual_metric,
        visibilitytol=config.visibilitytol,
        coupling=config.coupling,
        divided_difference=divided_difference,
        divided_overlap=divided_overlap,
        common_ranktol=config.common_ranktol,
        common_acceptance_ratio=config.common_acceptance_ratio,
    )
    history = IterationRecord[iteration_record(
        0,
        extraction,
        0,
        0,
        size(right_moments[1], 2),
        size(left_moments[1], 2),
    )]

    for iteration in 1:config.iterations
        iteration_accepted(history[end], chart, config) && break
        isempty(extraction.values) && break
        factors = residual_factors(
            T,
            extraction.right_interpolation_values,
            extraction.left_interpolation_values,
            extraction.right,
            extraction.left;
            ranktol=config.residual_ranktol,
        )
        right_id = Symbol("right_residual_", iteration)
        left_id = Symbol("left_residual_", iteration)
        right_tangent = moment_tangent(
            extraction.right_interpolation_values,
            config.moment_count,
            config.right_moment_width;
            cluster_tolerance=config.moment_cluster_tol,
            side=:right,
        )
        left_tangent = moment_tangent(
            extraction.left_interpolation_values,
            config.moment_count,
            config.left_moment_width;
            cluster_tolerance=config.moment_cluster_tol,
            side=:left,
        )
        add_right_probe!(cache, right_id, factors.right_basis; role=:residual)
        add_left_probe!(cache, left_id, factors.left_basis; role=:residual)
        right_moments, left_moments = corrected_moments(
            cache,
            extraction.right_interpolation_values,
            extraction.left_interpolation_values,
            extraction.right,
            extraction.left,
            right_id,
            left_id,
            factors.right_coefficients,
            factors.left_coefficients,
            2 * config.moment_count,
            right_tangent=right_tangent,
            left_tangent=left_tangent,
        )
        extraction = extract_two_sided(
            T,
            chart,
            right_moments,
            left_moments,
            right_observer,
            left_observer,
            config.moment_count;
            ranktol=config.ranktol,
            maxrank=config.maxrank,
            residual_scale=residual_scale,
            residual_metric=config.residual_metric,
            visibilitytol=config.visibilitytol,
            coupling=config.coupling,
            divided_difference=divided_difference,
            divided_overlap=divided_overlap,
            common_ranktol=config.common_ranktol,
            common_acceptance_ratio=config.common_acceptance_ratio,
        )
        push!(
            history,
            iteration_record(
                iteration,
                extraction,
                size(factors.right_basis, 2),
                size(factors.left_basis, 2),
                size(right_moments[1], 2),
                size(left_moments[1], 2),
            ),
        )
    end
    pairing_accepted = extraction.selection_mode === :common ||
        (!isempty(extraction.match_errors) &&
         maximum(extraction.match_errors) <= config.matchtol * chart.radius)
    residual_converged = !isempty(extraction.residuals) &&
        maximum(extraction.residuals) <= config.residual_tol && pairing_accepted
    count_matched = config.target_count !== nothing &&
        length(extraction.values) == config.target_count
    count_certified = count_matched && config.target_count_certified
    converged = residual_converged && count_matched
    certified = converged && count_certified
    termination_reason = if certified
        :converged_certified
    elseif converged
        :converged_uncertified
    elseif residual_converged && config.target_count === nothing
        :residual_converged_uncertified
    elseif residual_converged && !count_matched
        :count_mismatch
    elseif isempty(extraction.values)
        :empty_extraction
    elseif !pairing_accepted
        :pairing_mismatch
    else
        :iteration_limit
    end
    FusedResult(
        extraction,
        history,
        cache,
        config,
        residual_scale,
        pairing_accepted,
        residual_converged,
        count_matched,
        count_certified,
        converged,
        certified,
        termination_reason,
    )
end
