Base.@kwdef struct StateIterationConfig
    moment_count::Int
    iterations::Int = 8
    ranktol::Float64 = 1e-10
    residual_ranktol::Float64 = 1e-12
    residual_tol::Float64 = 1e-10
    maxrank::Int = typemax(Int)
    target_count::Union{Nothing,Int} = nothing
    target_count_certified::Bool = false
    common_ranktol::Float64 = 1e-10
    common_acceptance_ratio::Float64 = 1.05
    tangent_width::Union{Nothing,Int} = nothing
    compress_after::Int = typemax(Int)
    max_initial_probe_width::Union{Nothing,Int} = nothing
    rollback_ratio::Float64 = 1.0
    max_stagnation::Int = 2
    minimum_contour_separation::Float64 = 0.0
    require_solve_diagnostics::Bool = false
    maximum_solve_residual::Float64 = Inf
end

struct InvalidAnalyticDomainError <: Exception
    reason::String
end

struct StateCandidateFailure <: Exception
    reason::Symbol
    detail::String
end

Base.showerror(io::IO, error::StateCandidateFailure) =
    print(io, "state candidate failure [", error.reason, "]: ", error.detail)

Base.showerror(io::IO, error::InvalidAnalyticDomainError) =
    print(io, "invalid analytic chart: ", error.reason)

function validate_analytic_chart(chart, analytic_domain)
    analytic_domain === nothing && return
    outcome = analytic_domain(chart)
    valid, reason = if outcome isa Bool
        outcome, outcome ? "" : "operator is not single-valued and analytic on the chart"
    elseif hasproperty(outcome, :valid)
        Bool(outcome.valid), hasproperty(outcome, :reason) ? string(outcome.reason) :
            "operator is not single-valued and analytic on the chart"
    else
        throw(ArgumentError("analytic_domain must return Bool or a value with a valid field"))
    end
    valid || throw(InvalidAnalyticDomainError(reason))
end

struct StateIterationRecord
    iteration::Int
    error::Float64
    right_error::Float64
    left_error::Float64
    rank::Int
    representation::Symbol
    selection::Symbol
    right_residual_rank::Int
    left_residual_rank::Int
    right_probe_width::Int
    left_probe_width::Int
    overlap_ratio::Float64
    boundary_margin::Float64
    right_contour_separation::Float64
    left_contour_separation::Float64
    compression_rejected::Bool
end

struct StateIterationResult{S,C}
    state::S
    history::Vector{StateIterationRecord}
    cache::C
    config::StateIterationConfig
    residual_converged::Bool
    count_matched::Bool
    count_certified::Bool
    residual_certified::Bool
    converged::Bool
    certified::Bool
    termination_reason::Symbol
end

state_components(state::CommonStateRealization) = (
    right_state=state.state,
    left_state=state.state,
    right=state.right,
    left=state.left,
    right_residual=state.right_residual,
    left_residual=state.left_residual,
)

state_components(state::IndependentStateRealization) = (
    right_state=state.right_state,
    left_state=state.left_state,
    right=state.right,
    left=state.left,
    right_residual=state.right_residual,
    left_residual=state.left_residual,
)

function lift_normalized_independent_state(
    state::IndependentStateRealization,
    depth;
    center=0.0,
    radius=1.0,
)
    components = lift_normalized_components(
        state.right_state,
        state.right,
        state.right_residual,
        state.left_state,
        state.left,
        state.left_residual;
        lift_depth=depth,
        lift_center=center,
        lift_radius=radius,
    )
    IndependentStateRealization(
        components.right_state,
        components.left_state,
        components.right,
        components.left,
        components.right_residual,
        components.left_residual,
        state.right_singular_values,
        state.left_singular_values,
        state.right_detected_rank,
        state.left_detected_rank,
        state.rank,
    )
end

state_representation(::CommonStateRealization) = :common
state_representation(::IndependentStateRealization) = :independent
state_rank(state) = size(state_components(state).right_state, 1)

function state_inside_chart(chart, state)
    components = state_components(state)
    all(value -> in_chart(chart, value), eigvals(components.right_state)) &&
        all(value -> in_chart(chart, value), eigvals(components.left_state))
end

function state_contour_separation(chart, state)
    components = state_components(state)
    function separation(input_state)
        count = size(input_state, 1)
        identity_state = Matrix{ComplexF64}(I, count, count)
        minimum(
            minimum(svdvals((z .* identity_state .- input_state) ./ chart.radius))
            for z in chart.nodes
        )
    end
    right_sampled = Float64(separation(components.right_state))
    left_sampled = Float64(separation(components.left_state))
    cover_radius = Float64(chart_node_cover_radius(chart))
    right = max(0.0, right_sampled - cover_radius)
    left = max(0.0, left_sampled - cover_radius)
    (
        right=right,
        left=left,
        minimum=min(right, left),
        right_sampled=right_sampled,
        left_sampled=left_sampled,
        cover_radius=cover_radius,
    )
end

function state_modal_output(
    state::CommonStateRealization;
    condition_limit=1 / sqrt(eps(Float64)),
)
    decomposition = eigen(state.state)
    condition_number = Float64(cond(decomposition.vectors))
    available = isfinite(condition_number) && condition_number <= condition_limit
    if !available
        return (
            available=false,
            values=ComplexF64.(decomposition.values),
            left_values=ComplexF64.(decomposition.values),
            right=zeros(ComplexF64, size(state.right, 1), 0),
            left=zeros(ComplexF64, size(state.left, 1), 0),
            condition=condition_number,
            match_errors=zeros(Float64, length(decomposition.values)),
            reason=:ill_conditioned_state,
        )
    end
    (
        available=true,
        values=ComplexF64.(decomposition.values),
        left_values=ComplexF64.(decomposition.values),
        right=Matrix{ComplexF64}(state.right * decomposition.vectors),
        left=Matrix{ComplexF64}(state.left * adjoint(inv(decomposition.vectors))),
        condition=condition_number,
        match_errors=zeros(Float64, length(decomposition.values)),
        reason=:available,
    )
end

function state_modal_output(
    state::IndependentStateRealization;
    condition_limit=1 / sqrt(eps(Float64)),
    match_tolerance=sqrt(eps(Float64)),
)
    right_decomposition = eigen(state.right_state)
    left_decomposition = eigen(state.left_state)
    right_condition = Float64(cond(right_decomposition.vectors))
    left_condition = Float64(cond(left_decomposition.vectors))
    assignment, match_errors = bottleneck_match(
        right_decomposition.values,
        left_decomposition.values,
    )
    condition_number = max(right_condition, left_condition)
    match_scale = max(maximum(abs, right_decomposition.values; init=0.0), 1.0)
    available = isfinite(condition_number) && condition_number <= condition_limit &&
        maximum(match_errors; init=0.0) <= match_tolerance * match_scale
    if !available
        return (
            available=false,
            values=ComplexF64.(right_decomposition.values),
            left_values=ComplexF64.(left_decomposition.values[assignment]),
            right=zeros(ComplexF64, size(state.right, 1), 0),
            left=zeros(ComplexF64, size(state.left, 1), 0),
            condition=condition_number,
            match_errors=match_errors,
            reason=condition_number > condition_limit ? :ill_conditioned_state : :unmatched_states,
        )
    end
    (
        available=true,
        values=ComplexF64.(right_decomposition.values),
        left_values=ComplexF64.(left_decomposition.values[assignment]),
        right=Matrix{ComplexF64}(state.right * right_decomposition.vectors),
        left=Matrix{ComplexF64}(
            state.left * adjoint(inv(left_decomposition.vectors))[:, assignment],
        ),
        condition=condition_number,
        match_errors=match_errors,
        reason=:available,
    )
end

function state_error_data(state_error, state)
    value = state_error(state)
    if value isa Number
        error = Float64(value)
        return (right=error, left=error, maximum=error)
    end
    (
        right=Float64(value.right),
        left=Float64(value.left),
        maximum=Float64(value.maximum),
    )
end

function state_overlap_ratio(state)
    state isa CommonStateRealization || return NaN
    singulars = state.overlap_singular_values
    isempty(singulars) ? NaN : singulars[end] / singulars[1]
end

function state_iteration_record(
    iteration,
    state,
    errors,
    selection,
    right_residual_rank,
    left_residual_rank,
    right_probe_width,
    left_probe_width,
    chart,
    compression_rejected,
)
    components = state_components(state)
    boundary_margin = minimum(vcat(
        chart_boundary_margin.(Ref(chart), eigvals(components.right_state)),
        chart_boundary_margin.(Ref(chart), eigvals(components.left_state)),
    ))
    contour_separation = state_contour_separation(chart, state)
    StateIterationRecord(
        iteration,
        errors.maximum,
        errors.right,
        errors.left,
        state_rank(state),
        state_representation(state),
        selection,
        right_residual_rank,
        left_residual_rank,
        right_probe_width,
        left_probe_width,
        state_overlap_ratio(state),
        Float64(boundary_margin),
        contour_separation.right,
        contour_separation.left,
        compression_rejected,
    )
end
