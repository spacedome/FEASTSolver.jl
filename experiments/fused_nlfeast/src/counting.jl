struct ArgumentCountEstimate
    count::Int
    value::ComplexF64
    coarse_value::ComplexF64
    integrality_error::Float64
    refinement_error::Float64
    stable::Bool
    resolution_certified::Bool
    reliable::Bool
    nodes::Int
    refined_nodes::Int
end

struct MeromorphicCountEstimate
    index::ArgumentCountEstimate
    pole_multiplicity::Int
    pole_multiplicity_certified::Bool
    count::Int
    stable::Bool
    resolution_certified::Bool
    reliable::Bool
end

function argument_count_value(logarithmic_derivative, chart::AbstractContourChart)
    sum(
        weight * ComplexF64(logarithmic_derivative(z))
        for (z, weight) in zip(chart.nodes, chart.weights)
    )
end

function argument_principle_count(
    logarithmic_derivative,
    chart::AbstractContourChart;
    refinement_factor=2,
    integrality_tolerance=1e-3,
    refinement_tolerance=1e-3,
    resolution_certified=false,
)
    refinement_factor >= 2 || throw(ArgumentError("count refinement factor must be at least two"))
    integrality_tolerance >= 0 || throw(ArgumentError("count integrality tolerance must be nonnegative"))
    refinement_tolerance >= 0 || throw(ArgumentError("count refinement tolerance must be nonnegative"))
    coarse_value = ComplexF64(argument_count_value(logarithmic_derivative, chart))
    refined_nodes = refinement_factor * length(chart.nodes)
    refined_chart = refine_chart(chart, refined_nodes)
    value = ComplexF64(argument_count_value(logarithmic_derivative, refined_chart))
    count = round(Int, real(value))
    integrality_error = Float64(abs(value - count))
    refinement_error = Float64(abs(value - coarse_value))
    stable = count >= 0 && isfinite(integrality_error) && isfinite(refinement_error) &&
        integrality_error <= integrality_tolerance &&
        refinement_error <= refinement_tolerance
    reliable = resolution_certified && stable
    ArgumentCountEstimate(
        count,
        value,
        coarse_value,
        integrality_error,
        refinement_error,
        stable,
        resolution_certified,
        reliable,
        length(chart.nodes),
        refined_nodes,
    )
end

function determinant_winding_value(matrix_function, chart::AbstractContourChart)
    signs = ComplexF64[]
    for z in chart.nodes
        _, sign = logabsdet(matrix_function(z))
        iszero(sign) && error("operator determinant vanishes at a contour node")
        isfinite(sign) || error("operator determinant phase is nonfinite at a contour node")
        push!(signs, ComplexF64(sign))
    end
    sum(
        angle(signs[mod1(index + 1, length(signs))] / signs[index])
        for index in eachindex(signs)
    ) / (2pi)
end

function certified_determinant_winding_value(
    matrix_function,
    derivative_bound,
    chart::AbstractContourChart,
)
    samples = Dict{ComplexF64,NamedTuple}()
    function sample(point)
        get!(samples, ComplexF64(point)) do
            matrix = Matrix{ComplexF64}(matrix_function(point))
            size(matrix, 1) == size(matrix, 2) || throw(DimensionMismatch(
                "determinant winding requires square matrices",
            ))
            _, sign = logabsdet(matrix)
            singulars = svdvals(matrix)
            (
                sign=ComplexF64(sign),
                minimum_singular=isempty(singulars) ? Inf : Float64(singulars[end]),
                dimension=size(matrix, 1),
            )
        end
    end

    winding = 0.0
    certified = true
    minimum_boundary_singular = Inf
    maximum_phase_change_bound = 0.0
    for segment in chart_boundary_segments(chart)
        first_sample = sample(segment.start)
        second_sample = sample(segment.stop)
        first_sample.dimension == second_sample.dimension || throw(DimensionMismatch(
            "matrix dimension changes along the contour",
        ))
        bound = Float64(derivative_bound(segment.start, segment.stop))
        bound >= 0 && isfinite(bound) || throw(ArgumentError(
            "the derivative bound must be finite and nonnegative",
        ))
        lower_singular = min(
            first_sample.minimum_singular,
            second_sample.minimum_singular,
        ) - bound * segment.length / 2
        minimum_boundary_singular = min(minimum_boundary_singular, lower_singular)
        if lower_singular <= 0 || iszero(first_sample.sign) || iszero(second_sample.sign)
            certified = false
        else
            phase_bound = first_sample.dimension * bound * segment.length / lower_singular
            maximum_phase_change_bound = max(maximum_phase_change_bound, phase_bound)
            phase_bound < pi || (certified = false)
        end
        if iszero(first_sample.sign) || iszero(second_sample.sign)
            winding = NaN
        elseif isfinite(winding)
            winding += angle(second_sample.sign / first_sample.sign) / (2pi)
        end
    end
    (
        value=ComplexF64(winding),
        certified=certified,
        minimum_boundary_singular=Float64(minimum_boundary_singular),
        maximum_phase_change_bound=Float64(maximum_phase_change_bound),
    )
end

function determinant_winding_count(
    matrix_function,
    chart::AbstractContourChart;
    refinement_factor=2,
    integrality_tolerance=1e-8,
    refinement_tolerance=1e-8,
    phase_resolution_certified=false,
    derivative_bound=nothing,
)
    refinement_factor >= 2 || throw(ArgumentError("count refinement factor must be at least two"))
    integrality_tolerance >= 0 || throw(ArgumentError("integrality tolerance must be nonnegative"))
    refinement_tolerance >= 0 || throw(ArgumentError("refinement tolerance must be nonnegative"))
    coarse_data = derivative_bound === nothing ? nothing :
        certified_determinant_winding_value(matrix_function, derivative_bound, chart)
    coarse_value = coarse_data === nothing ?
        ComplexF64(determinant_winding_value(matrix_function, chart)) : coarse_data.value
    refined_nodes = refinement_factor * length(chart.nodes)
    refined_chart = refine_chart(chart, refined_nodes)
    refined_data = derivative_bound === nothing ? nothing :
        certified_determinant_winding_value(
            matrix_function,
            derivative_bound,
            refined_chart,
        )
    value = refined_data === nothing ?
        ComplexF64(determinant_winding_value(matrix_function, refined_chart)) :
        refined_data.value
    count = isfinite(value) ? round(Int, real(value)) : 0
    integrality_error = isfinite(value) ? Float64(abs(value - count)) : Inf
    refinement_error = isfinite(value) && isfinite(coarse_value) ?
        Float64(abs(value - coarse_value)) : Inf
    stable = count >= 0 &&
        isfinite(integrality_error) && isfinite(refinement_error) &&
        integrality_error <= integrality_tolerance &&
        refinement_error <= refinement_tolerance
    derived_certification = coarse_data !== nothing && coarse_data.certified &&
        refined_data.certified
    resolution_certified = phase_resolution_certified || derived_certification
    reliable = resolution_certified && stable
    ArgumentCountEstimate(
        count,
        value,
        coarse_value,
        integrality_error,
        refinement_error,
        stable,
        resolution_certified,
        reliable,
        length(chart.nodes),
        refined_nodes,
    )
end

function meromorphic_eigenvalue_count(
    logarithmic_derivative,
    chart::AbstractContourChart;
    pole_multiplicity,
    pole_multiplicity_certified=false,
    kwargs...,
)
    pole_count = Int(pole_multiplicity)
    pole_count >= 0 || throw(ArgumentError("pole multiplicity must be nonnegative"))
    index = argument_principle_count(logarithmic_derivative, chart; kwargs...)
    eigenvalue_count = index.count + pole_count
    stable = index.stable && eigenvalue_count >= 0
    reliable = stable && index.reliable && pole_multiplicity_certified
    MeromorphicCountEstimate(
        index,
        pole_count,
        pole_multiplicity_certified,
        eigenvalue_count,
        stable,
        index.resolution_certified,
        reliable,
    )
end
