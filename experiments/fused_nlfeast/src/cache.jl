abstract type AbstractContourChart end

struct CircularChart <: AbstractContourChart
    center::ComplexF64
    radius::Float64
    nodes::Vector{ComplexF64}
    weights::Vector{ComplexF64}
end

struct RectangularChart <: AbstractContourChart
    center::ComplexF64
    radius::Float64
    half_width::Float64
    half_height::Float64
    nodes::Vector{ComplexF64}
    weights::Vector{ComplexF64}
end

function CircularChart(center, radius, node_count::Integer)
    radius > 0 || throw(ArgumentError("chart radius must be positive"))
    node_count >= 4 || throw(ArgumentError("a circular chart needs at least four nodes"))
    center = ComplexF64(center)
    radius = Float64(radius)
    angles = ((2 .* (1:node_count) .- 1) .* pi) ./ node_count
    coordinates = ComplexF64[cis(angle) for angle in angles]
    nodes = center .+ radius .* coordinates
    weights = (radius / node_count) .* coordinates
    CircularChart(center, radius, nodes, weights)
end

function RectangularChart(center, half_width, half_height, node_count::Integer)
    half_width > 0 || throw(ArgumentError("rectangle half-width must be positive"))
    half_height > 0 || throw(ArgumentError("rectangle half-height must be positive"))
    node_count >= 8 || throw(ArgumentError("a rectangular chart needs at least eight nodes"))
    iszero(node_count % 4) || throw(ArgumentError("rectangular node count must be divisible by four"))
    center = ComplexF64(center)
    half_width = Float64(half_width)
    half_height = Float64(half_height)
    lower_left = center - half_width - im * half_height
    lower_right = center + half_width - im * half_height
    upper_right = center + half_width + im * half_height
    upper_left = center - half_width + im * half_height
    corners = (lower_left, lower_right, upper_right, upper_left, lower_left)
    nodes_per_edge = node_count ÷ 4
    edge_nodes, edge_weights = gausslegendre(nodes_per_edge)
    nodes = ComplexF64[]
    weights = ComplexF64[]
    for edge in 1:4
        midpoint = (corners[edge] + corners[edge + 1]) / 2
        half_step = (corners[edge + 1] - corners[edge]) / 2
        for (node, weight) in zip(edge_nodes, edge_weights)
            push!(nodes, midpoint + half_step * node)
            push!(weights, half_step * weight / (2pi * im))
        end
    end
    RectangularChart(
        center,
        max(half_width, half_height),
        half_width,
        half_height,
        nodes,
        weights,
    )
end

chart_coordinate(chart::AbstractContourChart, z) = (z - chart.center) / chart.radius
chart_inward_score(chart::CircularChart, z) = abs(chart_coordinate(chart, z))
function chart_inward_score(chart::RectangularChart, z)
    offset = z - chart.center
    max(abs(real(offset)) / chart.half_width, abs(imag(offset)) / chart.half_height)
end
chart_boundary_margin(chart::AbstractContourChart, z) = 1 - chart_inward_score(chart, z)
chart_node_cover_radius(chart::CircularChart) = 2sin(pi / (2length(chart.nodes)))
function chart_node_cover_radius(chart::RectangularChart)
    lower_left = chart.center - chart.half_width - im * chart.half_height
    lower_right = chart.center + chart.half_width - im * chart.half_height
    upper_right = chart.center + chart.half_width + im * chart.half_height
    upper_left = chart.center - chart.half_width + im * chart.half_height
    corners = (lower_left, lower_right, upper_right, upper_left, lower_left)
    nodes_per_edge = length(chart.nodes) ÷ 4
    covering_radius = 0.0
    for edge in 1:4
        indices = ((edge - 1) * nodes_per_edge + 1):(edge * nodes_per_edge)
        edge_nodes = view(chart.nodes, indices)
        covering_radius = max(
            covering_radius,
            abs(first(edge_nodes) - corners[edge]),
            abs(corners[edge + 1] - last(edge_nodes)),
        )
        for index in 1:(length(edge_nodes) - 1)
            covering_radius = max(
                covering_radius,
                abs(edge_nodes[index + 1] - edge_nodes[index]) / 2,
            )
        end
    end
    covering_radius / chart.radius
end

function chart_boundary_segments(chart::CircularChart)
    node_count = length(chart.nodes)
    segment_length = 2pi * chart.radius / node_count
    [
        (
            start=chart.nodes[index],
            stop=chart.nodes[mod1(index + 1, node_count)],
            length=segment_length,
        )
        for index in eachindex(chart.nodes)
    ]
end


function chart_boundary_segments(chart::RectangularChart)
    lower_left = chart.center - chart.half_width - im * chart.half_height
    lower_right = chart.center + chart.half_width - im * chart.half_height
    upper_right = chart.center + chart.half_width + im * chart.half_height
    upper_left = chart.center - chart.half_width + im * chart.half_height
    corners = (lower_left, lower_right, upper_right, upper_left, lower_left)
    nodes_per_edge = length(chart.nodes) ÷ 4
    segments = NamedTuple[]
    for edge in 1:4
        indices = ((edge - 1) * nodes_per_edge + 1):(edge * nodes_per_edge)
        points = vcat(corners[edge], chart.nodes[indices], corners[edge + 1])
        for index in 1:(length(points) - 1)
            push!(segments, (
                start=points[index],
                stop=points[index + 1],
                length=Float64(abs(points[index + 1] - points[index])),
            ))
        end
    end
    segments
end
in_chart(chart::CircularChart, z; tolerance=100eps(Float64)) = abs(chart_coordinate(chart, z)) <= 1 + tolerance
function in_chart(chart::RectangularChart, z; tolerance=100eps(Float64))
    offset = z - chart.center
    abs(real(offset)) <= chart.half_width + tolerance * chart.radius &&
        abs(imag(offset)) <= chart.half_height + tolerance * chart.radius
end

refine_chart(chart::CircularChart, node_count) = CircularChart(chart.center, chart.radius, node_count)
refine_chart(chart::RectangularChart, node_count) = RectangularChart(
    chart.center,
    chart.half_width,
    chart.half_height,
    node_count,
)

struct ContourSolveResult{M}
    solution::M
    relative_residual::Float64
    converged::Bool
    iterations::Int
end

function ContourSolveResult(
    solution;
    relative_residual,
    converged=true,
    iterations=0,
)
    residual = Float64(relative_residual)
    residual >= 0 && isfinite(residual) || throw(ArgumentError(
        "reported solve residual must be finite and nonnegative",
    ))
    iteration_count = Int(iterations)
    iteration_count >= 0 || throw(ArgumentError("solve iterations must be nonnegative"))
    ContourSolveResult(solution, residual, Bool(converged), iteration_count)
end

struct ContourSolveDiagnostic
    reported::Bool
    relative_residual::Float64
    converged::Bool
    iterations::Int
end

struct SampledProbeBlock
    role::Symbol
    probe::Matrix{ComplexF64}
    responses::Vector{Matrix{ComplexF64}}
    diagnostics::Vector{ContourSolveDiagnostic}
end

struct ContourSolveFailureError <: Exception
    side::Symbol
    node::ComplexF64
    reason::Symbol
    relative_residual::Float64
end

function Base.showerror(io::IO, error::ContourSolveFailureError)
    print(io, "contour solve failed on the ", error.side, " side at ", error.node)
    if error.reason === :unreported
        print(io, ": solve diagnostics were required but not reported")
    elseif error.reason === :not_converged
        print(io, ": the solve reported nonconvergence")
    elseif error.reason === :residual_too_large
        print(io, ": reported relative residual ", error.relative_residual, " exceeds the limit")
    else
        print(io, ": ", error.reason)
    end
end

mutable struct ContourSampleCache{C,R,L}
    chart::C
    right_solve::R
    left_solve::L
    right_blocks::Dict{Symbol,SampledProbeBlock}
    left_blocks::Dict{Symbol,SampledProbeBlock}
    right_diagnostics::Vector{ContourSolveDiagnostic}
    left_diagnostics::Vector{ContourSolveDiagnostic}
    right_solves::Int
    left_solves::Int
end

function ContourSampleCache(chart::AbstractContourChart, right_solve, left_solve)
    ContourSampleCache(
        chart,
        right_solve,
        left_solve,
        Dict{Symbol,SampledProbeBlock}(),
        Dict{Symbol,SampledProbeBlock}(),
        ContourSolveDiagnostic[],
        ContourSolveDiagnostic[],
        0,
        0,
    )
end

function sampled_probe_block(
    solve,
    chart,
    probe,
    role,
    side;
    require_diagnostics=false,
    maximum_relative_residual=Inf,
)
    maximum_relative_residual >= 0 || throw(ArgumentError(
        "maximum_relative_residual must be nonnegative",
    ))
    probe = Matrix{ComplexF64}(probe)
    responses = Matrix{ComplexF64}[]
    diagnostics = ContourSolveDiagnostic[]
    for z in chart.nodes
        if size(probe, 2) == 0
            push!(responses, zeros(ComplexF64, size(probe)))
            push!(diagnostics, ContourSolveDiagnostic(true, 0.0, true, 0))
            continue
        end
        output = solve(z, probe)
        if output isa ContourSolveResult
            diagnostic = ContourSolveDiagnostic(
                true,
                output.relative_residual,
                output.converged,
                output.iterations,
            )
            output.converged || throw(ContourSolveFailureError(
                side,
                z,
                :not_converged,
                output.relative_residual,
            ))
            if output.relative_residual > maximum_relative_residual
                throw(ContourSolveFailureError(
                    side,
                    z,
                    :residual_too_large,
                    output.relative_residual,
                ))
            end
            response = Matrix{ComplexF64}(output.solution)
            push!(diagnostics, diagnostic)
        else
            require_diagnostics && throw(ContourSolveFailureError(
                side,
                z,
                :unreported,
                NaN,
            ))
            response = Matrix{ComplexF64}(output)
            push!(diagnostics, ContourSolveDiagnostic(false, NaN, true, 0))
        end
        size(response) == size(probe) || throw(DimensionMismatch(
            "a contour solve must preserve the right-hand-side shape",
        ))
        push!(responses, response)
    end
    SampledProbeBlock(role, probe, responses, diagnostics)
end

function add_right_probe!(
    cache::ContourSampleCache,
    id::Symbol,
    probe;
    role=:realization,
    require_diagnostics=false,
    maximum_relative_residual=Inf,
)
    haskey(cache.right_blocks, id) && throw(ArgumentError("right probe id $id already exists"))
    block = sampled_probe_block(
        cache.right_solve,
        cache.chart,
        probe,
        role,
        :right;
        require_diagnostics=require_diagnostics,
        maximum_relative_residual=maximum_relative_residual,
    )
    cache.right_blocks[id] = block
    append!(cache.right_diagnostics, block.diagnostics)
    size(block.probe, 2) > 0 && (cache.right_solves += length(cache.chart.nodes))
    block
end

function add_left_probe!(
    cache::ContourSampleCache,
    id::Symbol,
    probe;
    role=:realization,
    require_diagnostics=false,
    maximum_relative_residual=Inf,
)
    haskey(cache.left_blocks, id) && throw(ArgumentError("left probe id $id already exists"))
    block = sampled_probe_block(
        cache.left_solve,
        cache.chart,
        probe,
        role,
        :left;
        require_diagnostics=require_diagnostics,
        maximum_relative_residual=maximum_relative_residual,
    )
    cache.left_blocks[id] = block
    append!(cache.left_diagnostics, block.diagnostics)
    size(block.probe, 2) > 0 && (cache.left_solves += length(cache.chart.nodes))
    block
end

function drop_right_probe!(cache::ContourSampleCache, id::Symbol)
    block = pop!(cache.right_blocks, id)
    block
end

function drop_left_probe!(cache::ContourSampleCache, id::Symbol)
    block = pop!(cache.left_blocks, id)
    block
end

function release_right_responses!(cache::ContourSampleCache, id::Symbol)
    block = cache.right_blocks[id]
    cache.right_blocks[id] = SampledProbeBlock(
        block.role,
        block.probe,
        Matrix{ComplexF64}[],
        block.diagnostics,
    )
    block.responses
end

function release_left_responses!(cache::ContourSampleCache, id::Symbol)
    block = cache.left_blocks[id]
    cache.left_blocks[id] = SampledProbeBlock(
        block.role,
        block.probe,
        Matrix{ComplexF64}[],
        block.diagnostics,
    )
    block.responses
end

solve_counts(cache::ContourSampleCache) = (right=cache.right_solves, left=cache.left_solves)

function solve_diagnostic_summary(cache::ContourSampleCache)
    diagnostics = vcat(cache.right_diagnostics, cache.left_diagnostics)
    reported = Base.count(diagnostic -> diagnostic.reported, diagnostics)
    residuals = [
        diagnostic.relative_residual
        for diagnostic in diagnostics if diagnostic.reported
    ]
    (
        total=length(diagnostics),
        reported=reported,
        unreported=length(diagnostics) - reported,
        maximum_relative_residual=isempty(residuals) ? NaN : maximum(residuals),
        maximum_iterations=maximum(
            (diagnostic.iterations for diagnostic in diagnostics if diagnostic.reported);
            init=0,
        ),
    )
end
