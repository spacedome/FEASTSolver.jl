function spectral_clusters(values; tolerance)
    tolerance >= 0 || throw(ArgumentError("cluster tolerance must be nonnegative"))
    count = length(values)
    count == 0 && return Vector{Int}[]
    visited = falses(count)
    clusters = Vector{Int}[]
    for seed in eachindex(values)
        visited[seed] && continue
        visited[seed] = true
        stack = Int[seed]
        cluster = Int[]
        while !isempty(stack)
            i = pop!(stack)
            push!(cluster, i)
            for j in eachindex(values)
                visited[j] && continue
                scale = max(1.0, abs(values[i]), abs(values[j]))
                if abs(values[i] - values[j]) <= tolerance * scale
                    visited[j] = true
                    push!(stack, j)
                end
            end
        end
        push!(clusters, cluster)
    end
    clusters
end

function maximum_cluster_size(values; tolerance)
    clusters = spectral_clusters(values; tolerance=tolerance)
    isempty(clusters) ? 0 : maximum(length, clusters)
end

function moment_tangent(
    values,
    moment_count,
    requested_width;
    cluster_tolerance,
    side,
)
    requested_width === nothing && return nothing
    requested_width > 0 || throw(ArgumentError("tangential moment width must be positive"))
    moment_count > 0 || throw(ArgumentError("moment_count must be positive"))
    side in (:right, :left) || throw(ArgumentError("side must be :right or :left"))
    count = length(values)
    count == 0 && return zeros(ComplexF64, 0, 0)
    width = min(requested_width, count)
    capacity_width = cld(count, moment_count)
    cluster_width = maximum_cluster_size(values; tolerance=cluster_tolerance)
    required_width = max(capacity_width, cluster_width)
    width >= required_width || throw(ArgumentError(
        "tangential width $width is too small: need at least $required_width " *
        "for state count $count, moment depth $moment_count, and detected clusters",
    ))
    width == count && return Matrix{ComplexF64}(I, count, count)
    clusters = sort(
        spectral_clusters(values; tolerance=cluster_tolerance);
        by=length,
        rev=true,
    )
    loads = zeros(Int, width)
    columns = zeros(Int, count)
    for cluster in clusters
        available = sortperm(loads)
        for (member, column) in zip(cluster, available)
            columns[member] = column
            loads[column] += 1
        end
    end
    maximum(loads) <= moment_count || error("internal tangential partition exceeded moment capacity")
    direction = side === :right ? 1 : -1
    sparse(
        1:count,
        columns,
        ComplexF64[
            cis(direction * 2pi * (row - 1) / count) / sqrt(loads[columns[row]])
            for row in 1:count
        ],
        count,
        width,
    )
end
