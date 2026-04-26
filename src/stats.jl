"""
    DenseFeastIterationStats

Per-iteration diagnostics recorded by dense serial FEAST when a
`DenseFeastStats` object is passed as `stats=`.
"""
struct DenseFeastIterationStats
    variant::Symbol
    iteration::Int
    eigenvalues_inside::Int
    converged_inside::Int
    max_res_inside::Float64
    total_ns::UInt64
    orthogonalization_ns::UInt64
    rayleigh_ritz_ns::UInt64
    residual_ns::UInt64
    filter_ns::UInt64
end

"""
    DenseFeastStats()

Lightweight diagnostics accumulator for dense serial FEAST variants. Pass an
instance as `stats=` to `feast!`, `gen_feast!`, or `dual_gen_feast!` to record
iteration counts, convergence counts, maximum residuals inside the contour, and
coarse phase timings.
"""
mutable struct DenseFeastStats
    solve_total_ns::UInt64
    iterations::Int
    orthogonalization_ns::UInt64
    rayleigh_ritz_ns::UInt64
    residual_ns::UInt64
    filter_ns::UInt64
    stored_factor_count::Int
    stored_factor_bytes::Int
    iteration_log::Vector{DenseFeastIterationStats}
end

function DenseFeastStats()
    DenseFeastStats(0, 0, 0, 0, 0, 0, 0, 0, DenseFeastIterationStats[])
end

function reset!(stats::DenseFeastStats)
    stats.solve_total_ns = 0
    stats.iterations = 0
    stats.orthogonalization_ns = 0
    stats.rayleigh_ritz_ns = 0
    stats.residual_ns = 0
    stats.filter_ns = 0
    stats.stored_factor_count = 0
    stats.stored_factor_bytes = 0
    empty!(stats.iteration_log)
    stats
end

_dense_stats_seconds(ns::UInt64) = ns / 1.0e9

function Base.show(io::IO, stats::DenseFeastStats)
    print(
        io,
        "DenseFeastStats(",
        "solve_total=", _dense_stats_seconds(stats.solve_total_ns), "s, ",
        "iterations=", stats.iterations, ", ",
        "orthogonalization=", _dense_stats_seconds(stats.orthogonalization_ns), "s, ",
        "rayleigh_ritz=", _dense_stats_seconds(stats.rayleigh_ritz_ns), "s, ",
        "residual=", _dense_stats_seconds(stats.residual_ns), "s, ",
        "filter=", _dense_stats_seconds(stats.filter_ns), "s, ",
        "stored_factor_count=", stats.stored_factor_count, ", ",
        "stored_factor_bytes=", stats.stored_factor_bytes, ", ",
        "logged_iterations=", length(stats.iteration_log), ")",
    )
end

_record_stored_factor_memory!(::Nothing, factors) = nothing

function _record_stored_factor_memory!(stats::DenseFeastStats, factors)
    stats.stored_factor_count = length(factors)
    stats.stored_factor_bytes = sum(Base.summarysize, factors)
    nothing
end

function _record_dense_feast_iteration!(
    stats::Nothing,
    variant::Symbol,
    nit::Integer,
    res::AbstractVector,
    inside::AbstractVector{Bool},
    max_res_inside,
    contour_nonempty::Bool,
    iter_start_ns::UInt64,
    orthogonalization_ns::UInt64,
    rayleigh_ritz_ns::UInt64,
    residual_ns::UInt64,
    filter_ns::UInt64,
    debug::Bool,
)
    nothing
end

function _record_dense_feast_iteration!(
    stats::DenseFeastStats,
    variant::Symbol,
    nit::Integer,
    res::AbstractVector,
    inside::AbstractVector{Bool},
    max_res_inside,
    contour_nonempty::Bool,
    iter_start_ns::UInt64,
    orthogonalization_ns::UInt64,
    rayleigh_ritz_ns::UInt64,
    residual_ns::UInt64,
    filter_ns::UInt64,
    debug::Bool,
)
    stats.iterations += 1
    stats.orthogonalization_ns += orthogonalization_ns
    stats.rayleigh_ritz_ns += rayleigh_ritz_ns
    stats.residual_ns += residual_ns
    stats.filter_ns += filter_ns
    push!(
        stats.iteration_log,
        DenseFeastIterationStats(
            variant,
            Int(nit),
            count(inside),
            _count_converged_inside(res, inside, 1e-5),
            contour_nonempty ? Float64(max_res_inside) : NaN,
            time_ns() - iter_start_ns,
            orthogonalization_ns,
            rayleigh_ritz_ns,
            residual_ns,
            filter_ns,
        ),
    )
    if debug
        _iter_timing_debug_print(stats.iteration_log[end])
    end
    nothing
end

function _count_converged_inside(res::AbstractVector, inside::AbstractVector{Bool}, spurious)
    count = 0
    @inbounds for i in eachindex(res, inside)
        if inside[i] && res[i] < spurious
            count += 1
        end
    end
    count
end

function _iter_timing_debug_print(stats::DenseFeastIterationStats)
    println(
        "timing:\t",
        "total=", _dense_stats_seconds(stats.total_ns), "s\t",
        "orth=", _dense_stats_seconds(stats.orthogonalization_ns), "s\t",
        "ritz=", _dense_stats_seconds(stats.rayleigh_ritz_ns), "s\t",
        "res=", _dense_stats_seconds(stats.residual_ns), "s\t",
        "filter=", _dense_stats_seconds(stats.filter_ns), "s",
    )
    nothing
end
