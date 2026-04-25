# Timing and diagnostic state for dense distributed FEAST.

struct DenseDistributedFeastIterationStats
    iteration::Int
    eigenvalues_inside::Int
    converged_inside::Int
    max_res_inside::Float64
    total_ns::UInt64
    qr_ns::UInt64
    rayleigh_ritz_ns::UInt64
    residual_ns::UInt64
    shared_copy_ns::UInt64
    worker_step_ns::UInt64
    reduce_ns::UInt64
end

mutable struct DenseDistributedFeastStats
    setup_prepare_ns::UInt64
    setup_shared_ns::UInt64
    setup_worker_ns::UInt64
    solve_total_ns::UInt64
    iterations::Int
    qr_ns::UInt64
    rayleigh_ritz_ns::UInt64
    residual_ns::UInt64
    shared_copy_ns::UInt64
    worker_step_ns::UInt64
    reduce_ns::UInt64
    iteration_log::Vector{DenseDistributedFeastIterationStats}
end

function DenseDistributedFeastStats()
    DenseDistributedFeastStats(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, DenseDistributedFeastIterationStats[])
end

function reset!(stats::DenseDistributedFeastStats)
    stats.setup_prepare_ns = 0
    stats.setup_shared_ns = 0
    stats.setup_worker_ns = 0
    stats.solve_total_ns = 0
    stats.iterations = 0
    stats.qr_ns = 0
    stats.rayleigh_ritz_ns = 0
    stats.residual_ns = 0
    stats.shared_copy_ns = 0
    stats.worker_step_ns = 0
    stats.reduce_ns = 0
    empty!(stats.iteration_log)
    stats
end

_add_elapsed!(::Nothing, field::Symbol, start_ns::UInt64) = nothing

function _add_elapsed!(stats::DenseDistributedFeastStats, field::Symbol, start_ns::UInt64)
    setfield!(stats, field, getfield(stats, field) + (time_ns() - start_ns))
    nothing
end

_add_ns!(::Nothing, field::Symbol, elapsed_ns::UInt64) = nothing

function _add_ns!(stats::DenseDistributedFeastStats, field::Symbol, elapsed_ns::UInt64)
    setfield!(stats, field, getfield(stats, field) + elapsed_ns)
    nothing
end

function _seconds(ns::UInt64)
    ns / 1.0e9
end

function Base.show(io::IO, stats::DenseDistributedFeastStats)
    print(
        io,
        "DenseDistributedFeastStats(",
        "setup_prepare=", _seconds(stats.setup_prepare_ns), "s, ",
        "setup_shared=", _seconds(stats.setup_shared_ns), "s, ",
        "setup_worker=", _seconds(stats.setup_worker_ns), "s, ",
        "solve_total=", _seconds(stats.solve_total_ns), "s, ",
        "iterations=", stats.iterations, ", ",
        "qr=", _seconds(stats.qr_ns), "s, ",
        "rayleigh_ritz=", _seconds(stats.rayleigh_ritz_ns), "s, ",
        "residual=", _seconds(stats.residual_ns), "s, ",
        "shared_copy=", _seconds(stats.shared_copy_ns), "s, ",
        "worker_step=", _seconds(stats.worker_step_ns), "s, ",
        "reduce=", _seconds(stats.reduce_ns), "s, ",
        "logged_iterations=", length(stats.iteration_log), ")",
    )
end

function _record_distributed_feast_iteration!(
    ::Nothing,
    nit,
    res,
    inside,
    max_res_inside,
    contour_nonempty,
    iter_start_ns,
    qr_ns,
    rayleigh_ritz_ns,
    residual_ns,
    shared_copy_ns,
    worker_step_ns,
    reduce_ns,
    debug,
)
    nothing
end

function _record_distributed_feast_iteration!(
    stats::DenseDistributedFeastStats,
    nit,
    res,
    inside,
    max_res_inside,
    contour_nonempty,
    iter_start_ns,
    qr_ns,
    rayleigh_ritz_ns,
    residual_ns,
    shared_copy_ns,
    worker_step_ns,
    reduce_ns,
    debug,
)
    push!(
        stats.iteration_log,
        DenseDistributedFeastIterationStats(
            nit,
            count(inside),
            _count_converged_inside(res, inside, 1e-5),
            contour_nonempty ? Float64(max_res_inside) : NaN,
            time_ns() - iter_start_ns,
            qr_ns,
            rayleigh_ritz_ns,
            residual_ns,
            shared_copy_ns,
            worker_step_ns,
            reduce_ns,
        ),
    )
    if debug
        _iter_timing_debug_print(stats.iteration_log[end])
    end
    nothing
end

function _iter_timing_debug_print(stats::DenseDistributedFeastIterationStats)
    println(
        "timing:\t",
        "total=", _seconds(stats.total_ns), "s\t",
        "qr=", _seconds(stats.qr_ns), "s\t",
        "ritz=", _seconds(stats.rayleigh_ritz_ns), "s\t",
        "res=", _seconds(stats.residual_ns), "s\t",
        "copy=", _seconds(stats.shared_copy_ns), "s\t",
        "worker=", _seconds(stats.worker_step_ns), "s\t",
        "reduce=", _seconds(stats.reduce_ns), "s",
    )
    nothing
end
