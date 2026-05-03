using BenchmarkTools
using Distributed

const REPO_ROOT = normpath(joinpath(@__DIR__, ".."))
include(joinpath(REPO_ROOT, "experiments", "moment_rii", "run.jl"))

const ENV_DEFAULTS = Dict(
    "FEAST_MOMENT_BENCH_PROBLEM" => "schrodinger",
    "FEAST_MOMENT_BENCH_WORKERS" => "2",
    "FEAST_MOMENT_BENCH_SAMPLES" => "1",
    "FEAST_MOMENT_BENCH_SECONDS" => "3600",
)

env(name) = get(ENV, name, ENV_DEFAULTS[name])

function trial_summary(trial)
    min_est = BenchmarkTools.minimum(trial)
    med_est = BenchmarkTools.median(trial)
    (
        samples=length(trial),
        min_s=min_est.time / 1e9,
        median_s=med_est.time / 1e9,
        memory=min_est.memory,
        allocs=min_est.allocs,
    )
end

function print_trial(label, trial)
    s = trial_summary(trial)
    println(
        label,
        ",samples=", s.samples,
        ",min_s=", s.min_s,
        ",median_s=", s.median_s,
        ",memory_bytes=", s.memory,
        ",allocs=", s.allocs,
    )
end

function print_remote_stats(label, result)
    println(
        label,
        ",expected=", result.target_count,
        ",serial_matched=", result.serial.matched,
        ",remote_matched=", result.remote.matched,
        ",worker_count=", length(result.workers),
        ",factorizations_first=", result.first_worker_factorizations,
        ",factorizations_second=", result.second_worker_factorizations,
        ",solution_buffers_first=", result.first_worker_solution_buffers,
        ",solution_buffers_second=", result.second_worker_solution_buffers,
        ",solves_first=", result.first_worker_solves,
        ",solves_second=", result.second_worker_solves,
        ",setup_s=", result.setup_elapsed_ns / 1e9,
        ",serial_update_s=", result.serial_update_elapsed_ns / 1e9,
        ",remote_first_s=", result.remote_first_elapsed_ns / 1e9,
        ",remote_second_s=", result.remote_second_elapsed_ns / 1e9,
        ",x_projection_gap=", result.x_projection_gap,
        ",y_projection_gap=", result.y_projection_gap,
    )
end

function run_schrodinger_serial()
    run_sparse_schrodinger_moment_gallery_smoke(; print_rows=false)
end

function run_schrodinger_remote(worker_count)
    run_sparse_schrodinger_remote_stored_factor_worker_smoke(; worker_count=worker_count, print_rows=false)
end

function main()
    problem = env("FEAST_MOMENT_BENCH_PROBLEM")
    workers = parse(Int, env("FEAST_MOMENT_BENCH_WORKERS"))
    samples = parse(Int, env("FEAST_MOMENT_BENCH_SAMPLES"))
    seconds_limit = parse(Float64, env("FEAST_MOMENT_BENCH_SECONDS"))

    problem == "schrodinger" || error("unsupported FEAST_MOMENT_BENCH_PROBLEM=$problem")
    println("config,problem=$problem,workers=$workers,samples=$samples,seconds=$seconds_limit")

    # Warm package, experiment, and worker code paths before timing.
    serial_result = run_schrodinger_serial()
    remote_result = run_schrodinger_remote(workers)

    serial_trial = run(@benchmarkable run_schrodinger_serial() samples=samples evals=1 seconds=seconds_limit)
    print_trial("moment,schrodinger_serial", serial_trial)
    println(
        "stats,schrodinger_serial",
        ",expected=", serial_result.target_count,
        ",initial_good=", serial_result.initial.good,
        ",updated_good=", serial_result.updated.good,
        ",updated_max_residual=", serial_result.updated.max_inside_residual,
        ",right_residual_rank=", serial_result.updated.right_residual_rank,
        ",left_residual_rank=", serial_result.updated.left_residual_rank,
    )

    remote_trial = run(@benchmarkable run_schrodinger_remote($workers) samples=samples evals=1 seconds=seconds_limit)
    print_trial("moment,schrodinger_remote_stored", remote_trial)
    print_remote_stats("stats,schrodinger_remote_stored", remote_result)
end

main()
