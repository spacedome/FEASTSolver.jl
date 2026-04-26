using BenchmarkTools
using Distributed
using FEASTSolver
using LinearAlgebra
using Random

const ENV_DEFAULTS = Dict(
    "FEAST_BENCH_VARIANTS" => "standard,generalized,nonlinear",
    "FEAST_BENCH_PROCS" => "1,2,4",
    "FEAST_BENCH_N" => "512",
    "FEAST_BENCH_M" => "16",
    "FEAST_BENCH_NODES" => "16",
    "FEAST_BENCH_ITER" => "3",
    "FEAST_BENCH_STORE" => "true",
    "FEAST_BENCH_MATRIX" => "perturbed_hermitian",
    "FEAST_BENCH_SAMPLES" => "1",
    "FEAST_BENCH_SECONDS" => "3600",
    "FEAST_BENCH_WARMUP" => "true",
    "FEAST_BENCH_SERIAL" => "true",
)

env(name) = get(ENV, name, ENV_DEFAULTS[name])
parse_csv(name) = filter!(!isempty, strip.(split(env(name), ",")))
parse_int_csv(name) = parse.(Int, parse_csv(name))
parse_bool(name) = parse(Bool, env(name))

external_workers() = filter(!=(myid()), workers())

function ensure_workers(count)
    current = external_workers()
    if length(current) < count
        addprocs(count - length(current); exeflags="--project=$(Base.active_project())")
    end
    for pid in external_workers()
        remotecall_wait(Main.eval, pid, :(using FEASTSolver; using LinearAlgebra; LinearAlgebra.BLAS.set_num_threads(1)))
    end
    external_workers()
end

function initial_subspace(n, m, seed)
    rand(MersenneTwister(seed), ComplexF64, n, m)
end

function dense_problem(kind, n)
    rng = MersenneTwister(19)
    D = Matrix(Diagonal(range(1.0, n; length=n)))
    if kind == "diagonal"
        D
    elseif kind == "perturbed_hermitian"
        E = randn(rng, n, n)
        D .+ 0.01 .* Symmetric(E)
    else
        error("unknown FEAST_BENCH_MATRIX=$kind; expected diagonal or perturbed_hermitian")
    end
end

function nonlinear_operator(A)
    n = size(A, 1)
    Iₙ = Matrix{ComplexF64}(I, n, n)
    Aₙ = ComplexF64.(A)
    z -> (z^2 - 0.25z) .* Iₙ .- Aₙ
end

function contour_params(n)
    center = min(8.5, max(2.5, n / 8))
    radius = min(8.0, max(1.6, n / 8 - 0.5))
    center, radius
end

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

function print_trial(label, variant, phase, processes, trial)
    s = trial_summary(trial)
    println(
        label,
        ",variant=", variant,
        ",phase=", phase,
        ",processes=", processes,
        ",samples=", s.samples,
        ",min_s=", s.min_s,
        ",median_s=", s.median_s,
        ",memory_bytes=", s.memory,
        ",allocs=", s.allocs,
    )
end

seconds(ns::UInt64) = ns / 1e9

const SERIAL_CONTEXT = Ref{Any}()
const SETUP_CONTEXT = Ref{Any}()
const SOLVE_CONTEXT = Ref{Any}()

function print_distributed_stats(variant, processes, stats)
    println(
        "stats",
        ",variant=", variant,
        ",processes=", processes,
        ",setup_prepare_s=", seconds(stats.setup_prepare_ns),
        ",setup_master_s=", seconds(stats.setup_master_ns),
        ",setup_worker_s=", seconds(stats.setup_worker_ns),
        ",solve_total_s=", seconds(stats.solve_total_ns),
        ",iterations=", stats.iterations,
        ",qr_s=", seconds(stats.qr_ns),
        ",rayleigh_ritz_s=", seconds(stats.rayleigh_ritz_ns),
        ",residual_s=", seconds(stats.residual_ns),
        ",input_transfer_s=", seconds(stats.input_transfer_ns),
        ",worker_step_s=", seconds(stats.worker_step_ns),
        ",reduce_s=", seconds(stats.reduce_ns),
    )
end

function print_serial_stats(variant, stats)
    println(
        "serial_stats",
        ",variant=", variant,
        ",solve_total_s=", seconds(stats.solve_total_ns),
        ",iterations=", stats.iterations,
        ",orthogonalization_s=", seconds(stats.orthogonalization_ns),
        ",rayleigh_ritz_s=", seconds(stats.rayleigh_ritz_ns),
        ",residual_s=", seconds(stats.residual_ns),
        ",filter_s=", seconds(stats.filter_ns),
    )
end

function run_serial!(variant, X0, A, B, T, nodes, iter, c, r, store)
    stats = DenseFeastStats()
    if variant == "standard"
        feast!(copy(X0), A; nodes=nodes, iter=iter, c=c, r=r, ϵ=-1.0, store=store, stats=stats)
    elseif variant == "generalized"
        gen_feast!(copy(X0), A, B; nodes=nodes, iter=iter, c=c, r=r, ϵ=-1.0, store=store, stats=stats)
    elseif variant == "nonlinear"
        nlfeast!(T, copy(X0), nodes, iter; c=c, r=r, ϵ=-1.0, store=store, stats=stats)
    else
        error("unknown variant=$variant")
    end
    stats
end

function run_serial_context!()
    ctx = SERIAL_CONTEXT[]
    run_serial!(ctx.variant, ctx.X0, ctx.A, ctx.B, ctx.T, ctx.nodes, ctx.iter, ctx.c, ctx.r, ctx.store)
end

function make_plan(variant, A, B, T, n, m, nodes, c, r, store, ids)
    stats = DenseDistributedFeastStats()
    plan = if variant == "standard"
        DenseDistributedFeastPlan(A, m; nodes=nodes, c=c, r=r, store=store, worker_ids=ids, worker_blas_threads=1, stats=stats)
    elseif variant == "generalized"
        DenseDistributedGeneralizedFeastPlan(A, B, m; nodes=nodes, c=c, r=r, store=store, worker_ids=ids, worker_blas_threads=1, stats=stats)
    elseif variant == "nonlinear"
        DenseDistributedNonlinearFeastPlan(T, n, m; nodes=nodes, c=c, r=r, store=store, worker_ids=ids, worker_blas_threads=1, stats=stats)
    else
        error("unknown variant=$variant")
    end
    plan, stats
end

function setup_context!()
    ctx = SETUP_CONTEXT[]
    plan, stats = make_plan(ctx.variant, ctx.A, ctx.B, ctx.T, ctx.n, ctx.m, ctx.nodes, ctx.c, ctx.r, ctx.store, ctx.worker_ids)
    close(plan)
    stats
end

function solve_with_plan!(variant, plan, X0, iter, stats)
    setup_prepare_ns = stats.setup_prepare_ns
    setup_master_ns = stats.setup_master_ns
    setup_worker_ns = stats.setup_worker_ns
    FEASTSolver.reset!(stats)
    stats.setup_prepare_ns = setup_prepare_ns
    stats.setup_master_ns = setup_master_ns
    stats.setup_worker_ns = setup_worker_ns
    if variant == "standard"
        distributed_feast!(copy(X0), plan; iter=iter, ϵ=-1.0, stats=stats)
    elseif variant == "generalized"
        distributed_gen_feast!(copy(X0), plan; iter=iter, ϵ=-1.0, stats=stats)
    elseif variant == "nonlinear"
        distributed_nlfeast!(copy(X0), plan, iter; ϵ=-1.0, stats=stats)
    else
        error("unknown variant=$variant")
    end
    stats
end

function solve_context!()
    ctx = SOLVE_CONTEXT[]
    solve_with_plan!(ctx.variant, ctx.plan, ctx.X0, ctx.iter, ctx.stats)
end

function warmup!(A, B, T)
    n = min(size(A, 1), 64)
    m = 4
    X = initial_subspace(n, m, 23)
    Aw = Matrix(A[1:n, 1:n])
    Bw = Matrix(B[1:n, 1:n])
    Tw = nonlinear_operator(Aw)
    feast!(copy(X), Aw; nodes=4, iter=1, c=2.5, r=1.6, ϵ=-1.0, store=false)
    gen_feast!(copy(X), Aw, Bw; nodes=4, iter=1, c=2.5, r=1.6, ϵ=-1.0, store=false)
    nlfeast!(Tw, copy(X), 4, 1; c=2.5, r=1.6, ϵ=-1.0, store=false)
    nothing
end

function main()
    variants = parse_csv("FEAST_BENCH_VARIANTS")
    counts = parse_int_csv("FEAST_BENCH_PROCS")
    n = parse(Int, env("FEAST_BENCH_N"))
    m = parse(Int, env("FEAST_BENCH_M"))
    nodes = parse(Int, env("FEAST_BENCH_NODES"))
    iter = parse(Int, env("FEAST_BENCH_ITER"))
    store = parse_bool("FEAST_BENCH_STORE")
    matrix_kind = env("FEAST_BENCH_MATRIX")
    samples = parse(Int, env("FEAST_BENCH_SAMPLES"))
    seconds_limit = parse(Float64, env("FEAST_BENCH_SECONDS"))
    run_serial_baseline = parse_bool("FEAST_BENCH_SERIAL")

    max_count = isempty(counts) ? 0 : maximum(counts)
    ids = max_count == 0 ? Int[] : ensure_workers(max_count)
    BLAS.set_num_threads(1)

    A = dense_problem(matrix_kind, n)
    B = Matrix(Diagonal(2.0 .+ (1.0:n) ./ n))
    T = nonlinear_operator(A)
    X0 = initial_subspace(n, m, 17)
    c, r = contour_params(n)

    println("config,n=$n,m=$m,nodes=$nodes,iter=$iter,store=$store,matrix=$matrix_kind,variants=$(join(variants, ':')),processes=$(join(counts, ':')),samples=$samples,seconds=$seconds_limit")

    if parse_bool("FEAST_BENCH_WARMUP")
        warmup!(A, B, T)
    end

    for variant in variants
        if run_serial_baseline
            SERIAL_CONTEXT[] = (; variant, X0, A, B, T, nodes, iter, c, r, store)
            serial_bench = @benchmarkable run_serial_context!() samples=samples evals=1 seconds=seconds_limit
            serial_trial = run(serial_bench)
            print_trial("trial", variant, "serial_solve", 0, serial_trial)
            print_serial_stats(variant, run_serial!(variant, X0, A, B, T, nodes, iter, c, r, store))
        end

        for count in counts
            worker_ids = ids[1:count]

            SETUP_CONTEXT[] = (; variant, A, B, T, n, m, nodes, c, r, store, worker_ids)
            setup_bench = @benchmarkable setup_context!() samples=samples evals=1 seconds=seconds_limit
            setup_trial = run(setup_bench)
            print_trial("trial", variant, "distributed_setup_close", count, setup_trial)

            plan, stats = make_plan(variant, A, B, T, n, m, nodes, c, r, store, worker_ids)
            try
                SOLVE_CONTEXT[] = (; variant, plan, X0, iter, stats)
                solve_bench = @benchmarkable solve_context!() samples=samples evals=1 seconds=seconds_limit
                solve_trial = run(solve_bench)
                print_trial("trial", variant, "distributed_solve", count, solve_trial)
                print_distributed_stats(variant, count, solve_with_plan!(variant, plan, X0, iter, stats))
            finally
                close(plan)
            end
        end
    end
end

main()
