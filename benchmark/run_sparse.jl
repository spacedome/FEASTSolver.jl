using BenchmarkTools
using FEASTSolver
using LinearAlgebra
using Random
using SparseArrays

const ENV_DEFAULTS = Dict(
    "FEAST_SPARSE_BENCH_GRID" => "45",
    "FEAST_SPARSE_BENCH_M" => "16",
    "FEAST_SPARSE_BENCH_NODES" => "16",
    "FEAST_SPARSE_BENCH_ITER" => "2",
    "FEAST_SPARSE_BENCH_STORE" => "false",
    "FEAST_SPARSE_BENCH_SAMPLES" => "3",
    "FEAST_SPARSE_BENCH_SECONDS" => "3600",
)

env(name) = get(ENV, name, ENV_DEFAULTS[name])
parse_bool(name) = parse(Bool, env(name))

function poisson2d(grid)
    T = spdiagm(-1 => fill(-1.0, grid - 1), 0 => fill(4.0, grid), 1 => fill(-1.0, grid - 1))
    Igrid = sparse(I, grid, grid)
    kron(Igrid, T) + kron(spdiagm(-1 => fill(-1.0, grid - 1), 1 => fill(-1.0, grid - 1)), Igrid)
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

seconds(ns::UInt64) = ns / 1e9

function print_stats(label, stats)
    println(
        label,
        ",solve_total_s=", seconds(stats.solve_total_ns),
        ",iterations=", stats.iterations,
        ",orthogonalization_s=", seconds(stats.orthogonalization_ns),
        ",rayleigh_ritz_s=", seconds(stats.rayleigh_ritz_ns),
        ",residual_s=", seconds(stats.residual_ns),
        ",filter_s=", seconds(stats.filter_ns),
    )
end

const FEAST_CONTEXT = Ref{Any}()

function run_feast_context!()
    ctx = FEAST_CONTEXT[]
    stats = DenseFeastStats()
    feast!(
        copy(ctx.X0),
        ctx.A;
        nodes=ctx.nodes,
        iter=ctx.iter,
        c=ctx.c,
        r=ctx.r,
        ϵ=-1.0,
        store=ctx.store,
        stats=stats,
    )
    stats
end

function warmup!(A, X0, nodes, c, r, store)
    feast!(copy(X0[:, 1:min(end, 4)]), A; nodes=min(nodes, 4), iter=1, c=c, r=r, ϵ=-1.0, store=store)
    nothing
end

function main()
    grid = parse(Int, env("FEAST_SPARSE_BENCH_GRID"))
    m = parse(Int, env("FEAST_SPARSE_BENCH_M"))
    nodes = parse(Int, env("FEAST_SPARSE_BENCH_NODES"))
    iter = parse(Int, env("FEAST_SPARSE_BENCH_ITER"))
    store = parse_bool("FEAST_SPARSE_BENCH_STORE")
    samples = parse(Int, env("FEAST_SPARSE_BENCH_SAMPLES"))
    seconds_limit = parse(Float64, env("FEAST_SPARSE_BENCH_SECONDS"))

    A = poisson2d(grid)
    n = size(A, 1)
    X0 = rand(MersenneTwister(91), ComplexF64, n, m)
    c, r = 0.25, 0.25
    contour = circular_contour_trapezoidal(c, r, nodes)
    z = contour_nodes(contour)[1]
    rhs = rand(MersenneTwister(92), ComplexF64, n, m)
    y = similar(rhs)
    Ishift = sparse(I, n, n)

    println("config,grid=$grid,n=$n,nnz=$(nnz(A)),m=$m,nodes=$nodes,iter=$iter,store=$store,samples=$samples,seconds=$seconds_limit")

    # Warm the package/JIT paths before timing.
    warmup!(A, X0, nodes, c, r, store)

    fresh_shift = @benchmarkable $A - $z * $Ishift samples=samples evals=1 seconds=seconds_limit
    print_trial("kernel,fresh_shift", run(fresh_shift))

    fresh_factor = @benchmarkable lu($A - $z * $Ishift) samples=samples evals=1 seconds=seconds_limit
    print_trial("kernel,fresh_shift_lu", run(fresh_factor))

    shift_ws = FEASTSolver._sparse_standard_shift_workspace(A)
    shifted = FEASTSolver._materialize_sparse_standard_shift!(shift_ws, z)
    factor = lu(shifted)

    reuse_shift = @benchmarkable FEASTSolver._materialize_sparse_standard_shift!($shift_ws, $z) samples=samples evals=1 seconds=seconds_limit
    print_trial("kernel,reuse_shift_materialize", run(reuse_shift))

    reuse_factor = @benchmarkable begin
        S = FEASTSolver._materialize_sparse_standard_shift!($shift_ws, $z)
        lu!($factor, S; reuse_symbolic=true)
    end samples=samples evals=1 seconds=seconds_limit
    print_trial("kernel,reuse_symbolic_lu!", run(reuse_factor))

    factored_solve = @benchmarkable ldiv!($y, $factor, $rhs) samples=samples evals=1 seconds=seconds_limit
    print_trial("kernel,factored_solve", run(factored_solve))

    FEAST_CONTEXT[] = (; A, X0, nodes, iter, c, r, store)
    feast_trial = run(@benchmarkable run_feast_context!() samples=samples evals=1 seconds=seconds_limit)
    print_trial("feast,standard_sparse", feast_trial)
    print_stats("stats,standard_sparse", run_feast_context!())
end

main()
