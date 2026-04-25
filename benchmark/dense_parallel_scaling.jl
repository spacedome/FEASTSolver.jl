using Distributed

counts = parse.(Int, split(get(ENV, "FEAST_BENCH_PROCS", "1,2,4,8"), ","))
max_count = maximum(counts)
external_workers() = filter(!=(myid()), workers())
if length(external_workers()) < max_count
    addprocs(max_count - length(external_workers()); exeflags="--project=$(Base.active_project())")
end

@everywhere using FEASTSolver
@everywhere using LinearAlgebra
@everywhere LinearAlgebra.BLAS.set_num_threads(1)

using FEASTSolver
using LinearAlgebra
using Random

function initial_subspace(n, m, seed)
    rand(MersenneTwister(seed), ComplexF64, n, m)
end

n = parse(Int, get(ENV, "FEAST_BENCH_N", "512"))
m = parse(Int, get(ENV, "FEAST_BENCH_M", "16"))
nodes = parse(Int, get(ENV, "FEAST_BENCH_NODES", "16"))
iter = parse(Int, get(ENV, "FEAST_BENCH_ITER", "4"))
store = parse(Bool, get(ENV, "FEAST_BENCH_STORE", "false"))
matrix_kind = get(ENV, "FEAST_BENCH_MATRIX", "diagonal")
warmup = parse(Bool, get(ENV, "FEAST_BENCH_WARMUP", "true"))
profile = parse(Bool, get(ENV, "FEAST_BENCH_PROFILE", "false"))

rng = MersenneTwister(19)
A = if matrix_kind == "diagonal"
    Matrix(Diagonal(range(1.0, n; length=n)))
elseif matrix_kind == "perturbed_hermitian"
    A0 = Matrix(Diagonal(range(1.0, n; length=n)))
    E = randn(rng, n, n)
    A0 .+ 0.01 .* Symmetric(E)
else
    error("unknown FEAST_BENCH_MATRIX=$matrix_kind; expected diagonal or perturbed_hermitian")
end
X0 = initial_subspace(n, m, 17)
c = 8.5
r = 8.0

println("matrix,$matrix_kind")
println("config,n=$n,m=$m,nodes=$nodes,iter=$iter,store=$store,counts=$(join(counts, ':'))")
if warmup
    warm_n = 32
    warm_m = min(m, 4)
    warm_A = Matrix(Diagonal(1.0:warm_n))
    warm_X = initial_subspace(warm_n, warm_m, 23)
    feast!(copy(warm_X), warm_A; nodes=4, iter=1, c=2.5, r=1.6, ϵ=-1.0, store=false)
    if !isempty(counts)
        warm_worker_count = min(length(external_workers()), maximum(counts), nodes)
        distributed_feast!(
            copy(warm_X),
            warm_A;
            nodes=max(4, warm_worker_count),
            iter=1,
            c=2.5,
            r=1.6,
            ϵ=-1.0,
            store=false,
            worker_ids=external_workers()[1:warm_worker_count],
            worker_blas_threads=1,
        )
    end
end

serial_time = @elapsed feast!(copy(X0), A; nodes=nodes, iter=iter, c=c, r=r, ϵ=-1.0, store=store)
println("serial,$serial_time")

for count in counts
    ids = external_workers()[1:count]
    GC.gc()
    local plan
    stats = DenseDistributedFeastStats()
    setup_time = @elapsed plan = DenseDistributedFeastPlan(
        A,
        m;
        nodes=nodes,
        c=c,
        r=r,
        store=store,
        worker_ids=ids,
        worker_blas_threads=1,
        stats=stats,
    )
    try
        solve_time = @elapsed distributed_feast!(
            copy(X0),
            plan;
            iter=iter,
            ϵ=-1.0,
            stats=stats,
        )
        println("distributed_setup,$count,$setup_time")
        println("distributed_solve,$count,$solve_time")
        println("distributed_total,$count,$(setup_time + solve_time)")
        if profile
            println("distributed_profile,$count,$stats")
        end
    finally
        close(plan)
    end
end
