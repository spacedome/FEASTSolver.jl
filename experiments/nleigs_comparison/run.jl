using Distributed
using FEASTSolver
using LinearAlgebra
using MatrixMarket
using NonlinearEigenproblems
using Printf
using Random

const ENV_DEFAULTS = Dict(
    "FEAST_EXPERIMENT_PROBLEMS" => "butterfly",
    "FEAST_EXPERIMENT_METHODS" => "feast,nleigs",
    "FEAST_EXPERIMENT_PROCS" => "0,2",
    "FEAST_EXPERIMENT_FORMAT" => "pretty",
    "FEAST_EXPERIMENT_FEAST_CONFIGS" => "",
    "FEAST_EXPERIMENT_SEED" => "9901",
    "FEAST_EXPERIMENT_PROBLEM_N" => "",
    "FEAST_EXPERIMENT_HADELER_ALPHA" => "100",
    "FEAST_EXPERIMENT_CONTOUR_CENTER" => "",
    "FEAST_EXPERIMENT_CONTOUR_RADIUS" => "",
    "FEAST_EXPERIMENT_BLAS_THREADS" => "1",
    "FEAST_EXPERIMENT_WORKER_BLAS_THREADS" => "1",
    "FEAST_EXPERIMENT_FEAST_NODES" => "",
    "FEAST_EXPERIMENT_FEAST_ITER" => "",
    "FEAST_EXPERIMENT_FEAST_STORE" => "",
    "FEAST_EXPERIMENT_M" => "",
    "FEAST_EXPERIMENT_WARMUP" => "true",
    "FEAST_EXPERIMENT_NLEIGS_MAXIT" => "100",
    "FEAST_EXPERIMENT_NLEIGS_MINIT" => "20",
    "FEAST_EXPERIMENT_NLEIGS_MAXDGR" => "100",
    "FEAST_EXPERIMENT_NLEIGS_TOL" => "1e-8",
    "FEAST_EXPERIMENT_NLEIGS_POLYGON_POINTS" => "96",
    "FEAST_EXPERIMENT_NLEIGS_BLKSIZE" => "20",
    "FEAST_EXPERIMENT_NLEIGS_STATIC" => "false",
    "FEAST_EXPERIMENT_NLEIGS_LEJA" => "1",
    "FEAST_EXPERIMENT_NLEIGS_REUSEFACT" => "1",
)

env(name) = get(ENV, name, ENV_DEFAULTS[name])
parse_csv(name) = filter!(!isempty, strip.(split(env(name), ",")))
parse_int_csv(name) = parse.(Int, parse_csv(name))
parse_bool(name) = parse(Bool, env(name))
output_csv() = lowercase(env("FEAST_EXPERIMENT_FORMAT")) == "csv"
problem_n(default) = isempty(env("FEAST_EXPERIMENT_PROBLEM_N")) ? default : parse(Int, env("FEAST_EXPERIMENT_PROBLEM_N"))
blas_threads() = parse(Int, env("FEAST_EXPERIMENT_BLAS_THREADS"))
worker_blas_threads() = parse(Int, env("FEAST_EXPERIMENT_WORKER_BLAS_THREADS"))

external_workers() = filter(!=(myid()), workers())

function ensure_workers(count, blas_threads)
    count <= 0 && return Int[]
    current = external_workers()
    if length(current) < count
        addprocs(count - length(current); exeflags="--project=$(Base.active_project())")
    end
    for pid in external_workers()
        remotecall_wait(Main.eval, pid, :(using FEASTSolver; using LinearAlgebra; LinearAlgebra.BLAS.set_num_threads($blas_threads)))
    end
    external_workers()[1:count]
end

function circle_polygon(c, r, n)
    θ = range(0, 2π; length=n + 1)[1:end-1]
    ComplexF64[c + r * cis(t) for t in θ]
end

function initial_subspace(n, m, seed)
    rand(MersenneTwister(seed), ComplexF64, n, m)
end

function load_butterfly_matrices()
    root = dirname(dirname(@__DIR__))
    [ComplexF64.(Matrix(mmread(joinpath(root, "data", "butterflyM$(i).mtx")))) for i in 0:4]
end

function polynomial_operator(A)
    z -> z^4 * A[5] + z^3 * A[4] + z^2 * A[3] + z * A[2] + A[1]
end

function relative_residual(T, λ, x)
    Tλ = T(λ)
    norm(Tλ * x) / norm(Tλ)
end

function relative_residuals(T, λ, X)
    [relative_residual(T, λ[i], view(X, :, i)) for i in eachindex(λ)]
end

function gun_residual_tools(nep)
    sigma1 = 0.0
    sigma2 = 108.8774
    nK = 1.474544889815002e5
    nM = 2.726114618171165e-2
    nW1 = 2.328612251920476e0
    nW2 = 3.793375498194695e0

    denominator(λ) = nK + abs(λ) * nM + sqrt(abs(λ - sigma1^2)) * nW1 + sqrt(abs(λ - sigma2^2)) * nW2

    function residual(λ, v)
        norm(compute_Mlincomb(nep, λ, v)) / denominator(λ)
    end

    function residual_update!(res, X, R, Λ)
        @inbounds for j in axes(X, 2)
            xnorm = zero(real(eltype(X)))
            for i in axes(X, 1)
                xnorm += abs2(X[i, j])
            end
            inv_xnorm = inv(sqrt(xnorm))
            x = view(X, :, j)
            for i in axes(X, 1)
                x[i] *= inv_xnorm
            end
            y = compute_Mlincomb(nep, Λ[j], x)
            copyto!(view(R, :, j), y)
            res[j] = norm(y) / denominator(Λ[j])
        end
        res
    end

    residual, residual_update!
end

function spmf_action_residual_tools(nep)
    matrix_norms = norm.(nep.A)

    function denominator(λ)
        total = zero(real(typeof(λ)))
        for i in eachindex(nep.A)
            total += abs(nep.fi[i](λ)) * matrix_norms[i]
        end
        total
    end

    function residual(λ, v)
        norm(compute_Mlincomb(nep, λ, v)) / denominator(λ)
    end

    function residual_update!(res, X, R, Λ)
        @inbounds for j in axes(X, 2)
            xnorm = zero(real(eltype(X)))
            for i in axes(X, 1)
                xnorm += abs2(X[i, j])
            end
            inv_xnorm = inv(sqrt(xnorm))
            x = view(X, :, j)
            for i in axes(X, 1)
                x[i] *= inv_xnorm
            end
            y = compute_Mlincomb(nep, Λ[j], x)
            copyto!(view(R, :, j), y)
            res[j] = norm(y) / denominator(Λ[j])
        end
        res
    end

    residual, residual_update!
end

function problem_config(name)
    if name == "butterfly"
        A = load_butterfly_matrices()
        T = polynomial_operator(A)
        nep = PEP(A)
        n = size(A[1], 1)
        return (;
            name,
            T,
            nep,
            n,
            c=1.0 + 1.0im,
            r=0.5,
            feast_label="default",
            m=30,
            feast_nodes=16,
            feast_iter=3,
            feast_store=true,
            feast_tol=1e-8,
            spurious=5e-3,
            nleigs_tol=1e-8,
            nleigs_singularities=[Inf],
            residual=(λ, x) -> relative_residual(T, λ, x),
            residual_update=nothing,
        )
    elseif name == "gun"
        nep = nep_gallery("nlevp_native_gun")
        T = z -> compute_Mder(nep, z)
        n = size(nep, 1)
        residual, residual_update = gun_residual_tools(nep)
        return (;
            name,
            T,
            nep,
            n,
            c=140000.0 + 0.0im,
            r=30000.0,
            feast_label="default",
            m=32,
            feast_nodes=32,
            feast_iter=3,
            feast_store=false,
            feast_tol=1e-8,
            spurious=1e-5,
            nleigs_tol=1e-8,
            nleigs_singularities=[Inf],
            residual,
            residual_update,
        )
    elseif name == "loaded_string"
        n = problem_n(500)
        nep = nep_gallery("nlevp_native_loaded_string", n, 1, 1)
        T = z -> compute_Mder(nep, z)
        return (;
            name,
            T,
            nep,
            n,
            c=4.5 + 0.0im,
            r=0.5,
            feast_label="default",
            m=6,
            feast_nodes=16,
            feast_iter=4,
            feast_store=true,
            feast_tol=1e-8,
            spurious=1e-5,
            nleigs_tol=1e-8,
            nleigs_singularities=[1.0],
            residual=(λ, x) -> relative_residual(T, λ, x),
            residual_update=nothing,
        )
    elseif name == "hadeler"
        n = problem_n(500)
        alpha = parse(Int, env("FEAST_EXPERIMENT_HADELER_ALPHA"))
        nep = nep_gallery("nlevp_native_hadeler", alpha, n)
        T = z -> compute_Mder(nep, z)
        residual, residual_update = spmf_action_residual_tools(nep)
        return (;
            name,
            T,
            nep,
            n,
            c=-25.0 + 0.0im,
            r=3.0,
            feast_label="default",
            m=28,
            feast_nodes=24,
            feast_iter=4,
            feast_store=true,
            feast_tol=1e-8,
            spurious=1e-5,
            nleigs_tol=1e-8,
            nleigs_singularities=[Inf],
            residual,
            residual_update,
        )
    else
        error("unknown experiment problem '$name'; expected butterfly, gun, loaded_string, or hadeler")
    end
end

function pretty_seconds(x)
    isnan(x) && return "NaN"
    @sprintf("%.3fs", x)
end

function pretty_residual(x)
    isnan(x) && return "NaN"
    @sprintf("%.3e", x)
end

function csv_extra(extra)
    isempty(pairs(extra)) && return ""
    join((",$k=$v" for (k, v) in pairs(extra)))
end

function print_trace(trace)
    isempty(trace) && return
    println("    trace: ", trace)
end

function summarize_result(problem, method, processes, elapsed, λ, X, residuals; extra=(;))
    inside = in_contour(λ, problem.c, problem.r)
    inside_res = residuals[inside]
    converged = inside_res .< problem.nleigs_tol
    spurious = inside_res .>= problem.spurious
    max_inside = isempty(inside_res) ? NaN : maximum(inside_res)
    min_inside = isempty(inside_res) ? NaN : minimum(inside_res)
    if output_csv()
        println(
            "result",
            ",problem=", problem.name,
            ",config=", problem.feast_label,
            ",method=", method,
            ",processes=", processes,
            ",elapsed_s=", elapsed,
            ",returned=", length(λ),
            ",inside=", count(inside),
            ",converged_inside=", count(converged),
            ",spurious_inside=", count(spurious),
            ",min_res_inside=", min_inside,
            ",max_res_inside=", max_inside,
            csv_extra(extra),
        )
        return
    end

    process_label = processes == 0 ? "serial" : "$(processes) workers"
    println("  ", method, " [", problem.feast_label, ", ", process_label, "]")
    println("    elapsed: ", pretty_seconds(elapsed))
    println(
        "    eigenpairs: returned=", length(λ),
        ", inside=", count(inside),
        ", converged_inside=", count(converged),
        ", spurious_inside=", count(spurious),
    )
    println("    residuals inside: min=", pretty_residual(min_inside), ", max=", pretty_residual(max_inside))
    for (k, v) in pairs(extra)
        k === :trace && continue
        if k in (:solve_s, :setup_s, :filter_s, :worker_s, :residual_s)
            println("    ", k, ": ", pretty_seconds(v))
        else
            println("    ", k, ": ", v)
        end
    end
    haskey(extra, :trace) && print_trace(extra.trace)
end

function feast_stop_label(stats, requested_iter, converged_inside, inside)
    if inside > 0 && converged_inside == inside
        "all_inside_converged"
    elseif stats.iterations < requested_iter + 1
        "spurious_or_partial_converged"
    else
        "iteration_limit"
    end
end

function feast_trace(stats)
    isempty(stats.iteration_log) && return ""
    entries = map(stats.iteration_log) do item
        string(
            item.iteration,
            ":inside=", item.eigenvalues_inside,
            ":conv1e-5=", item.converged_inside,
            ":max=", item.max_res_inside,
        )
    end
    join(entries, "|")
end

function common_residuals(problem, λ, X)
    [problem.residual(λ[i], view(X, :, i)) for i in eachindex(λ)]
end

function run_feast(problem, processes, seed)
    X = initial_subspace(problem.n, problem.m, seed)
    stats = processes == 0 ? DenseFeastStats() : DenseDistributedFeastStats()
    worker_threads = worker_blas_threads()
    worker_ids = ensure_workers(processes, worker_threads)
    if processes == 0
        local λ
        local V
        local res
        elapsed = @elapsed begin
            λ, V, res = nlfeast!(
                problem.T,
                X,
                problem.feast_nodes,
                problem.feast_iter;
                c=problem.c,
                r=problem.r,
                ϵ=problem.feast_tol,
                store=problem.feast_store,
                spurious=problem.spurious,
                residual_update=problem.residual_update,
                stats=stats,
            )
        end
        common_res = common_residuals(problem, λ, V)
        inside = in_contour(λ, problem.c, problem.r)
        converged_inside = count(common_res[inside] .< problem.nleigs_tol)
        extra = (;
            m=problem.m,
            nodes=problem.feast_nodes,
            iter_limit=problem.feast_iter,
            store=problem.feast_store,
            blas_threads=BLAS.get_num_threads(),
            iterations=stats.iterations,
            rii_steps=max(stats.iterations - 1, 0),
            beyn_only=stats.iterations <= 1,
            stop=feast_stop_label(stats, problem.feast_iter, converged_inside, count(inside)),
            solve_s=stats.solve_total_ns / 1e9,
            filter_s=stats.filter_ns / 1e9,
            residual_s=stats.residual_ns / 1e9,
            trace=feast_trace(stats),
        )
        summarize_result(problem, "nlfeast", 0, elapsed, λ, V, common_res; extra=extra)
    else
        local λ
        local V
        local res
        elapsed = @elapsed begin
            λ, V, res = distributed_nlfeast!(
                problem.T,
                X,
                problem.feast_nodes,
                problem.feast_iter;
                c=problem.c,
                r=problem.r,
                ϵ=problem.feast_tol,
                store=problem.feast_store,
                spurious=problem.spurious,
                worker_ids=worker_ids,
                worker_blas_threads=worker_threads,
                residual_update=problem.residual_update,
                stats=stats,
            )
        end
        common_res = common_residuals(problem, λ, V)
        inside = in_contour(λ, problem.c, problem.r)
        converged_inside = count(common_res[inside] .< problem.nleigs_tol)
        extra = (;
            m=problem.m,
            nodes=problem.feast_nodes,
            iter_limit=problem.feast_iter,
            store=problem.feast_store,
            worker_blas_threads=worker_threads,
            iterations=stats.iterations,
            rii_steps=max(stats.iterations - 1, 0),
            beyn_only=stats.iterations <= 1,
            stop=feast_stop_label(stats, problem.feast_iter, converged_inside, count(inside)),
            setup_s=(stats.setup_prepare_ns + stats.setup_shared_ns + stats.setup_worker_ns) / 1e9,
            solve_s=stats.solve_total_ns / 1e9,
            worker_s=stats.worker_step_ns / 1e9,
            residual_s=stats.residual_ns / 1e9,
            trace=feast_trace(stats),
        )
        summarize_result(problem, "distributed_nlfeast", processes, elapsed, λ, V, common_res; extra=extra)
    end
    nothing
end

function warmup_feast!(problem, seed)
    warm_m = min(problem.m, max(4, problem.m ÷ 2))
    warm_nodes = min(problem.feast_nodes, 8)
    X = initial_subspace(problem.n, warm_m, seed + 101)
    try
        nlfeast!(
            problem.T,
            X,
            warm_nodes,
            0;
            c=problem.c,
            r=problem.r,
            ϵ=-1.0,
            store=false,
            spurious=problem.spurious,
        )
    catch err
        @warn "FEAST warmup failed; continuing without it" problem=problem.name exception=(err, catch_backtrace())
    end
    nothing
end

function warmup_nleigs!(problem, seed)
    polygon_points = min(parse(Int, env("FEAST_EXPERIMENT_NLEIGS_POLYGON_POINTS")), 24)
    Σ = circle_polygon(problem.c, problem.r, polygon_points)
    v = initial_subspace(problem.n, 1, seed + 202)[:, 1]
    try
        nleigs(
            problem.nep,
            Σ;
            maxit=5,
            minit=1,
            maxdgr=5,
            tol=1e-4,
            v=v,
            errmeasure=(λ, x) -> problem.residual(λ, x),
            Ξ=problem.nleigs_singularities,
            blksize=min(parse(Int, env("FEAST_EXPERIMENT_NLEIGS_BLKSIZE")), 8),
            return_details=false,
        )
    catch err
        @warn "NLEIGS warmup failed; continuing without it" problem=problem.name exception=(err, catch_backtrace())
    end
    nothing
end

function run_nleigs(problem, seed)
    polygon_points = parse(Int, env("FEAST_EXPERIMENT_NLEIGS_POLYGON_POINTS"))
    Σ = circle_polygon(problem.c, problem.r, polygon_points)
    maxit = parse(Int, env("FEAST_EXPERIMENT_NLEIGS_MAXIT"))
    minit = parse(Int, env("FEAST_EXPERIMENT_NLEIGS_MINIT"))
    maxdgr = parse(Int, env("FEAST_EXPERIMENT_NLEIGS_MAXDGR"))
    tol = parse(Float64, env("FEAST_EXPERIMENT_NLEIGS_TOL"))
    blksize = parse(Int, env("FEAST_EXPERIMENT_NLEIGS_BLKSIZE"))
    static = parse_bool("FEAST_EXPERIMENT_NLEIGS_STATIC")
    leja = parse(Int, env("FEAST_EXPERIMENT_NLEIGS_LEJA"))
    reusefact = parse(Int, env("FEAST_EXPERIMENT_NLEIGS_REUSEFACT"))
    v = initial_subspace(problem.n, 1, seed)[:, 1]
    errmeasure = (λ, x) -> problem.residual(λ, x)

    local λ
    local V
    local res
    elapsed = @elapsed begin
        λ, V, res, _ = nleigs(
            problem.nep,
            Σ;
            maxit=maxit,
            minit=minit,
            maxdgr=maxdgr,
            tol=tol,
            v=v,
            errmeasure=errmeasure,
            Ξ=problem.nleigs_singularities,
            blksize=blksize,
            static=static,
            leja=leja,
            reusefact=reusefact,
            return_details=false,
        )
    end
    residuals = common_residuals(problem, λ, V)
    extra = (;
        maxit,
        minit,
        maxdgr,
        blksize,
        static,
        leja,
        reusefact,
        polygon_points,
        singularities=join(problem.nleigs_singularities, ":"),
    )
    summarize_result(problem, "nleigs", 0, elapsed, λ, V, residuals; extra=extra)
    nothing
end

function override_problem(problem)
    center = env("FEAST_EXPERIMENT_CONTOUR_CENTER")
    radius = env("FEAST_EXPERIMENT_CONTOUR_RADIUS")
    nodes = env("FEAST_EXPERIMENT_FEAST_NODES")
    iter = env("FEAST_EXPERIMENT_FEAST_ITER")
    store = env("FEAST_EXPERIMENT_FEAST_STORE")
    m = env("FEAST_EXPERIMENT_M")
    isempty(center) && isempty(radius) && isempty(nodes) && isempty(iter) && isempty(store) && isempty(m) && return problem
    (; problem...,
       c=isempty(center) ? problem.c : parse(ComplexF64, center),
       r=isempty(radius) ? problem.r : parse(Float64, radius),
       feast_nodes=isempty(nodes) ? problem.feast_nodes : parse(Int, nodes),
       feast_iter=isempty(iter) ? problem.feast_iter : parse(Int, iter),
       feast_store=isempty(store) ? problem.feast_store : parse(Bool, store),
       m=isempty(m) ? problem.m : parse(Int, m))
end

function parse_feast_config(problem, spec, index)
    fields = Dict{String, String}()
    for part in strip.(split(spec, ","))
        isempty(part) && continue
        key_value = split(part, "="; limit=2)
        length(key_value) == 2 || error("invalid FEAST config '$part'; expected key=value")
        fields[strip(key_value[1])] = strip(key_value[2])
    end
    label = get(fields, "label", "feast$(index)")
    (;
        problem...,
        feast_label=label,
        m=haskey(fields, "m") ? parse(Int, fields["m"]) : problem.m,
        feast_nodes=haskey(fields, "nodes") ? parse(Int, fields["nodes"]) : problem.feast_nodes,
        feast_iter=haskey(fields, "iter") ? parse(Int, fields["iter"]) : problem.feast_iter,
        feast_store=haskey(fields, "store") ? parse(Bool, fields["store"]) : problem.feast_store,
    )
end

function feast_configurations(problem)
    configs = env("FEAST_EXPERIMENT_FEAST_CONFIGS")
    if isempty(strip(configs))
        return [problem]
    end
    specs = filter!(!isempty, strip.(split(configs, ";")))
    [parse_feast_config(problem, spec, i) for (i, spec) in enumerate(specs)]
end

function main()
    BLAS.set_num_threads(blas_threads())
    problems = parse_csv("FEAST_EXPERIMENT_PROBLEMS")
    methods = parse_csv("FEAST_EXPERIMENT_METHODS")
    processes = parse_int_csv("FEAST_EXPERIMENT_PROCS")
    seed = parse(Int, env("FEAST_EXPERIMENT_SEED"))

    if output_csv()
        println(
            "config",
            ",problems=", join(problems, ":"),
            ",methods=", join(methods, ":"),
            ",processes=", join(processes, ":"),
            ",blas_threads=", BLAS.get_num_threads(),
            ",worker_blas_threads=", worker_blas_threads(),
            ",seed=", seed,
        )
    else
        println("NLEIGS comparison experiment")
        println("  problems: ", join(problems, ", "))
        println("  methods: ", join(methods, ", "))
        println("  FEAST processes: ", join(processes, ", "))
        println("  BLAS threads: ", BLAS.get_num_threads())
        println("  worker BLAS threads: ", worker_blas_threads())
        println("  seed: ", seed)
    end

    for name in problems
        problem = override_problem(problem_config(name))
        configs = feast_configurations(problem)
        if output_csv()
            println(
                "problem",
                ",name=", problem.name,
                ",n=", problem.n,
                ",c=", problem.c,
                ",r=", problem.r,
            )
        else
            println()
            println("Problem: ", problem.name)
            println("  n: ", problem.n)
            println("  contour: center=", problem.c, ", radius=", problem.r)
        end
        if parse_bool("FEAST_EXPERIMENT_WARMUP")
            if "feast" in methods
                warmup_feast!(problem, seed)
            end
            if "nleigs" in methods
                warmup_nleigs!(problem, seed)
            end
        end
        if "feast" in methods
            for feast_problem in configs
                for count in processes
                    run_feast(feast_problem, count, seed)
                end
            end
        end
        if "nleigs" in methods
            run_nleigs(problem, seed)
        end
    end
end

main()
