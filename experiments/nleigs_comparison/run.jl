using Distributed
using FEASTSolver
using LinearAlgebra
using MatrixMarket
using NonlinearEigenproblems
using Printf
using Random

import NonlinearEigenproblems: create_linsolver, lin_solve

const ENV_DEFAULTS = Dict(
    "FEAST_EXPERIMENT_PROBLEMS" => "butterfly,pep0,schrodinger_movebc",
    "FEAST_EXPERIMENT_METHODS" => "feast,nleigs",
    "FEAST_EXPERIMENT_PROCS" => "0,4,8",
    "FEAST_EXPERIMENT_FORMAT" => "pretty",
    "FEAST_EXPERIMENT_COLOR" => "auto",
    "FEAST_EXPERIMENT_REPEATS" => "1",
    "FEAST_EXPERIMENT_FEAST_CONFIGS" => "",
    "FEAST_EXPERIMENT_SEED" => "9901",
    "FEAST_EXPERIMENT_PROBLEM_N" => "",
    "FEAST_EXPERIMENT_HADELER_ALPHA" => "100",
    "FEAST_EXPERIMENT_CONTOUR_CENTER" => "",
    "FEAST_EXPERIMENT_CONTOUR_RADIUS" => "",
    "FEAST_EXPERIMENT_NLEIGS_CONTOUR_CENTER" => "",
    "FEAST_EXPERIMENT_NLEIGS_CONTOUR_RADIUS" => "",
    "FEAST_EXPERIMENT_BLAS_THREADS" => string(Sys.CPU_THREADS),
    "FEAST_EXPERIMENT_WORKER_BLAS_THREADS" => "1",
    "FEAST_EXPERIMENT_FEAST_NODES" => "",
    "FEAST_EXPERIMENT_FEAST_ITER" => "",
    "FEAST_EXPERIMENT_FEAST_STORE" => "",
    "FEAST_EXPERIMENT_FEAST_MATERIALIZE_NODES" => "false",
    "FEAST_EXPERIMENT_M" => "",
    "FEAST_EXPERIMENT_WARMUP" => "true",
    "FEAST_EXPERIMENT_VALIDATE_T_UPDATE" => "true",
    "FEAST_EXPERIMENT_NLEIGS_MAXIT" => "",
    "FEAST_EXPERIMENT_NLEIGS_MINIT" => "20",
    "FEAST_EXPERIMENT_NLEIGS_MAXDGR" => "100",
    "FEAST_EXPERIMENT_NLEIGS_TOL" => "1e-8",
    "FEAST_EXPERIMENT_NLEIGS_POLYGON_POINTS" => "",
    "FEAST_EXPERIMENT_NLEIGS_POLYGON_PHASE" => "",
    "FEAST_EXPERIMENT_NLEIGS_BLKSIZE" => "",
    "FEAST_EXPERIMENT_NLEIGS_STATIC" => "false",
    "FEAST_EXPERIMENT_NLEIGS_LEJA" => "1",
    "FEAST_EXPERIMENT_NLEIGS_REUSEFACT" => "auto",
)

env(name) = get(ENV, name, ENV_DEFAULTS[name])
parse_csv(name) = filter!(!isempty, strip.(split(env(name), ",")))
parse_int_csv(name) = parse.(Int, parse_csv(name))
parse_bool(name) = parse(Bool, env(name))
output_csv() = lowercase(env("FEAST_EXPERIMENT_FORMAT")) == "csv"
problem_n(default) = isempty(get(ENV, "FEAST_EXPERIMENT_PROBLEM_N", "")) ? default : parse(Int, ENV["FEAST_EXPERIMENT_PROBLEM_N"])
blas_threads() = parse(Int, env("FEAST_EXPERIMENT_BLAS_THREADS"))
worker_blas_threads() = parse(Int, env("FEAST_EXPERIMENT_WORKER_BLAS_THREADS"))
feast_materialize_nodes() = parse_bool("FEAST_EXPERIMENT_FEAST_MATERIALIZE_NODES")
pep0_default_subspace(n) = max(24, 2 * ceil(Int, 0.01 * n))

function problem_int(problem, env_name, field, fallback)
    value = env(env_name)
    isempty(value) || return parse(Int, value)
    field in propertynames(problem) && return getproperty(problem, field)
    fallback
end

nleigs_center(problem) = isempty(env("FEAST_EXPERIMENT_NLEIGS_CONTOUR_CENTER")) ? problem.c : parse(ComplexF64, env("FEAST_EXPERIMENT_NLEIGS_CONTOUR_CENTER"))
nleigs_radius(problem) = isempty(env("FEAST_EXPERIMENT_NLEIGS_CONTOUR_RADIUS")) ? problem.r : parse(Float64, env("FEAST_EXPERIMENT_NLEIGS_CONTOUR_RADIUS"))

function nleigs_polygon_phase(problem, polygon_points)
    value = env("FEAST_EXPERIMENT_NLEIGS_POLYGON_PHASE")
    isempty(value) && (:nleigs_polygon_phase in propertynames(problem)) && return getproperty(problem, :nleigs_polygon_phase) == :feast_nodes ? π / polygon_points : 0.0
    isempty(value) && return 0.0
    value == "vertices" && return 0.0
    value == "feast_nodes" && return π / polygon_points
    error("FEAST_EXPERIMENT_NLEIGS_POLYGON_PHASE must be vertices or feast_nodes")
end

function color_enabled()
    output_csv() && return false
    color = lowercase(env("FEAST_EXPERIMENT_COLOR"))
    color == "always" && return true
    color == "never" && return false
    color == "auto" || error("FEAST_EXPERIMENT_COLOR must be auto, always, or never")
    !haskey(ENV, "NO_COLOR") && !isempty(get(ENV, "TERM", "")) && get(ENV, "TERM", "") != "dumb"
end

paint(s, code) = color_enabled() ? "\e[$(code)m$(s)\e[0m" : string(s)
bold(s) = paint(s, "1")
dim(s) = paint(s, "2")
green(s) = paint(s, "32")
yellow(s) = paint(s, "33")
red(s) = paint(s, "31")
cyan(s) = paint(s, "36")
magenta(s) = paint(s, "35")
bright_orange(s) = paint(s, "1;38;5;208")

function experiment_repeats()
    n = parse(Int, env("FEAST_EXPERIMENT_REPEATS"))
    n > 0 || error("FEAST_EXPERIMENT_REPEATS must be positive")
    n
end

function nleigs_reusefact(problem)
    value = lowercase(strip(env("FEAST_EXPERIMENT_NLEIGS_REUSEFACT")))
    if value == "auto"
        return (problem.feast_store ? 1 : 0), "auto"
    end
    parsed = parse(Int, value)
    parsed in (0, 1, 2) || error("FEAST_EXPERIMENT_NLEIGS_REUSEFACT must be auto, 0, 1, or 2")
    parsed, "explicit"
end

mutable struct CountingLinSolverCreator{C} <: LinSolverCreator
    inner::C
    factorizations::Int
    solves::Int
end

CountingLinSolverCreator(inner) = CountingLinSolverCreator(inner, 0, 0)

struct CountingLinSolver{S,C} <: LinSolver
    inner::S
    counter::CountingLinSolverCreator{C}
end

function create_linsolver(counter::CountingLinSolverCreator, nep, λ)
    counter.factorizations += 1
    CountingLinSolver(create_linsolver(counter.inner, nep, λ), counter)
end

function lin_solve(solver::CountingLinSolver, rhs::AbstractVecOrMat; tol=0)
    solver.counter.solves += 1
    lin_solve(solver.inner, rhs; tol)
end

external_workers() = filter(!=(myid()), workers())

function ensure_workers(count, blas_threads)
    count <= 0 && return Int[]
    current = external_workers()
    if length(current) < count
        addprocs(count - length(current); exeflags="--project=$(Base.active_project())")
    end
    for pid in external_workers()
        remotecall_wait(Main.eval, pid, :(using FEASTSolver; using LinearAlgebra; using NonlinearEigenproblems; LinearAlgebra.BLAS.set_num_threads($blas_threads)))
    end
    external_workers()[1:count]
end

function circle_polygon(c, r, n; phase=0.0)
    θ = range(0, 2π; length=n + 1)[1:end-1]
    ComplexF64[c + r * cis(t + phase) for t in θ]
end

function initial_subspace(n, m, seed)
    rand(MersenneTwister(seed), ComplexF64, n, m)
end

function load_butterfly_matrices()
    root = dirname(dirname(@__DIR__))
    [ComplexF64.(Matrix(mmread(joinpath(root, "data", "butterflyM$(i).mtx")))) for i in 0:4]
end

function relative_residual(T, λ, x)
    Tλ = T(λ)
    norm(Tλ * x) / norm(Tλ)
end

function relative_residuals(T, λ, X)
    [relative_residual(T, λ[i], view(X, :, i)) for i in eachindex(λ)]
end

function feast_operator_action_residual_tools(T; normalize_by_matrix=true)
    M = operator_prototype(T)
    x = zeros(ComplexF64, size(T, 1))
    y = similar(x)

    function residual(λ, v)
        mul!(y, T, λ, v)
        if normalize_by_matrix
            materialize!(M, T, λ)
            return norm(y) / norm(M)
        end
        norm(y) / norm(v)
    end

    function residual_update!(res, X, R, Λ)
        @inbounds for j in axes(X, 2)
            xnorm = zero(real(eltype(X)))
            for i in axes(X, 1)
                xnorm += abs2(X[i, j])
            end
            inv_xnorm = inv(sqrt(xnorm))
            for i in axes(X, 1)
                x[i] = X[i, j] * inv_xnorm
                X[i, j] = x[i]
            end
            mul!(y, T, Λ[j], x)
            copyto!(view(R, :, j), y)
            if normalize_by_matrix
                materialize!(M, T, Λ[j])
                res[j] = norm(y) / norm(M)
            else
                res[j] = norm(y)
            end
        end
        res
    end

    residual, residual_update!
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

function pep_action_residual_tools(nep)
    matrix_norms = norm.(nep.A)

    function denominator(λ)
        total = zero(real(typeof(λ)))
        power = one(λ)
        for i in eachindex(nep.A)
            total += abs(power) * matrix_norms[i]
            power *= λ
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
        T = feast_gallery("polynomial", A)
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
            T_update=matrix_materializer(T),
            T_prototype=operator_prototype(T),
        )
    elseif name == "gun"
        T = feast_gallery("nlevp_native_gun")
        nep = nep_gallery("nlevp_native_gun")
        residual, residual_update = gun_residual_tools(nep)
        return (;
            name,
            T,
            nep,
            n=size(T, 1),
            c=140000.0 + 0.0im,
            r=30000.0,
            feast_label="sparse-default",
            m=32,
            feast_nodes=8,
            feast_iter=3,
            feast_store=false,
            feast_tol=1e-8,
            spurious=1e-5,
            nleigs_tol=1e-8,
            nleigs_maxit=150,
            nleigs_blksize=32,
            nleigs_polygon_points=32,
            nleigs_singularities=[0.0, 108.8774^2],
            residual,
            residual_update,
            T_update=matrix_materializer(T),
            T_prototype=operator_prototype(T),
        )
    elseif name == "schrodinger_movebc"
        n = problem_n(50000)
        T = feast_gallery("schrodinger_movebc", n)
        nep = nep_gallery("schrodinger_movebc", n)
        residual, residual_update = feast_operator_action_residual_tools(T; normalize_by_matrix=false)
        return (;
            name,
            T,
            nep,
            n,
            c=-35.0 + 0.0im,
            r=4.2,
            feast_label="sparse-default",
            m=8,
            feast_nodes=24,
            feast_iter=6,
            feast_store=false,
            feast_tol=1e-5,
            spurious=1e-5,
            nleigs_tol=1e-5,
            nleigs_solver_tol=1e-5,
            nleigs_maxit=150,
            nleigs_blksize=16,
            nleigs_polygon_points=24,
            nleigs_singularities=[-10.0],
            residual,
            residual_update,
            T_update=matrix_materializer(T),
            T_prototype=operator_prototype(T),
        )
    elseif name == "loaded_string"
        n = problem_n(500)
        T = feast_gallery("nlevp_native_loaded_string", n, 1, 1)
        nep = nep_gallery("nlevp_native_loaded_string", n, 1, 1)
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
            T_update=matrix_materializer(T),
            T_prototype=operator_prototype(T),
        )
    elseif name == "hadeler"
        n = problem_n(500)
        alpha = parse(Int, env("FEAST_EXPERIMENT_HADELER_ALPHA"))
        T = feast_gallery("nlevp_native_hadeler", alpha, n)
        nep = nep_gallery("nlevp_native_hadeler", alpha, n)
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
            T_update=matrix_materializer(T),
            T_prototype=operator_prototype(T),
        )
    elseif name == "pep0"
        n = problem_n(3000)
        nep = nep_gallery("pep0", n)
        T = feast_gallery("polynomial", nep.A)
        residual, residual_update = pep_action_residual_tools(nep)
        m = pep0_default_subspace(n)
        # Region/config search notes, kept here to avoid redoing the same sweep:
        # * radius 0.2 at n=1200 had roughly 50 interior eigenvalues; m=44/60
        #   was too small, and 16 nodes left spurious/edge values.
        # * radius 0.1 at n=1200 gave a clean 11-eigenvalue region with
        #   m=24, nodes=32, and two RII steps.
        # * radius 0.1 at n=3000 with m=60, nodes=32, store=false converged
        #   cleanly with 32 interior eigenpairs on 8 workers, but NLEIGS only
        #   recovered a small subset.
        # * radius 0.095 at n=3000 gives a better apples-to-apples comparison:
        #   FEAST and NLEIGS both recover 27 interior eigenpairs under the
        #   no-store/no-reuse memory policy. This radius sensitivity is part of
        #   the benchmark finding and should be reported.
        # NLEIGS tuning tried here included target radii 0.08/0.09/0.095/0.1/0.12,
        # 32/96 polygon points, aligned/unaligned polygon phases, blksize 32/96,
        # maxit 150, maxdgr 300, leja 1/2, and reusefact 0/1/2. At radius 0.1,
        # aligned 32-point NLEIGS improved from the earlier small subset to 18
        # eigenpairs, but still did not recover FEAST's 32-eigenpair region.
        # At radius 0.095, aligned 32-point NLEIGS with reusefact=0 recovered
        # only 7 eigenpairs in the full benchmark, so the default stays at the
        # robust unaligned 96-point polygon.
        return (;
            name,
            T,
            nep,
            n,
            c=0.0 + 0.0im,
            r=0.095,
            feast_label="default",
            m,
            feast_nodes=32,
            feast_iter=4,
            feast_store=false,
            feast_tol=1e-8,
            spurious=1e-5,
            nleigs_tol=1e-8,
            nleigs_maxit=150,
            nleigs_blksize=32,
            nleigs_polygon_points=96,
            nleigs_singularities=[Inf],
            residual,
            residual_update,
            T_update=matrix_materializer(T),
            T_prototype=operator_prototype(T),
        )
    elseif name == "pep0_sym"
        n = problem_n(500)
        gallery_nep = nep_gallery("pep0_sym", n)
        matrices = Matrix.(gallery_nep.A)
        # NEP-PACK's NLEIGS RK conversion type-asserts the result of vcat(Av...)
        # against eltype(Av). vcat of Symmetric wrappers is a plain Matrix, so use
        # the dense coefficients explicitly on both sides of this comparison.
        nep = PEP(matrices)
        T = feast_gallery("polynomial", matrices)
        residual, residual_update = pep_action_residual_tools(nep)
        m = pep0_default_subspace(n)
        return (;
            name,
            T,
            nep,
            n,
            c=0.0 + 0.0im,
            r=0.095,
            feast_label="default",
            m,
            feast_nodes=32,
            feast_iter=4,
            feast_store=false,
            feast_tol=1e-8,
            spurious=1e-5,
            nleigs_tol=1e-8,
            nleigs_maxit=150,
            nleigs_blksize=32,
            nleigs_polygon_points=96,
            nleigs_singularities=[Inf],
            residual,
            residual_update,
            T_update=matrix_materializer(T),
            T_prototype=operator_prototype(T),
        )
    else
        error("unknown experiment problem '$name'; expected butterfly, loaded_string, hadeler, pep0, or pep0_sym")
    end
end

function pretty_seconds(x)
    isnan(x) && return "NaN"
    @sprintf("%.3fs", x)
end

function pretty_gib(bytes)
    @sprintf("%.2f GiB", bytes / 1024^3)
end

function pretty_residual(x)
    isnan(x) && return "NaN"
    @sprintf("%.3e", x)
end

function median_value(xs)
    sorted = sort(xs)
    n = length(sorted)
    n == 0 && return NaN
    isodd(n) ? sorted[(n + 1) ÷ 2] : (sorted[n ÷ 2] + sorted[n ÷ 2 + 1]) / 2
end

function timing_summary(times)
    (;
        samples=length(times),
        elapsed_min_s=minimum(times),
        elapsed_median_s=median_value(times),
        elapsed_max_s=maximum(times),
        elapsed_samples=join((@sprintf("%.6g", t) for t in times), ":"),
    )
end

function progress(message)
    output_csv() && return nothing
    println("    ", dim(message))
    flush(stdout)
    nothing
end

function is_seconds_key(k)
    name = String(k)
    endswith(name, "_s") && name != "elapsed_samples"
end

cpu_model() = isempty(Sys.cpu_info()) ? "unknown" : Sys.cpu_info()[1].model
csv_value(x) = replace(string(x), "," => ";")

function csv_extra(extra)
    isempty(pairs(extra)) && return ""
    join((",$k=$v" for (k, v) in pairs(extra)))
end

function print_trace(trace)
    isempty(trace) && return
    println("    ", dim("trace:"), " ", dim(trace))
end

function residual_status(problem, max_converged_inside, converged_count)
    converged_count == 0 && return yellow("none")
    text = pretty_residual(max_converged_inside)
    max_converged_inside < problem.nleigs_tol && return green(text)
    yellow(text)
end

function convergence_status(problem, converged_count, spurious_count)
    if converged_count > 0 && spurious_count == 0
        green("converged")
    elseif converged_count > 0
        yellow("converged + spurious")
    elseif spurious_count > 0
        red("spurious")
    else
        yellow("partial")
    end
end

function summarize_result(problem, method, processes, elapsed, λ, X, residuals; extra=(;))
    inside = in_contour(λ, problem.c, problem.r)
    inside_res = residuals[inside]
    converged = inside_res .< problem.nleigs_tol
    spurious = inside_res .>= problem.spurious
    max_inside = isempty(inside_res) ? NaN : maximum(inside_res)
    min_inside = isempty(inside_res) ? NaN : minimum(inside_res)
    converged_inside_res = inside_res[converged]
    max_converged_inside = isempty(converged_inside_res) ? NaN : maximum(converged_inside_res)
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
    inside_count = count(inside)
    converged_count = count(converged)
    spurious_count = count(spurious)
    status = convergence_status(problem, converged_count, spurious_count)
    println("  ", bold(cyan(method)), " [", problem.feast_label, ", ", process_label, "] ", status)
    timed_samples = haskey(extra, :timed_samples) ? extra.timed_samples : 1
    elapsed_label = timed_samples == 1 ? "elapsed:" : "elapsed median ($(timed_samples) timed samples):"
    println("    ", bold(elapsed_label), " ", bright_orange(pretty_seconds(elapsed)))
    if haskey(extra, :elapsed_min_s) && haskey(extra, :elapsed_max_s)
        println(
            "    ", dim("timing samples:"), " min=", magenta(pretty_seconds(extra.elapsed_min_s)),
            ", median=", magenta(pretty_seconds(extra.elapsed_median_s)),
            ", max=", magenta(pretty_seconds(extra.elapsed_max_s)),
            ", raw=", dim(extra.elapsed_samples),
        )
    end
    println(
        "    ", bold("eigenpairs:"), " returned=", length(λ),
        ", inside=", inside_count,
        ", converged_inside=", converged_count,
        ", spurious_inside=", spurious_count,
    )
    println(
        "    ", bold("residuals converged inside:"), " max=",
        residual_status(problem, max_converged_inside, converged_count),
        "    ", dim("(all inside: min=$(pretty_residual(min_inside)), max=$(pretty_residual(max_inside)))"),
    )
    for (k, v) in pairs(extra)
        k in (:trace, :timed_samples, :elapsed_min_s, :elapsed_median_s, :elapsed_max_s, :elapsed_samples) && continue
        if v isa Real && is_seconds_key(k)
            println("    ", k, ": ", magenta(pretty_seconds(v)))
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

function validate_operator_hooks(problem)
    parse_bool("FEAST_EXPERIMENT_VALIDATE_T_UPDATE") || return nothing
    problem.T_update === nothing && return nothing

    z = problem.c + problem.r * cis(0.37)
    reference = problem.T(z)
    buffer = similar(reference)
    problem.T_update(buffer, z)
    relerr = norm(buffer - reference) / max(norm(reference), eps(Float64))
    relerr <= 1e-10 || error("T_update validation failed for $(problem.name): relative error $relerr")
    if !output_csv()
        println("  T_update validation relative error: ", pretty_residual(relerr))
    end
    nothing
end

function run_feast(problem, processes, seed)
    worker_threads = worker_blas_threads()
    worker_ids = ensure_workers(processes, worker_threads)

    function solve_once()
        X = initial_subspace(problem.n, problem.m, seed)
        stats = processes == 0 ? DenseFeastStats() : DenseDistributedFeastStats()
        local λ
        local V
        local res
        elapsed = if processes == 0
            @elapsed begin
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
        else
            @elapsed begin
                λ, V, res = distributed_nlfeast!(
                    problem.T,
                    X,
                    problem.feast_nodes,
                    problem.feast_iter;
                    c=problem.c,
                    r=problem.r,
                    ϵ=problem.feast_tol,
                    store=problem.feast_store,
                    materialize_nodes=feast_materialize_nodes(),
                    spurious=problem.spurious,
                    worker_ids=worker_ids,
                    worker_blas_threads=worker_threads,
                    residual_update=problem.residual_update,
                    stats=stats,
                )
            end
        end
        common_res = common_residuals(problem, λ, V)
        (; elapsed, λ, V, res, common_res, stats)
    end

    if parse_bool("FEAST_EXPERIMENT_WARMUP")
        try
            progress("warmup: $(problem.name) $(problem.feast_label) $(processes == 0 ? "serial nlfeast" : "$(processes)-worker distributed_nlfeast")")
            solve_once()
        catch err
            @warn "FEAST benchmark warmup failed; continuing without it" problem=problem.name config=problem.feast_label processes exception=(err, catch_backtrace())
        end
    end

    times = Float64[]
    last = nothing
    repeats = experiment_repeats()
    for sample_index in 1:repeats
        progress("timed sample $(sample_index)/$(repeats): $(problem.name) $(problem.feast_label) $(processes == 0 ? "serial nlfeast" : "$(processes)-worker distributed_nlfeast")")
        sample = solve_once()
        push!(times, sample.elapsed)
        last = sample
    end

    λ = last.λ
    V = last.V
    common_res = last.common_res
    stats = last.stats
    elapsed = median_value(times)
    timing = timing_summary(times)
    inside = in_contour(λ, problem.c, problem.r)
    converged_inside = count(common_res[inside] .< problem.nleigs_tol)
    if processes == 0
        extra = (;
            timed_samples=timing.samples,
            elapsed_min_s=timing.elapsed_min_s,
            elapsed_median_s=timing.elapsed_median_s,
            elapsed_max_s=timing.elapsed_max_s,
            elapsed_samples=timing.elapsed_samples,
            m=problem.m,
            nodes=problem.feast_nodes,
            iter_limit=problem.feast_iter,
            store=problem.feast_store,
            blas_threads=BLAS.get_num_threads(),
            iterations=stats.iterations,
            rii_steps=max(stats.iterations - 1, 0),
            beyn_only=stats.iterations <= 1,
            stop=feast_stop_label(stats, problem.feast_iter, converged_inside, count(inside)),
            linear_solves=problem.feast_nodes * stats.iterations,
            factorizations=problem.feast_store ? problem.feast_nodes : problem.feast_nodes * stats.iterations,
            solve_s=stats.solve_total_ns / 1e9,
            filter_s=stats.filter_ns / 1e9,
            residual_s=stats.residual_ns / 1e9,
            trace=feast_trace(stats),
        )
        summarize_result(problem, "nlfeast", 0, elapsed, λ, V, common_res; extra=extra)
    else
        extra = (;
            timed_samples=timing.samples,
            elapsed_min_s=timing.elapsed_min_s,
            elapsed_median_s=timing.elapsed_median_s,
            elapsed_max_s=timing.elapsed_max_s,
            elapsed_samples=timing.elapsed_samples,
            m=problem.m,
            nodes=problem.feast_nodes,
            iter_limit=problem.feast_iter,
            store=problem.feast_store,
            materialize_nodes=feast_materialize_nodes(),
            worker_blas_threads=worker_threads,
            iterations=stats.iterations,
            rii_steps=max(stats.iterations - 1, 0),
            beyn_only=stats.iterations <= 1,
            stop=feast_stop_label(stats, problem.feast_iter, converged_inside, count(inside)),
            linear_solves=problem.feast_nodes * stats.iterations,
            factorizations=problem.feast_store ? problem.feast_nodes : problem.feast_nodes * stats.iterations,
            setup_s=(stats.setup_prepare_ns + stats.setup_master_ns + stats.setup_worker_ns) / 1e9,
            setup_prepare_s=stats.setup_prepare_ns / 1e9,
            setup_master_s=stats.setup_master_ns / 1e9,
            setup_worker_s=stats.setup_worker_ns / 1e9,
            solve_s=stats.solve_total_ns / 1e9,
            worker_s=stats.worker_step_ns / 1e9,
            worker_solve_sum_s=stats.worker_solve_ns / 1e9,
            worker_materialize_sum_s=stats.worker_materialize_ns / 1e9,
            worker_linsolve_sum_s=stats.worker_linsolve_ns / 1e9,
            worker_accum_sum_s=stats.worker_accum_ns / 1e9,
            input_transfer_s=stats.input_transfer_ns / 1e9,
            reduce_s=stats.reduce_ns / 1e9,
            ritz_s=stats.rayleigh_ritz_ns / 1e9,
            residual_s=stats.residual_ns / 1e9,
            trace=feast_trace(stats),
        )
        summarize_result(problem, "distributed_nlfeast", processes, elapsed, λ, V, common_res; extra=extra)
    end
    nothing
end

function run_nleigs(problem, seed)
    polygon_points = problem_int(problem, "FEAST_EXPERIMENT_NLEIGS_POLYGON_POINTS", :nleigs_polygon_points, 96)
    polygon_phase = nleigs_polygon_phase(problem, polygon_points)
    center = nleigs_center(problem)
    radius = nleigs_radius(problem)
    Σ = circle_polygon(center, radius, polygon_points; phase=polygon_phase)
    maxit = problem_int(problem, "FEAST_EXPERIMENT_NLEIGS_MAXIT", :nleigs_maxit, 100)
    minit = parse(Int, env("FEAST_EXPERIMENT_NLEIGS_MINIT"))
    maxdgr = parse(Int, env("FEAST_EXPERIMENT_NLEIGS_MAXDGR"))
    tol = isempty(get(ENV, "FEAST_EXPERIMENT_NLEIGS_TOL", "")) && :nleigs_solver_tol in propertynames(problem) ?
          problem.nleigs_solver_tol :
          parse(Float64, env("FEAST_EXPERIMENT_NLEIGS_TOL"))
    blksize = problem_int(problem, "FEAST_EXPERIMENT_NLEIGS_BLKSIZE", :nleigs_blksize, 20)
    static = parse_bool("FEAST_EXPERIMENT_NLEIGS_STATIC")
    leja = parse(Int, env("FEAST_EXPERIMENT_NLEIGS_LEJA"))
    reusefact, reusefact_source = nleigs_reusefact(problem)
    errmeasure = (λ, x) -> problem.residual(λ, x)

    function solve_once()
        v = initial_subspace(problem.n, 1, seed)[:, 1]
        linsolver = CountingLinSolverCreator(DefaultLinSolverCreator())
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
                linsolvercreator=linsolver,
                return_details=false,
            )
        end
        residuals = common_residuals(problem, λ, V)
        (; elapsed, λ, V, res, residuals, linsolver)
    end

    if parse_bool("FEAST_EXPERIMENT_WARMUP")
        try
            progress("warmup: $(problem.name) $(problem.feast_label) nleigs")
            solve_once()
        catch err
            @warn "NLEIGS benchmark warmup failed; continuing without it" problem=problem.name exception=(err, catch_backtrace())
        end
    end

    times = Float64[]
    last = nothing
    repeats = experiment_repeats()
    for sample_index in 1:repeats
        progress("timed sample $(sample_index)/$(repeats): $(problem.name) $(problem.feast_label) nleigs")
        sample = solve_once()
        push!(times, sample.elapsed)
        last = sample
    end
    λ = last.λ
    V = last.V
    residuals = last.residuals
    linsolver = last.linsolver
    elapsed = median_value(times)
    timing = timing_summary(times)
    extra = (;
        timed_samples=timing.samples,
        elapsed_min_s=timing.elapsed_min_s,
        elapsed_median_s=timing.elapsed_median_s,
        elapsed_max_s=timing.elapsed_max_s,
        elapsed_samples=timing.elapsed_samples,
        maxit,
        minit,
        maxdgr,
        blksize,
        static,
        leja,
        reusefact,
        reusefact_source,
        matched_feast_store=problem.feast_store,
        target_center=center,
        target_radius=radius,
        polygon_points,
        polygon_phase,
        singularities=join(problem.nleigs_singularities, ":"),
        linear_solves=linsolver.solves,
        factorizations=linsolver.factorizations,
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
    repeats = experiment_repeats()
    total_memory = Sys.total_memory()
    free_memory = Sys.free_memory()

    if output_csv()
        println(
            "config",
            ",problems=", join(problems, ":"),
            ",methods=", join(methods, ":"),
            ",processes=", join(processes, ":"),
            ",color=", env("FEAST_EXPERIMENT_COLOR"),
            ",repeats=", repeats,
            ",blas_threads=", BLAS.get_num_threads(),
            ",worker_blas_threads=", worker_blas_threads(),
            ",seed=", seed,
            ",julia_version=", VERSION,
            ",cpu_threads=", Sys.CPU_THREADS,
            ",cpu_model=", csv_value(cpu_model()),
            ",total_memory_bytes=", total_memory,
            ",free_memory_bytes=", free_memory,
            ",warmup=", parse_bool("FEAST_EXPERIMENT_WARMUP"),
        )
    else
        println(bold(cyan("NLEIGS comparison experiment")))
        println("  problems: ", bold(join(problems, ", ")))
        println("  methods: ", join(methods, ", "))
        println("  FEAST processes: ", join(processes, ", "))
        println("  repeats: ", repeats, " timed sample", repeats == 1 ? "" : "s", " per method/config/process")
        println("  BLAS threads: ", BLAS.get_num_threads())
        println("  worker BLAS threads: ", worker_blas_threads())
        println("  seed: ", seed)
        println("  Julia: ", VERSION)
        println("  CPU: ", cpu_model(), " (", Sys.CPU_THREADS, " threads)")
        println("  memory: total=", pretty_gib(total_memory), ", free=", pretty_gib(free_memory))
        println("  warmup: ", parse_bool("FEAST_EXPERIMENT_WARMUP"), parse_bool("FEAST_EXPERIMENT_WARMUP") ? " (one extra untimed full solve per method/config/process)" : "")
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
            println(bold("Problem: "), bold(problem.name))
            println("  n: ", problem.n)
            println("  contour: center=", problem.c, ", radius=", problem.r)
        end
        if "feast" in methods
            for feast_problem in configs
                validate_operator_hooks(feast_problem)
                for count in processes
                    run_feast(feast_problem, count, seed)
                end
            end
        end
        if "nleigs" in methods
            if "feast" in methods
                for feast_problem in configs
                    run_nleigs(feast_problem, seed)
                end
            else
                run_nleigs(problem, seed)
            end
        end
    end
end

main()
