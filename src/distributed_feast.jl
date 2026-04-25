const _DISTRIBUTED_DENSE_FEAST_WORKSPACES = Dict{Symbol, Any}()

abstract type AbstractDenseDistributedFeastPlan end

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

"""
    DenseDistributedFeastPlan(A, m0, [contour]; kwargs...)

Persistent contour-worker plan for dense standard FEAST. The plan owns shared
buffers, fixed worker-to-node assignments, and optional worker-local LU factors.
"""
mutable struct DenseDistributedFeastPlan <: AbstractDenseDistributedFeastPlan
    key::Symbol
    A
    X_shared
    R_shared
    Qparts
    contour::Contour
    worker_ids::Vector{Int}
    assignments::Vector{Vector{Int}}
    store::Bool
    worker_blas_threads::Int
    Λ::Vector{ComplexF64}
    res::Vector{Float64}
    inside::BitVector
    R::Matrix{ComplexF64}
    Q::Matrix{ComplexF64}
    Aq::Matrix{ComplexF64}
    Xq::Matrix{ComplexF64}
    qr_ws
    eigen_ws
    futures::Vector{Any}
    stats::Union{Nothing, DenseDistributedFeastStats}
    closed::Bool
end

"""
    DenseDistributedGeneralizedFeastPlan(A, B, m0, [contour]; kwargs...)

Persistent contour-worker plan for dense generalized FEAST on `Ax = λBx`.
"""
mutable struct DenseDistributedGeneralizedFeastPlan <: AbstractDenseDistributedFeastPlan
    key::Symbol
    A
    B
    X_shared
    R_shared
    Qparts
    contour::Contour
    worker_ids::Vector{Int}
    assignments::Vector{Vector{Int}}
    store::Bool
    worker_blas_threads::Int
    Λ::Vector{ComplexF64}
    res::Vector{Float64}
    inside::BitVector
    R::Matrix{ComplexF64}
    Q::Matrix{ComplexF64}
    Aq::Matrix{ComplexF64}
    Bq::Matrix{ComplexF64}
    Xq::Matrix{ComplexF64}
    qr_ws
    eigen_ws
    futures::Vector{Any}
    stats::Union{Nothing, DenseDistributedFeastStats}
    closed::Bool
end

"""
    DenseDistributedDualGeneralizedFeastPlan(A, B, m0, [contour]; kwargs...)

Persistent contour-worker plan for dense dual generalized FEAST on non-normal
pencils. Right and left subspace filters share the same worker-owned shifted
factors when possible.
"""
mutable struct DenseDistributedDualGeneralizedFeastPlan <: AbstractDenseDistributedFeastPlan
    key::Symbol
    A
    B
    Xr_shared
    Xl_shared
    Rr_shared
    Rl_shared
    Qrparts
    Qlparts
    contour::Contour
    worker_ids::Vector{Int}
    assignments::Vector{Vector{Int}}
    store::Bool
    worker_blas_threads::Int
    Λ::Vector{ComplexF64}
    resr::Vector{Float64}
    inside::BitVector
    Rr::Matrix{ComplexF64}
    Rl::Matrix{ComplexF64}
    Qr::Matrix{ComplexF64}
    Ql::Matrix{ComplexF64}
    Aq::Matrix{ComplexF64}
    Bq::Matrix{ComplexF64}
    Xql::Matrix{ComplexF64}
    Xqr::Matrix{ComplexF64}
    eigen_ws
    svd_ws
    residual_shift::Matrix{ComplexF64}
    residual_x::Vector{ComplexF64}
    residual_y::Vector{ComplexF64}
    futures::Vector{Any}
    stats::Union{Nothing, DenseDistributedFeastStats}
    closed::Bool
end

mutable struct DenseFeastDistributedWorkspace
    A
    X
    R
    Qparts
    part_index::Int
    node_indices::Vector{Int}
    nodes::Vector{ComplexF64}
    weights::Vector{ComplexF64}
    temp::Matrix{ComplexF64}
    ZmA::Matrix{ComplexF64}
    resolvent::Vector{ComplexF64}
    lu_ws
    stored_shifts
    stored_lu_ws
    old_blas_threads::Int
end

mutable struct DenseGeneralizedFeastDistributedWorkspace
    A
    B
    X
    R
    Qparts
    part_index::Int
    node_indices::Vector{Int}
    nodes::Vector{ComplexF64}
    weights::Vector{ComplexF64}
    temp::Matrix{ComplexF64}
    ZmA::Matrix{ComplexF64}
    resolvent::Vector{ComplexF64}
    lu_ws
    stored_shifts
    stored_lu_ws
    old_blas_threads::Int
end

mutable struct DenseDualGeneralizedFeastDistributedWorkspace
    A
    B
    Xr
    Xl
    Rr
    Rl
    Qrparts
    Qlparts
    part_index::Int
    node_indices::Vector{Int}
    nodes::Vector{ComplexF64}
    weights::Vector{ComplexF64}
    temp::Matrix{ComplexF64}
    ZmA::Matrix{ComplexF64}
    resolvent::Vector{ComplexF64}
    lu_ws
    stored_shifts
    stored_lu_ws
    old_blas_threads::Int
end

function DenseDistributedFeastPlan(
    A::AbstractMatrix,
    m₀::Integer;
    nodes::Integer=8,
    c=complex(0.0, 0.0),
    r=1.0,
    store=false,
    worker_ids=_default_feast_worker_ids(),
    worker_blas_threads::Integer=1,
    stats::Union{Nothing, DenseDistributedFeastStats}=nothing,
)
    contour = circular_contour_trapezoidal(c, r, nodes)
    DenseDistributedFeastPlan(
        A,
        m₀,
        contour;
        store=store,
        worker_ids=worker_ids,
        worker_blas_threads=worker_blas_threads,
        stats=stats,
    )
end

function DenseDistributedFeastPlan(
    A::AbstractMatrix,
    m₀::Integer,
    contour::Contour;
    store=false,
    worker_ids=_default_feast_worker_ids(),
    worker_blas_threads::Integer=1,
    stats::Union{Nothing, DenseDistributedFeastStats}=nothing,
)
    m₀ > 0 || error("subspace dimension must be positive")
    size(A, 1) == size(A, 2) || error("Incorrect dimensions of A, must be square")

    worker_ids = _normalize_feast_worker_ids(worker_ids)
    isempty(worker_ids) && error("DenseDistributedFeastPlan requires worker processes; call addprocs(...) or start Julia with -p")

    nodes = size(contour.nodes, 1)
    worker_ids, assignments = _dense_feast_worker_assignments(worker_ids, nodes)
    all_pids = unique([myid(); worker_ids])

    start_ns = time_ns()
    _prepare_dense_feast_workers!(worker_ids)
    _add_elapsed!(stats, :setup_prepare_ns, start_ns)

    start_ns = time_ns()
    N = size(A, 1)
    A_shared = SharedArray{ComplexF64}(size(A); pids=all_pids)
    X_shared = SharedArray{ComplexF64}((N, m₀); pids=all_pids)
    R_shared = SharedArray{ComplexF64}((N, m₀); pids=all_pids)
    Qparts = SharedArray{ComplexF64}((N, m₀, length(worker_ids)); pids=all_pids)
    copyto!(A_shared, A)

    Λ = zeros(ComplexF64, m₀)
    res = zeros(m₀)
    inside = falses(m₀)
    R = zeros(ComplexF64, N, m₀)
    Q = zeros(ComplexF64, N, m₀)
    Aq = zeros(ComplexF64, m₀, m₀)
    Xq = zeros(ComplexF64, m₀, m₀)
    qr_ws = QRWs(Q)
    eigen_ws = EigenWs(Aq, rvecs=true)
    key = gensym(:dense_feast)
    futures = Vector{Any}(undef, length(worker_ids))
    _add_elapsed!(stats, :setup_shared_ns, start_ns)

    plan = DenseDistributedFeastPlan(
        key,
        A_shared,
        X_shared,
        R_shared,
        Qparts,
        contour,
        worker_ids,
        assignments,
        Bool(store),
        Int(worker_blas_threads),
        Λ,
        res,
        inside,
        R,
        Q,
        Aq,
        Xq,
        qr_ws,
        eigen_ws,
        futures,
        stats,
        false,
    )

    try
        start_ns = time_ns()
        _init_dense_feast_workers!(plan)
        _add_elapsed!(stats, :setup_worker_ns, start_ns)
    catch err
        close(plan)
        rethrow(err)
    end

    plan
end

function DenseDistributedGeneralizedFeastPlan(
    A::AbstractMatrix,
    B,
    m₀::Integer;
    nodes::Integer=8,
    c=complex(0.0, 0.0),
    r=1.0,
    store=false,
    worker_ids=_default_feast_worker_ids(),
    worker_blas_threads::Integer=1,
    stats::Union{Nothing, DenseDistributedFeastStats}=nothing,
)
    contour = circular_contour_trapezoidal(c, r, nodes)
    DenseDistributedGeneralizedFeastPlan(
        A,
        B,
        m₀,
        contour;
        store=store,
        worker_ids=worker_ids,
        worker_blas_threads=worker_blas_threads,
        stats=stats,
    )
end

function DenseDistributedGeneralizedFeastPlan(
    A::AbstractMatrix,
    B,
    m₀::Integer,
    contour::Contour;
    store=false,
    worker_ids=_default_feast_worker_ids(),
    worker_blas_threads::Integer=1,
    stats::Union{Nothing, DenseDistributedFeastStats}=nothing,
)
    m₀ > 0 || error("subspace dimension must be positive")
    size(A, 1) == size(A, 2) || error("Incorrect dimensions of A, must be square")

    worker_ids = _normalize_feast_worker_ids(worker_ids)
    isempty(worker_ids) && error("DenseDistributedGeneralizedFeastPlan requires worker processes; call addprocs(...) or start Julia with -p")

    nodes = size(contour.nodes, 1)
    worker_ids, assignments = _dense_feast_worker_assignments(worker_ids, nodes)
    all_pids = unique([myid(); worker_ids])

    start_ns = time_ns()
    _prepare_dense_feast_workers!(worker_ids)
    _add_elapsed!(stats, :setup_prepare_ns, start_ns)

    start_ns = time_ns()
    N = size(A, 1)
    A_shared = _shared_complex_matrix(A, (N, N), all_pids)
    B_shared = _shared_complex_matrix(B, (N, N), all_pids)
    X_shared = SharedArray{ComplexF64}((N, m₀); pids=all_pids)
    R_shared = SharedArray{ComplexF64}((N, m₀); pids=all_pids)
    Qparts = SharedArray{ComplexF64}((N, m₀, length(worker_ids)); pids=all_pids)

    Λ = zeros(ComplexF64, m₀)
    res = zeros(m₀)
    inside = falses(m₀)
    R = zeros(ComplexF64, N, m₀)
    Q = zeros(ComplexF64, N, m₀)
    Aq = zeros(ComplexF64, m₀, m₀)
    Bq = zeros(ComplexF64, m₀, m₀)
    Xq = zeros(ComplexF64, m₀, m₀)
    qr_ws = QRWs(Q)
    eigen_ws = GeneralizedEigenWs(Aq, lvecs=true, rvecs=true)
    key = gensym(:dense_gen_feast)
    futures = Vector{Any}(undef, length(worker_ids))
    _add_elapsed!(stats, :setup_shared_ns, start_ns)

    plan = DenseDistributedGeneralizedFeastPlan(
        key,
        A_shared,
        B_shared,
        X_shared,
        R_shared,
        Qparts,
        contour,
        worker_ids,
        assignments,
        Bool(store),
        Int(worker_blas_threads),
        Λ,
        res,
        inside,
        R,
        Q,
        Aq,
        Bq,
        Xq,
        qr_ws,
        eigen_ws,
        futures,
        stats,
        false,
    )

    try
        start_ns = time_ns()
        _init_dense_gen_feast_workers!(plan)
        _add_elapsed!(stats, :setup_worker_ns, start_ns)
    catch err
        close(plan)
        rethrow(err)
    end

    plan
end

function DenseDistributedDualGeneralizedFeastPlan(
    A::AbstractMatrix,
    B,
    m₀::Integer;
    nodes::Integer=8,
    c=complex(0.0, 0.0),
    r=1.0,
    store=false,
    worker_ids=_default_feast_worker_ids(),
    worker_blas_threads::Integer=1,
    stats::Union{Nothing, DenseDistributedFeastStats}=nothing,
)
    contour = circular_contour_trapezoidal(c, r, nodes)
    DenseDistributedDualGeneralizedFeastPlan(
        A,
        B,
        m₀,
        contour;
        store=store,
        worker_ids=worker_ids,
        worker_blas_threads=worker_blas_threads,
        stats=stats,
    )
end

function DenseDistributedDualGeneralizedFeastPlan(
    A::AbstractMatrix,
    B,
    m₀::Integer,
    contour::Contour;
    store=false,
    worker_ids=_default_feast_worker_ids(),
    worker_blas_threads::Integer=1,
    stats::Union{Nothing, DenseDistributedFeastStats}=nothing,
)
    m₀ > 0 || error("subspace dimension must be positive")
    size(A, 1) == size(A, 2) || error("Incorrect dimensions of A, must be square")

    worker_ids = _normalize_feast_worker_ids(worker_ids)
    isempty(worker_ids) && error("DenseDistributedDualGeneralizedFeastPlan requires worker processes; call addprocs(...) or start Julia with -p")

    nodes = size(contour.nodes, 1)
    worker_ids, assignments = _dense_feast_worker_assignments(worker_ids, nodes)
    all_pids = unique([myid(); worker_ids])

    start_ns = time_ns()
    _prepare_dense_feast_workers!(worker_ids)
    _add_elapsed!(stats, :setup_prepare_ns, start_ns)

    start_ns = time_ns()
    N = size(A, 1)
    A_shared = _shared_complex_matrix(A, (N, N), all_pids)
    B_shared = _shared_complex_matrix(B, (N, N), all_pids)
    Xr_shared = SharedArray{ComplexF64}((N, m₀); pids=all_pids)
    Xl_shared = SharedArray{ComplexF64}((N, m₀); pids=all_pids)
    Rr_shared = SharedArray{ComplexF64}((N, m₀); pids=all_pids)
    Rl_shared = SharedArray{ComplexF64}((N, m₀); pids=all_pids)
    Qrparts = SharedArray{ComplexF64}((N, m₀, length(worker_ids)); pids=all_pids)
    Qlparts = SharedArray{ComplexF64}((N, m₀, length(worker_ids)); pids=all_pids)

    Λ = zeros(ComplexF64, m₀)
    resr = zeros(m₀)
    inside = falses(m₀)
    Rr = zeros(ComplexF64, N, m₀)
    Rl = zeros(ComplexF64, N, m₀)
    Qr = zeros(ComplexF64, N, m₀)
    Ql = zeros(ComplexF64, N, m₀)
    Aq = zeros(ComplexF64, m₀, m₀)
    Bq = zeros(ComplexF64, m₀, m₀)
    Xql = zeros(ComplexF64, m₀, m₀)
    Xqr = zeros(ComplexF64, m₀, m₀)
    eigen_ws = GeneralizedEigenWs(Aq, lvecs=true, rvecs=true)
    svd_ws = SVDsddWs(Bq, job='A')
    residual_shift = zeros(ComplexF64, N, N)
    residual_x = zeros(ComplexF64, N)
    residual_y = zeros(ComplexF64, N)
    key = gensym(:dense_dual_gen_feast)
    futures = Vector{Any}(undef, length(worker_ids))
    _add_elapsed!(stats, :setup_shared_ns, start_ns)

    plan = DenseDistributedDualGeneralizedFeastPlan(
        key,
        A_shared,
        B_shared,
        Xr_shared,
        Xl_shared,
        Rr_shared,
        Rl_shared,
        Qrparts,
        Qlparts,
        contour,
        worker_ids,
        assignments,
        Bool(store),
        Int(worker_blas_threads),
        Λ,
        resr,
        inside,
        Rr,
        Rl,
        Qr,
        Ql,
        Aq,
        Bq,
        Xql,
        Xqr,
        eigen_ws,
        svd_ws,
        residual_shift,
        residual_x,
        residual_y,
        futures,
        stats,
        false,
    )

    try
        start_ns = time_ns()
        _init_dense_dual_gen_feast_workers!(plan)
        _add_elapsed!(stats, :setup_worker_ns, start_ns)
    catch err
        close(plan)
        rethrow(err)
    end

    plan
end

function distributed_feast!(
    X::AbstractMatrix,
    A::AbstractMatrix;
    nodes::Integer=8,
    iter::Integer=10,
    c=complex(0.0, 0.0),
    r=1.0,
    ϵ=1e-12,
    debug=false,
    store=false,
    worker_ids=_default_feast_worker_ids(),
    worker_blas_threads::Integer=1,
    stats::Union{Nothing, DenseDistributedFeastStats}=nothing,
)
    contour = circular_contour_trapezoidal(c, r, nodes)
    distributed_feast!(
        X,
        A,
        contour;
        iter=iter,
        ϵ=ϵ,
        debug=debug,
        store=store,
        worker_ids=worker_ids,
        worker_blas_threads=worker_blas_threads,
        stats=stats,
    )
end

function distributed_feast!(
    X::AbstractMatrix,
    A::AbstractMatrix,
    contour::Contour;
    iter::Integer=10,
    ϵ=1e-12,
    debug=false,
    store=false,
    worker_ids=_default_feast_worker_ids(),
    worker_blas_threads::Integer=1,
    stats::Union{Nothing, DenseDistributedFeastStats}=nothing,
)
    plan = DenseDistributedFeastPlan(
        A,
        size(X, 2),
        contour;
        store=store,
        worker_ids=worker_ids,
        worker_blas_threads=worker_blas_threads,
        stats=stats,
    )
    try
        distributed_feast!(X, plan; iter=iter, ϵ=ϵ, debug=debug, stats=stats)
    finally
        close(plan)
    end
end

function distributed_feast!(
    X::AbstractMatrix,
    plan::DenseDistributedFeastPlan;
    iter::Integer=10,
    ϵ=1e-12,
    debug=false,
    stats::Union{Nothing, DenseDistributedFeastStats}=nothing,
)
    plan.closed && error("DenseDistributedFeastPlan is closed")
    stats = stats === nothing ? plan.stats : stats
    N, m₀ = size(plan.X_shared)
    size(X) == (N, m₀) || error("Incorrect dimensions of X, must match planned A and subspace dimension")

    A = plan.A
    contour = plan.contour
    Λ = plan.Λ
    res = plan.res
    inside = plan.inside
    R = plan.R
    Q = plan.Q
    Aq = plan.Aq
    Xq = plan.Xq

    solve_start_ns = time_ns()
    copyto!(Q, X)
    for nit=0:iter
        iter_start_ns = time_ns()
        qr_ns = UInt64(0)
        rayleigh_ritz_ns = UInt64(0)
        residual_ns = UInt64(0)
        shared_copy_ns = UInt64(0)
        worker_step_ns = UInt64(0)
        reduce_ns = UInt64(0)

        if stats !== nothing
            stats.iterations += 1
        end
        start_ns = time_ns()
        dense_lapack_qr!(Q, plan.qr_ws)
        qr_ns = time_ns() - start_ns
        _add_ns!(stats, :qr_ns, qr_ns)

        start_ns = time_ns()
        mul!(R, A, Q)
        mul!(Aq, Q', R)
        dense_lapack_eigen!(Λ, Xq, Aq, plan.eigen_ws)
        rayleigh_ritz_ns = time_ns() - start_ns
        _add_ns!(stats, :rayleigh_ritz_ns, rayleigh_ritz_ns)

        start_ns = time_ns()
        mul!(X, Q, Xq)
        update_R!(X, R, Λ, A)
        residuals!(res, R, Λ, A)
        in_contour!(inside, Λ, contour)
        max_res_inside, contour_nonempty = maximum_masked(res, inside)
        residual_ns = time_ns() - start_ns
        _add_ns!(stats, :residual_ns, residual_ns)

        if debug
            iter_debug_print(nit, Λ, res, contour, 1e-5)
        end
        converged = contour_nonempty && max_res_inside < ϵ
        if converged
            if debug
                println("converged in $nit iteration")
            end
        end
        if !converged && nit < iter
            start_ns = time_ns()
            copyto!(plan.X_shared, X)
            copyto!(plan.R_shared, R)
            shared_copy_ns = time_ns() - start_ns
            _add_ns!(stats, :shared_copy_ns, shared_copy_ns)

            start_ns = time_ns()
            for (i, pid) in enumerate(plan.worker_ids)
                plan.futures[i] = remotecall(_dense_feast_worker_step!, pid, plan.key, Λ)
            end
            for future in plan.futures
                fetch(future)
            end
            worker_step_ns = time_ns() - start_ns
            _add_ns!(stats, :worker_step_ns, worker_step_ns)

            start_ns = time_ns()
            _sum_dense_feast_qparts!(Q, plan.Qparts)
            reduce_ns = time_ns() - start_ns
            _add_ns!(stats, :reduce_ns, reduce_ns)
        end

        _record_distributed_feast_iteration!(
            stats,
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

        if converged
            break
        end
    end
    _add_elapsed!(stats, :solve_total_ns, solve_start_ns)

    in_contour!(inside, Λ, contour)
    _, contour_nonempty = maximum_masked(res, inside)
    if !contour_nonempty
        println("no eigenvalues found in contour!")
    end
    Λ[inside], X[:, inside], res[inside]
end

function distributed_gen_feast!(
    X::AbstractMatrix,
    A::AbstractMatrix,
    B;
    nodes::Integer=8,
    iter::Integer=10,
    c=complex(0.0, 0.0),
    r=1.0,
    ϵ=1e-12,
    debug=false,
    store=false,
    worker_ids=_default_feast_worker_ids(),
    worker_blas_threads::Integer=1,
    stats::Union{Nothing, DenseDistributedFeastStats}=nothing,
)
    contour = circular_contour_trapezoidal(c, r, nodes)
    distributed_gen_feast!(
        X,
        A,
        B,
        contour;
        iter=iter,
        ϵ=ϵ,
        debug=debug,
        store=store,
        worker_ids=worker_ids,
        worker_blas_threads=worker_blas_threads,
        stats=stats,
    )
end

function distributed_gen_feast!(
    X::AbstractMatrix,
    A::AbstractMatrix,
    B,
    contour::Contour;
    iter::Integer=10,
    ϵ=1e-12,
    debug=false,
    store=false,
    worker_ids=_default_feast_worker_ids(),
    worker_blas_threads::Integer=1,
    stats::Union{Nothing, DenseDistributedFeastStats}=nothing,
)
    plan = DenseDistributedGeneralizedFeastPlan(
        A,
        B,
        size(X, 2),
        contour;
        store=store,
        worker_ids=worker_ids,
        worker_blas_threads=worker_blas_threads,
        stats=stats,
    )
    try
        distributed_gen_feast!(X, plan; iter=iter, ϵ=ϵ, debug=debug, stats=stats)
    finally
        close(plan)
    end
end

function distributed_gen_feast!(
    X::AbstractMatrix,
    plan::DenseDistributedGeneralizedFeastPlan;
    iter::Integer=10,
    ϵ=1e-12,
    debug=false,
    stats::Union{Nothing, DenseDistributedFeastStats}=nothing,
)
    plan.closed && error("DenseDistributedGeneralizedFeastPlan is closed")
    stats = stats === nothing ? plan.stats : stats
    N, m₀ = size(plan.X_shared)
    size(X) == (N, m₀) || error("Incorrect dimensions of X, must match planned A and subspace dimension")

    A = plan.A
    B = plan.B
    contour = plan.contour
    Λ = plan.Λ
    res = plan.res
    inside = plan.inside
    R = plan.R
    Q = plan.Q
    Aq = plan.Aq
    Bq = plan.Bq
    Xq = plan.Xq

    solve_start_ns = time_ns()
    copyto!(Q, X)
    for nit=0:iter
        iter_start_ns = time_ns()
        qr_ns = UInt64(0)
        rayleigh_ritz_ns = UInt64(0)
        residual_ns = UInt64(0)
        shared_copy_ns = UInt64(0)
        worker_step_ns = UInt64(0)
        reduce_ns = UInt64(0)

        if stats !== nothing
            stats.iterations += 1
        end
        start_ns = time_ns()
        dense_lapack_qr!(Q, plan.qr_ws)
        qr_ns = time_ns() - start_ns
        _add_ns!(stats, :qr_ns, qr_ns)

        start_ns = time_ns()
        mul!(R, A, Q)
        mul!(Aq, Q', R)
        mul!(R, B, Q)
        mul!(Bq, Q', R)
        dense_lapack_generalized_eigen!(Λ, Xq, Aq, Bq, plan.eigen_ws)
        rayleigh_ritz_ns = time_ns() - start_ns
        _add_ns!(stats, :rayleigh_ritz_ns, rayleigh_ritz_ns)

        start_ns = time_ns()
        mul!(X, Q, Xq)
        update_R!(X, R, Λ, A, B, plan.Q)
        residuals!(res, R, Λ, A)
        in_contour!(inside, Λ, contour)
        max_res_inside, contour_nonempty = maximum_masked(res, inside)
        residual_ns = time_ns() - start_ns
        _add_ns!(stats, :residual_ns, residual_ns)

        if debug
            iter_debug_print(nit, Λ, res, contour, 1e-5)
        end
        converged = contour_nonempty && max_res_inside < ϵ
        if converged && debug
            println("converged in $nit iteration")
        end
        if !converged && nit < iter
            start_ns = time_ns()
            copyto!(plan.X_shared, X)
            copyto!(plan.R_shared, R)
            shared_copy_ns = time_ns() - start_ns
            _add_ns!(stats, :shared_copy_ns, shared_copy_ns)

            start_ns = time_ns()
            for (i, pid) in enumerate(plan.worker_ids)
                plan.futures[i] = remotecall(_dense_gen_feast_worker_step!, pid, plan.key, Λ)
            end
            for future in plan.futures
                fetch(future)
            end
            worker_step_ns = time_ns() - start_ns
            _add_ns!(stats, :worker_step_ns, worker_step_ns)

            start_ns = time_ns()
            _sum_dense_feast_qparts!(Q, plan.Qparts)
            reduce_ns = time_ns() - start_ns
            _add_ns!(stats, :reduce_ns, reduce_ns)
        end

        _record_distributed_feast_iteration!(
            stats,
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

        if converged
            break
        end
    end
    _add_elapsed!(stats, :solve_total_ns, solve_start_ns)

    in_contour!(inside, Λ, contour)
    _, contour_nonempty = maximum_masked(res, inside)
    if !contour_nonempty
        println("no eigenvalues found in contour!")
    end
    Λ[inside], X[:, inside], res[inside]
end

function distributed_dual_gen_feast!(
    Xr::AbstractMatrix,
    Xl::AbstractMatrix,
    A::AbstractMatrix,
    B;
    nodes::Integer=8,
    iter::Integer=10,
    c=complex(0.0, 0.0),
    r=1.0,
    ϵ=1e-12,
    debug=false,
    store=false,
    worker_ids=_default_feast_worker_ids(),
    worker_blas_threads::Integer=1,
    stats::Union{Nothing, DenseDistributedFeastStats}=nothing,
)
    contour = circular_contour_trapezoidal(c, r, nodes)
    distributed_dual_gen_feast!(
        Xr,
        Xl,
        A,
        B,
        contour;
        iter=iter,
        ϵ=ϵ,
        debug=debug,
        store=store,
        worker_ids=worker_ids,
        worker_blas_threads=worker_blas_threads,
        stats=stats,
    )
end

function distributed_dual_gen_feast!(
    Xr::AbstractMatrix,
    Xl::AbstractMatrix,
    A::AbstractMatrix,
    B,
    contour::Contour;
    iter::Integer=10,
    ϵ=1e-12,
    debug=false,
    store=false,
    worker_ids=_default_feast_worker_ids(),
    worker_blas_threads::Integer=1,
    stats::Union{Nothing, DenseDistributedFeastStats}=nothing,
)
    plan = DenseDistributedDualGeneralizedFeastPlan(
        A,
        B,
        size(Xr, 2),
        contour;
        store=store,
        worker_ids=worker_ids,
        worker_blas_threads=worker_blas_threads,
        stats=stats,
    )
    try
        distributed_dual_gen_feast!(Xr, Xl, plan; iter=iter, ϵ=ϵ, debug=debug, stats=stats)
    finally
        close(plan)
    end
end

function distributed_dual_gen_feast!(
    Xr::AbstractMatrix,
    Xl::AbstractMatrix,
    plan::DenseDistributedDualGeneralizedFeastPlan;
    iter::Integer=10,
    ϵ=1e-12,
    debug=false,
    stats::Union{Nothing, DenseDistributedFeastStats}=nothing,
)
    plan.closed && error("DenseDistributedDualGeneralizedFeastPlan is closed")
    stats = stats === nothing ? plan.stats : stats
    N, m₀ = size(plan.Xr_shared)
    size(Xr) == (N, m₀) || error("Incorrect dimensions of Xr, must match planned A and subspace dimension")
    size(Xl) == (N, m₀) || error("Incorrect dimensions of Xl, must match planned A and subspace dimension")

    A = plan.A
    B = plan.B
    contour = plan.contour
    Λ = plan.Λ
    resr = plan.resr
    inside = plan.inside
    Rr = plan.Rr
    Rl = plan.Rl
    Qr = plan.Qr
    Ql = plan.Ql
    Aq = plan.Aq
    Bq = plan.Bq
    Xql = plan.Xql
    Xqr = plan.Xqr

    solve_start_ns = time_ns()
    copyto!(Qr, Xr)
    copyto!(Ql, Xl)
    for nit=0:iter
        iter_start_ns = time_ns()
        qr_ns = UInt64(0)
        rayleigh_ritz_ns = UInt64(0)
        residual_ns = UInt64(0)
        shared_copy_ns = UInt64(0)
        worker_step_ns = UInt64(0)
        reduce_ns = UInt64(0)

        if stats !== nothing
            stats.iterations += 1
        end
        start_ns = time_ns()
        mul!(Rr, B, Qr)
        mul!(Bq, Ql', Rr)
        U, S, Vt = dense_lapack_svd!(Bq, plan.svd_ws)
        mul!(Rr, Qr, adjoint(Vt))
        Qr .= Rr
        inv_scale_columns!(Qr, S)
        mul!(Rl, Ql, U)
        Ql .= Rl
        inv_scale_columns!(Ql, S)
        qr_ns = time_ns() - start_ns
        _add_ns!(stats, :qr_ns, qr_ns)

        start_ns = time_ns()
        mul!(Rr, A, Qr)
        mul!(Aq, Ql', Rr)
        mul!(Rr, B, Qr)
        mul!(Bq, Ql', Rr)
        dense_lapack_generalized_eigen!(Λ, Xql, Xqr, Aq, Bq, plan.eigen_ws)
        mul!(Xr, Qr, Xqr)
        mul!(Xl, Ql, Xql)
        rayleigh_ritz_ns = time_ns() - start_ns
        _add_ns!(stats, :rayleigh_ritz_ns, rayleigh_ritz_ns)

        start_ns = time_ns()
        update_R_shifted!(Xr, Rr, Λ, A, B, plan.residual_shift, plan.residual_x, plan.residual_y)
        update_R_shifted!(Xl, Rl, Λ, A', B', plan.residual_shift, plan.residual_x, plan.residual_y)
        residuals!(resr, Rr, Λ, A)
        in_contour!(inside, Λ, contour)
        max_res_inside, contour_nonempty = maximum_masked(resr, inside)
        residual_ns = time_ns() - start_ns
        _add_ns!(stats, :residual_ns, residual_ns)

        if debug
            iter_debug_print(nit, Λ, resr, contour, 1e-5)
        end
        converged = contour_nonempty && max_res_inside < ϵ
        if converged && debug
            println("converged in $nit iteration")
        end
        if !converged && nit < iter
            start_ns = time_ns()
            copyto!(plan.Xr_shared, Xr)
            copyto!(plan.Xl_shared, Xl)
            copyto!(plan.Rr_shared, Rr)
            copyto!(plan.Rl_shared, Rl)
            shared_copy_ns = time_ns() - start_ns
            _add_ns!(stats, :shared_copy_ns, shared_copy_ns)

            start_ns = time_ns()
            for (i, pid) in enumerate(plan.worker_ids)
                plan.futures[i] = remotecall(_dense_dual_gen_feast_worker_step!, pid, plan.key, Λ)
            end
            for future in plan.futures
                fetch(future)
            end
            worker_step_ns = time_ns() - start_ns
            _add_ns!(stats, :worker_step_ns, worker_step_ns)

            start_ns = time_ns()
            _sum_dense_feast_qparts!(Qr, plan.Qrparts)
            _sum_dense_feast_qparts!(Ql, plan.Qlparts)
            reduce_ns = time_ns() - start_ns
            _add_ns!(stats, :reduce_ns, reduce_ns)
        end

        _record_distributed_feast_iteration!(
            stats,
            nit,
            resr,
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

        if converged
            break
        end
    end
    _add_elapsed!(stats, :solve_total_ns, solve_start_ns)

    in_contour!(inside, Λ, contour)
    _, contour_nonempty = maximum_masked(resr, inside)
    if !contour_nonempty
        println("no eigenvalues found in contour!")
    end
    Λ[inside], Xr[:, inside], Xl[:, inside], resr[inside]
end

_default_feast_worker_ids() = _normalize_feast_worker_ids(workers())

function _normalize_feast_worker_ids(worker_ids)
    ids = unique(collect(worker_ids))
    filter!(!=(myid()), ids)
    ids
end

function _shared_complex_matrix(A::AbstractMatrix, dims, all_pids)
    size(A) == dims || error("matrix dimensions must match A")
    shared = SharedArray{ComplexF64}(dims; pids=all_pids)
    copyto!(shared, A)
    shared
end

function _shared_complex_matrix(B::UniformScaling, dims, all_pids)
    shared = SharedArray{ComplexF64}(dims; pids=all_pids)
    fill!(shared, 0)
    n = min(dims...)
    @inbounds for i in 1:n
        shared[i, i] = B.λ
    end
    shared
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

function _prepare_dense_feast_workers!(worker_ids)
    for pid in worker_ids
        remotecall_wait(Main.eval, pid, :(using SharedArrays; using FEASTSolver))
    end
    nothing
end

function _dense_feast_worker_assignments(worker_ids::Vector{Int}, node_count::Int)
    active_count = min(length(worker_ids), node_count)
    active_workers = worker_ids[1:active_count]
    assignments = [Int[] for _ in active_workers]
    for node_index in 1:node_count
        push!(assignments[mod1(node_index, active_count)], node_index)
    end
    active_workers, assignments
end

function _init_dense_feast_workers!(plan::DenseDistributedFeastPlan)
    nodes = ComplexF64.(plan.contour.nodes)
    weights = ComplexF64.(plan.contour.weights)
    for (part_index, pid) in enumerate(plan.worker_ids)
        plan.futures[part_index] = remotecall(
            _init_dense_feast_worker!,
            pid,
            plan.key,
            plan.A,
            plan.X_shared,
            plan.R_shared,
            plan.Qparts,
            part_index,
            plan.assignments[part_index],
            nodes,
            weights,
            plan.store,
            plan.worker_blas_threads,
        )
    end
    for future in plan.futures
        fetch(future)
    end
    nothing
end

function _init_dense_gen_feast_workers!(plan::DenseDistributedGeneralizedFeastPlan)
    nodes = ComplexF64.(plan.contour.nodes)
    weights = ComplexF64.(plan.contour.weights)
    for (part_index, pid) in enumerate(plan.worker_ids)
        plan.futures[part_index] = remotecall(
            _init_dense_gen_feast_worker!,
            pid,
            plan.key,
            plan.A,
            plan.B,
            plan.X_shared,
            plan.R_shared,
            plan.Qparts,
            part_index,
            plan.assignments[part_index],
            nodes,
            weights,
            plan.store,
            plan.worker_blas_threads,
        )
    end
    for future in plan.futures
        fetch(future)
    end
    nothing
end

function _init_dense_dual_gen_feast_workers!(plan::DenseDistributedDualGeneralizedFeastPlan)
    nodes = ComplexF64.(plan.contour.nodes)
    weights = ComplexF64.(plan.contour.weights)
    for (part_index, pid) in enumerate(plan.worker_ids)
        plan.futures[part_index] = remotecall(
            _init_dense_dual_gen_feast_worker!,
            pid,
            plan.key,
            plan.A,
            plan.B,
            plan.Xr_shared,
            plan.Xl_shared,
            plan.Rr_shared,
            plan.Rl_shared,
            plan.Qrparts,
            plan.Qlparts,
            part_index,
            plan.assignments[part_index],
            nodes,
            weights,
            plan.store,
            plan.worker_blas_threads,
        )
    end
    for future in plan.futures
        fetch(future)
    end
    nothing
end

function _init_dense_feast_worker!(
    key::Symbol,
    A,
    X,
    R,
    Qparts,
    part_index::Int,
    node_indices::Vector{Int},
    nodes::Vector{ComplexF64},
    weights::Vector{ComplexF64},
    store::Bool,
    worker_blas_threads::Int,
)
    old_blas_threads = BLAS.get_num_threads()
    BLAS.set_num_threads(worker_blas_threads)

    N, m₀ = size(X)
    ZmA = zeros(ComplexF64, size(A))
    temp = zeros(ComplexF64, N, m₀)
    resolvent = zeros(ComplexF64, m₀)
    lu_ws = dense_lapack_lu_workspace(ZmA)

    stored_shifts = nothing
    stored_lu_ws = nothing
    if store
        stored_shifts = [zeros(ComplexF64, size(A)) for _ in node_indices]
        stored_lu_ws = [dense_lapack_lu_workspace(stored_shifts[i]) for i in eachindex(node_indices)]
        for (local_index, node_index) in pairs(node_indices)
            materialize_standard_shift!(stored_shifts[local_index], A, nodes[node_index])
            dense_lapack_factor!(stored_shifts[local_index], stored_lu_ws[local_index])
        end
    end

    _DISTRIBUTED_DENSE_FEAST_WORKSPACES[key] = DenseFeastDistributedWorkspace(
        A,
        X,
        R,
        Qparts,
        part_index,
        node_indices,
        nodes,
        weights,
        temp,
        ZmA,
        resolvent,
        lu_ws,
        stored_shifts,
        stored_lu_ws,
        old_blas_threads,
    )
    nothing
end

function _init_dense_gen_feast_worker!(
    key::Symbol,
    A,
    B,
    X,
    R,
    Qparts,
    part_index::Int,
    node_indices::Vector{Int},
    nodes::Vector{ComplexF64},
    weights::Vector{ComplexF64},
    store::Bool,
    worker_blas_threads::Int,
)
    old_blas_threads = BLAS.get_num_threads()
    BLAS.set_num_threads(worker_blas_threads)

    N, m₀ = size(X)
    ZmA = zeros(ComplexF64, size(A))
    temp = zeros(ComplexF64, N, m₀)
    resolvent = zeros(ComplexF64, m₀)
    lu_ws = dense_lapack_lu_workspace(ZmA)

    stored_shifts = nothing
    stored_lu_ws = nothing
    if store
        stored_shifts = [zeros(ComplexF64, size(A)) for _ in node_indices]
        stored_lu_ws = [dense_lapack_lu_workspace(stored_shifts[i]) for i in eachindex(node_indices)]
        for (local_index, node_index) in pairs(node_indices)
            materialize_generalized_shift!(stored_shifts[local_index], A, B, nodes[node_index])
            dense_lapack_factor!(stored_shifts[local_index], stored_lu_ws[local_index])
        end
    end

    _DISTRIBUTED_DENSE_FEAST_WORKSPACES[key] = DenseGeneralizedFeastDistributedWorkspace(
        A,
        B,
        X,
        R,
        Qparts,
        part_index,
        node_indices,
        nodes,
        weights,
        temp,
        ZmA,
        resolvent,
        lu_ws,
        stored_shifts,
        stored_lu_ws,
        old_blas_threads,
    )
    nothing
end

function _init_dense_dual_gen_feast_worker!(
    key::Symbol,
    A,
    B,
    Xr,
    Xl,
    Rr,
    Rl,
    Qrparts,
    Qlparts,
    part_index::Int,
    node_indices::Vector{Int},
    nodes::Vector{ComplexF64},
    weights::Vector{ComplexF64},
    store::Bool,
    worker_blas_threads::Int,
)
    old_blas_threads = BLAS.get_num_threads()
    BLAS.set_num_threads(worker_blas_threads)

    N, m₀ = size(Xr)
    ZmA = zeros(ComplexF64, size(A))
    temp = zeros(ComplexF64, N, m₀)
    resolvent = zeros(ComplexF64, m₀)
    lu_ws = dense_lapack_lu_workspace(ZmA)

    stored_shifts = nothing
    stored_lu_ws = nothing
    if store
        stored_shifts = [zeros(ComplexF64, size(A)) for _ in node_indices]
        stored_lu_ws = [dense_lapack_lu_workspace(stored_shifts[i]) for i in eachindex(node_indices)]
        for (local_index, node_index) in pairs(node_indices)
            materialize_generalized_shift!(stored_shifts[local_index], A, B, nodes[node_index])
            dense_lapack_factor!(stored_shifts[local_index], stored_lu_ws[local_index])
        end
    end

    _DISTRIBUTED_DENSE_FEAST_WORKSPACES[key] = DenseDualGeneralizedFeastDistributedWorkspace(
        A,
        B,
        Xr,
        Xl,
        Rr,
        Rl,
        Qrparts,
        Qlparts,
        part_index,
        node_indices,
        nodes,
        weights,
        temp,
        ZmA,
        resolvent,
        lu_ws,
        stored_shifts,
        stored_lu_ws,
        old_blas_threads,
    )
    nothing
end

function _dense_feast_worker_step!(key::Symbol, Λ::Vector{ComplexF64})
    ws = _DISTRIBUTED_DENSE_FEAST_WORKSPACES[key]
    Qpart = view(ws.Qparts, :, :, ws.part_index)
    fill!(Qpart, 0)

    for (local_index, node_index) in pairs(ws.node_indices)
        fill_resolvent!(ws.resolvent, ws.nodes[node_index], Λ)
        if ws.stored_shifts === nothing
            materialize_standard_shift!(ws.ZmA, ws.A, ws.nodes[node_index])
            dense_lapack_linsolve!(ws.temp, ws.ZmA, ws.R, ws.lu_ws)
        else
            dense_lapack_solve_factored!(
                ws.temp,
                ws.stored_shifts[local_index],
                ws.R,
                ws.stored_lu_ws[local_index],
            )
        end

        @inbounds for j in axes(ws.temp, 2)
            α = ws.resolvent[j] * ws.weights[node_index]
            for i in axes(ws.temp, 1)
                Qpart[i, j] += (ws.X[i, j] - ws.temp[i, j]) * α
            end
        end
    end
    nothing
end

function _dense_gen_feast_worker_step!(key::Symbol, Λ::Vector{ComplexF64})
    ws = _DISTRIBUTED_DENSE_FEAST_WORKSPACES[key]
    Qpart = view(ws.Qparts, :, :, ws.part_index)
    fill!(Qpart, 0)

    for (local_index, node_index) in pairs(ws.node_indices)
        fill_resolvent!(ws.resolvent, ws.nodes[node_index], Λ)
        if ws.stored_shifts === nothing
            materialize_generalized_shift!(ws.ZmA, ws.A, ws.B, ws.nodes[node_index])
            dense_lapack_linsolve!(ws.temp, ws.ZmA, ws.R, ws.lu_ws)
        else
            dense_lapack_solve_factored!(
                ws.temp,
                ws.stored_shifts[local_index],
                ws.R,
                ws.stored_lu_ws[local_index],
            )
        end

        @inbounds for j in axes(ws.temp, 2)
            α = ws.resolvent[j] * ws.weights[node_index]
            for i in axes(ws.temp, 1)
                Qpart[i, j] += (ws.X[i, j] - ws.temp[i, j]) * α
            end
        end
    end
    nothing
end

function _dense_dual_gen_feast_worker_step!(key::Symbol, Λ::Vector{ComplexF64})
    ws = _DISTRIBUTED_DENSE_FEAST_WORKSPACES[key]
    Qrpart = view(ws.Qrparts, :, :, ws.part_index)
    Qlpart = view(ws.Qlparts, :, :, ws.part_index)
    fill!(Qrpart, 0)
    fill!(Qlpart, 0)

    for (local_index, node_index) in pairs(ws.node_indices)
        if ws.stored_shifts === nothing
            materialize_generalized_shift!(ws.ZmA, ws.A, ws.B, ws.nodes[node_index])
            dense_lapack_factor!(ws.ZmA, ws.lu_ws)
            fill_resolvent!(ws.resolvent, ws.nodes[node_index], Λ)
            dense_lapack_solve_factored!(ws.temp, ws.ZmA, ws.Rr, ws.lu_ws, 'N')
        else
            fill_resolvent!(ws.resolvent, ws.nodes[node_index], Λ)
            dense_lapack_solve_factored!(
                ws.temp,
                ws.stored_shifts[local_index],
                ws.Rr,
                ws.stored_lu_ws[local_index],
                'N',
            )
        end

        @inbounds for j in axes(ws.temp, 2)
            α = ws.resolvent[j] * ws.weights[node_index]
            for i in axes(ws.temp, 1)
                Qrpart[i, j] += (ws.Xr[i, j] - ws.temp[i, j]) * α
            end
        end

        fill_adjoint_resolvent!(ws.resolvent, ws.nodes[node_index], Λ)
        if ws.stored_shifts === nothing
            dense_lapack_solve_factored!(ws.temp, ws.ZmA, ws.Rl, ws.lu_ws, 'C')
        else
            dense_lapack_solve_factored!(
                ws.temp,
                ws.stored_shifts[local_index],
                ws.Rl,
                ws.stored_lu_ws[local_index],
                'C',
            )
        end

        @inbounds for j in axes(ws.temp, 2)
            α = ws.resolvent[j] * conj(ws.weights[node_index])
            for i in axes(ws.temp, 1)
                Qlpart[i, j] += (ws.Xl[i, j] - ws.temp[i, j]) * α
            end
        end
    end
    nothing
end

function _sum_dense_feast_qparts!(Q::AbstractMatrix, Qparts)
    fill!(Q, 0)
    @inbounds for k in axes(Qparts, 3), j in axes(Qparts, 2), i in axes(Qparts, 1)
        Q[i, j] += Qparts[i, j, k]
    end
    Q
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

function _cleanup_dense_feast_worker!(key::Symbol)
    ws = pop!(_DISTRIBUTED_DENSE_FEAST_WORKSPACES, key, nothing)
    if ws !== nothing
        BLAS.set_num_threads(ws.old_blas_threads)
    end
    nothing
end

function close(plan::AbstractDenseDistributedFeastPlan)
    plan.closed && return nothing
    for pid in plan.worker_ids
        try
            remotecall_wait(_cleanup_dense_feast_worker!, pid, plan.key)
        catch err
            @warn "failed to clean up distributed FEAST worker state" pid exception=(err, catch_backtrace())
        end
    end
    plan.closed = true
    nothing
end
