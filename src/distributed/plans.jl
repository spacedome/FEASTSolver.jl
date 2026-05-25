abstract type AbstractDenseDistributedFeastPlan end

"""
    DenseDistributedFeastPlan(A, m0, [contour]; kwargs...)

Persistent contour-worker plan for dense standard FEAST. The plan owns shared
buffers, fixed worker-to-node assignments, and optional worker-local LU factors.
"""
mutable struct DenseDistributedFeastPlan <: AbstractDenseDistributedFeastPlan
    key::Symbol
    A
    X_buffer
    R_buffer
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
    X_buffer
    R_buffer
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
    Xr_buffer
    Xl_buffer
    Rr_buffer
    Rl_buffer
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

function DenseDistributedFeastPlan(
    A::AbstractMatrix,
    m₀::Integer;
    nodes::Integer=8,
    c=complex(0.0, 0.0),
    r=1.0,
    contour=nothing,
    store=false,
    worker_ids=_default_feast_worker_ids(),
    worker_blas_threads::Integer=1,
    stats::Union{Nothing, DenseDistributedFeastStats}=nothing,
)
    contour = contour === nothing ? circular_contour_trapezoidal(c, r, nodes) : contour
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
    start_ns = time_ns()
    _prepare_dense_feast_workers!(worker_ids)
    _add_elapsed!(stats, :setup_prepare_ns, start_ns)

    start_ns = time_ns()
    N = size(A, 1)
    A_local = _local_complex_matrix(A, (N, N))
    X_buffer = zeros(ComplexF64, N, m₀)
    R_buffer = zeros(ComplexF64, N, m₀)
    Qparts = [zeros(ComplexF64, N, m₀) for _ in worker_ids]

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
    _add_elapsed!(stats, :setup_master_ns, start_ns)

    plan = DenseDistributedFeastPlan(
        key,
        A_local,
        X_buffer,
        R_buffer,
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
    contour=nothing,
    store=false,
    worker_ids=_default_feast_worker_ids(),
    worker_blas_threads::Integer=1,
    stats::Union{Nothing, DenseDistributedFeastStats}=nothing,
)
    contour = contour === nothing ? circular_contour_trapezoidal(c, r, nodes) : contour
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
    start_ns = time_ns()
    _prepare_dense_feast_workers!(worker_ids)
    _add_elapsed!(stats, :setup_prepare_ns, start_ns)

    start_ns = time_ns()
    N = size(A, 1)
    A_shared = _local_complex_matrix(A, (N, N))
    B_shared = _local_complex_matrix(B, (N, N))
    X_buffer = zeros(ComplexF64, N, m₀)
    R_buffer = zeros(ComplexF64, N, m₀)
    Qparts = [zeros(ComplexF64, N, m₀) for _ in worker_ids]

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
    _add_elapsed!(stats, :setup_master_ns, start_ns)

    plan = DenseDistributedGeneralizedFeastPlan(
        key,
        A_shared,
        B_shared,
        X_buffer,
        R_buffer,
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
    contour=nothing,
    store=false,
    worker_ids=_default_feast_worker_ids(),
    worker_blas_threads::Integer=1,
    stats::Union{Nothing, DenseDistributedFeastStats}=nothing,
)
    contour = contour === nothing ? circular_contour_trapezoidal(c, r, nodes) : contour
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
    start_ns = time_ns()
    _prepare_dense_feast_workers!(worker_ids)
    _add_elapsed!(stats, :setup_prepare_ns, start_ns)

    start_ns = time_ns()
    N = size(A, 1)
    A_shared = _local_complex_matrix(A, (N, N))
    B_shared = _local_complex_matrix(B, (N, N))
    Xr_buffer = zeros(ComplexF64, N, m₀)
    Xl_buffer = zeros(ComplexF64, N, m₀)
    Rr_buffer = zeros(ComplexF64, N, m₀)
    Rl_buffer = zeros(ComplexF64, N, m₀)
    Qrparts = [zeros(ComplexF64, N, m₀) for _ in worker_ids]
    Qlparts = [zeros(ComplexF64, N, m₀) for _ in worker_ids]

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
    _add_elapsed!(stats, :setup_master_ns, start_ns)

    plan = DenseDistributedDualGeneralizedFeastPlan(
        key,
        A_shared,
        B_shared,
        Xr_buffer,
        Xl_buffer,
        Rr_buffer,
        Rl_buffer,
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
