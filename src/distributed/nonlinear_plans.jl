# Distributed canonical nonlinear FEAST plan.
#
# Each worker owns a stable subset of contour nodes, plus optional node-local
# factorizations. The master process owns Beyn compression and residual checks.

"""
    DenseDistributedNonlinearFeastPlan(T, n, m0, [contour]; kwargs...)

Persistent contour-worker plan for the canonical nonlinear FEAST prototype.
`T` is serialized to each worker once at setup, so captured problem data remain
worker-local across iterations.
"""
mutable struct DenseDistributedNonlinearFeastPlan <: AbstractDenseDistributedFeastPlan
    key::Symbol
    T
    X_buffer
    R_buffer
    Q₀parts
    Q₁parts
    contour::Contour
    worker_ids::Vector{Int}
    assignments::Vector{Vector{Int}}
    store::Bool
    materialize_nodes::Bool
    matrix_update
    matrix_prototype
    worker_blas_threads::Int
    factorizer
    left_divider
    Λ::Vector{ComplexF64}
    res::Vector{Float64}
    inside::BitVector
    R::Matrix{ComplexF64}
    Q₀::Matrix{ComplexF64}
    Q₁::Matrix{ComplexF64}
    A::Matrix{ComplexF64}
    B::Matrix{ComplexF64}
    Xq::Matrix{ComplexF64}
    qr_ws
    svd_ws
    eigen_ws
    residual_matrix
    residual_x::Vector{ComplexF64}
    residual_y::Vector{ComplexF64}
    futures::Vector{Any}
    stats::Union{Nothing, DenseDistributedFeastStats}
    closed::Bool
end

function DenseDistributedNonlinearFeastPlan(
    T,
    n::Integer,
    m₀::Integer;
    nodes::Integer=8,
    c=complex(0.0, 0.0),
    r=1.0,
    contour=nothing,
    kwargs...,
)
    contour = contour === nothing ? circular_contour_trapezoidal(c, r, nodes) : contour
    DenseDistributedNonlinearFeastPlan(T, n, m₀, contour; kwargs...)
end

function DenseDistributedNonlinearFeastPlan(
    T,
    n::Integer,
    m₀::Integer,
    contour::Contour;
    store=true,
    materialize_nodes=true,
    matrix_update=nothing,
    matrix_prototype=nothing,
    worker_ids=_default_feast_worker_ids(),
    worker_blas_threads::Integer=1,
    factorizer=lu,
    left_divider=ldiv!,
    stats::Union{Nothing, DenseDistributedFeastStats}=nothing,
)
    n > 0 || error("problem dimension must be positive")
    m₀ > 0 || error("subspace dimension must be positive")
    matrix_update, matrix_prototype = _nlfeast_operator_storage(T, matrix_update, matrix_prototype)

    worker_ids = _normalize_feast_worker_ids(worker_ids)
    isempty(worker_ids) && error("DenseDistributedNonlinearFeastPlan requires worker processes; call addprocs(...) or start Julia with -p")

    node_count = size(contour.nodes, 1)
    worker_ids, assignments = _dense_feast_worker_assignments(worker_ids, node_count)
    start_ns = time_ns()
    _prepare_dense_feast_workers!(worker_ids)
    _add_elapsed!(stats, :setup_prepare_ns, start_ns)

    start_ns = time_ns()
    N = Int(n)
    X_buffer = zeros(ComplexF64, N, m₀)
    R_buffer = zeros(ComplexF64, N, m₀)
    Q₀parts = [zeros(ComplexF64, N, m₀) for _ in worker_ids]
    Q₁parts = [zeros(ComplexF64, N, m₀) for _ in worker_ids]

    Λ = zeros(ComplexF64, m₀)
    res = zeros(m₀)
    inside = falses(m₀)
    R = zeros(ComplexF64, N, m₀)
    Q₀ = zeros(ComplexF64, N, m₀)
    Q₁ = zeros(ComplexF64, N, m₀)
    A = zeros(ComplexF64, m₀, m₀)
    B = zeros(ComplexF64, m₀, m₀)
    Xq = zeros(ComplexF64, m₀, m₀)
    qr_ws = QRWs(Q₀)
    svd_ws = SVDsddWs(Q₀, job='S')
    eigen_ws = EigenWs(A, rvecs=true)

    # Avoid materializing T(z) during plan construction. Large dense NEPs can
    # spend seconds here, and action residual hooks do not need a matrix buffer.
    residual_matrix = nothing
    residual_x = zeros(ComplexF64, N)
    residual_y = zeros(ComplexF64, N)
    key = gensym(:dense_nlfeast)
    futures = Vector{Any}(undef, length(worker_ids))
    _add_elapsed!(stats, :setup_master_ns, start_ns)

    plan = DenseDistributedNonlinearFeastPlan(
        key,
        T,
        X_buffer,
        R_buffer,
        Q₀parts,
        Q₁parts,
        contour,
        worker_ids,
        assignments,
        Bool(store),
        Bool(materialize_nodes),
        matrix_update,
        matrix_prototype,
        Int(worker_blas_threads),
        factorizer,
        left_divider,
        Λ,
        res,
        inside,
        R,
        Q₀,
        Q₁,
        A,
        B,
        Xq,
        qr_ws,
        svd_ws,
        eigen_ws,
        residual_matrix,
        residual_x,
        residual_y,
        futures,
        stats,
        false,
    )

    try
        start_ns = time_ns()
        _init_dense_nlfeast_workers!(plan)
        _add_elapsed!(stats, :setup_worker_ns, start_ns)
    catch err
        close(plan)
        rethrow(err)
    end

    plan
end
