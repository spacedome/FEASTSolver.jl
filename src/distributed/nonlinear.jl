# Distributed canonical nonlinear FEAST.
#
# The parallelism model mirrors dense linear FEAST: each worker owns a stable
# subset of contour nodes, plus optional node-local factorizations. The master
# process owns Beyn compression and residual checks.

"""
    DenseDistributedNonlinearFeastPlan(T, n, m0, [contour]; kwargs...)

Persistent contour-worker plan for the canonical nonlinear FEAST prototype.
`T` is serialized to each worker once at setup, so captured problem data remain
worker-local across iterations.
"""
mutable struct DenseDistributedNonlinearFeastPlan <: AbstractDenseDistributedFeastPlan
    key::Symbol
    T
    X_shared
    R_shared
    Q₀parts
    Q₁parts
    contour::Contour
    worker_ids::Vector{Int}
    assignments::Vector{Vector{Int}}
    store::Bool
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

mutable struct DenseNonlinearFeastDistributedWorkspace
    node_matrices
    X
    R
    Q₀parts
    Q₁parts
    part_index::Int
    node_indices::Vector{Int}
    nodes::Vector{ComplexF64}
    weights::Vector{ComplexF64}
    temp::Matrix{ComplexF64}
    resolvent::Vector{ComplexF64}
    factorizer
    left_divider
    Tz
    lu_ws
    stored_factors
    old_blas_threads::Int
end

function DenseDistributedNonlinearFeastPlan(
    T,
    n::Integer,
    m₀::Integer;
    nodes::Integer=8,
    c=complex(0.0, 0.0),
    r=1.0,
    kwargs...,
)
    contour = circular_contour_trapezoidal(c, r, nodes)
    DenseDistributedNonlinearFeastPlan(T, n, m₀, contour; kwargs...)
end

function DenseDistributedNonlinearFeastPlan(
    T,
    n::Integer,
    m₀::Integer,
    contour::Contour;
    store=true,
    worker_ids=_default_feast_worker_ids(),
    worker_blas_threads::Integer=1,
    factorizer=lu,
    left_divider=ldiv!,
    stats::Union{Nothing, DenseDistributedFeastStats}=nothing,
)
    n > 0 || error("problem dimension must be positive")
    m₀ > 0 || error("subspace dimension must be positive")

    worker_ids = _normalize_feast_worker_ids(worker_ids)
    isempty(worker_ids) && error("DenseDistributedNonlinearFeastPlan requires worker processes; call addprocs(...) or start Julia with -p")

    node_count = size(contour.nodes, 1)
    worker_ids, assignments = _dense_feast_worker_assignments(worker_ids, node_count)
    all_pids = unique([myid(); worker_ids])

    start_ns = time_ns()
    _prepare_dense_feast_workers!(worker_ids)
    _add_elapsed!(stats, :setup_prepare_ns, start_ns)

    start_ns = time_ns()
    N = Int(n)
    X_shared = SharedArray{ComplexF64}((N, m₀); pids=all_pids)
    R_shared = SharedArray{ComplexF64}((N, m₀); pids=all_pids)
    Q₀parts = SharedArray{ComplexF64}((N, m₀, length(worker_ids)); pids=all_pids)
    Q₁parts = SharedArray{ComplexF64}((N, m₀, length(worker_ids)); pids=all_pids)

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

    T_prototype = T(ComplexF64(contour.nodes[1]))
    residual_matrix = T_prototype isa StridedMatrix ? similar(T_prototype) : nothing
    residual_x = zeros(ComplexF64, N)
    residual_y = zeros(ComplexF64, N)
    key = gensym(:dense_nlfeast)
    futures = Vector{Any}(undef, length(worker_ids))
    _add_elapsed!(stats, :setup_shared_ns, start_ns)

    plan = DenseDistributedNonlinearFeastPlan(
        key,
        T,
        X_shared,
        R_shared,
        Q₀parts,
        Q₁parts,
        contour,
        worker_ids,
        assignments,
        Bool(store),
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

function distributed_nlfeast!(
    T,
    X::AbstractMatrix{ComplexF64},
    nodes::Integer,
    iter::Integer;
    c=complex(0.0, 0.0),
    r=1.0,
    kwargs...,
)
    contour = circular_contour_trapezoidal(c, r, nodes)
    distributed_nlfeast!(T, X, contour, iter; kwargs...)
end

function distributed_nlfeast!(
    T,
    X::AbstractMatrix{ComplexF64},
    contour::Contour,
    iter::Integer;
    store=true,
    worker_ids=_default_feast_worker_ids(),
    worker_blas_threads::Integer=1,
    factorizer=lu,
    left_divider=ldiv!,
    stats::Union{Nothing, DenseDistributedFeastStats}=nothing,
    kwargs...,
)
    plan = DenseDistributedNonlinearFeastPlan(
        T,
        size(X, 1),
        size(X, 2),
        contour;
        store=store,
        worker_ids=worker_ids,
        worker_blas_threads=worker_blas_threads,
        factorizer=factorizer,
        left_divider=left_divider,
        stats=stats,
    )
    try
        distributed_nlfeast!(X, plan, iter; stats=stats, kwargs...)
    finally
        close(plan)
    end
end

function distributed_nlfeast!(
    X::AbstractMatrix{ComplexF64},
    plan::DenseDistributedNonlinearFeastPlan,
    iter::Integer;
    ϵ=10e-12,
    debug=false,
    spurious=1e-5,
    residual_update=nothing,
    stats::Union{Nothing, DenseDistributedFeastStats}=nothing,
)
    plan.closed && error("DenseDistributedNonlinearFeastPlan is closed")
    stats = stats === nothing ? plan.stats : stats
    N, m₀ = size(plan.X_shared)
    size(X) == (N, m₀) || error("Incorrect dimensions of X, must match planned problem and subspace dimension")

    T = plan.T
    contour = plan.contour
    Λ = plan.Λ
    res = plan.res
    inside = plan.inside
    R = plan.R
    Q₀ = plan.Q₀
    Q₁ = plan.Q₁
    A = plan.A
    B = plan.B
    Xq = plan.Xq

    dense_lapack_qr!(X, plan.qr_ws)

    solve_start_ns = time_ns()
    for nit=0:iter
        iter_start_ns = time_ns()
        rayleigh_ritz_ns = UInt64(0)
        residual_ns = UInt64(0)
        shared_copy_ns = UInt64(0)
        worker_step_ns = UInt64(0)
        reduce_ns = UInt64(0)

        if stats !== nothing
            stats.iterations += 1
        end

        start_ns = time_ns()
        copyto!(plan.X_shared, X)
        if nit > 0
            copyto!(plan.R_shared, R)
        end
        shared_copy_ns = time_ns() - start_ns
        _add_ns!(stats, :shared_copy_ns, shared_copy_ns)

        start_ns = time_ns()
        first_iteration = nit == 0
        for (i, pid) in enumerate(plan.worker_ids)
            plan.futures[i] = remotecall(_dense_nlfeast_worker_step!, pid, plan.key, Λ, first_iteration)
        end
        for future in plan.futures
            fetch(future)
        end
        worker_step_ns = time_ns() - start_ns
        _add_ns!(stats, :worker_step_ns, worker_step_ns)

        start_ns = time_ns()
        _sum_dense_feast_qparts!(Q₀, plan.Q₀parts)
        _sum_dense_feast_qparts!(Q₁, plan.Q₁parts)
        reduce_ns = time_ns() - start_ns
        _add_ns!(stats, :reduce_ns, reduce_ns)

        start_ns = time_ns()
        beyn_svd_step!(Q₀, Q₁, A, B, X, Λ, plan.svd_ws, plan.eigen_ws, Xq)
        rayleigh_ritz_ns = time_ns() - start_ns
        _add_ns!(stats, :rayleigh_ritz_ns, rayleigh_ritz_ns)

        start_ns = time_ns()
        if residual_update !== nothing
            # Expert fast path for action-only NEPs that cannot cheaply form T(λ).
            residual_update(res, X, R, Λ)
        elseif plan.residual_matrix === nothing
            update_nonlinear_residuals!(res, X, R, Λ, T, plan.residual_x, plan.residual_y)
        else
            update_nonlinear_residuals!(res, X, R, Λ, T, plan.residual_matrix, plan.residual_x, plan.residual_y)
        end
        in_contour!(inside, Λ, contour)
        max_res_inside, contour_nonempty = maximum_masked(res, inside)
        residual_ns = time_ns() - start_ns
        _add_ns!(stats, :residual_ns, residual_ns)

        if debug
            iter_debug_print(nit, Λ, res, contour, spurious)
        end

        converged = contour_nonempty && max_res_inside < ϵ
        max_spurious_res_inside, spurious_found = maximum_below_masked(res, inside, spurious)
        spurious_converged = nit > 1 && spurious_found && max_spurious_res_inside < ϵ
        _record_distributed_feast_iteration!(
            stats,
            nit,
            res,
            inside,
            max_res_inside,
            contour_nonempty,
            iter_start_ns,
            UInt64(0),
            rayleigh_ritz_ns,
            residual_ns,
            shared_copy_ns,
            worker_step_ns,
            reduce_ns,
            debug,
        )

        if converged || spurious_converged
            break
        end
    end
    _add_elapsed!(stats, :solve_total_ns, solve_start_ns)

    normalize!(X)
    Λ, X, res
end

function _init_dense_nlfeast_workers!(plan::DenseDistributedNonlinearFeastPlan)
    nodes = ComplexF64.(plan.contour.nodes)
    weights = ComplexF64.(plan.contour.weights)
    # Materialize T(z) on the master so locally scoped callables do not need to
    # deserialize on worker processes. Workers still own their shifts/factors.
    node_matrices = [plan.T(nodes[i]) for i in eachindex(nodes)]
    for (part_index, pid) in enumerate(plan.worker_ids)
        assigned_matrices = node_matrices[plan.assignments[part_index]]
        plan.futures[part_index] = remotecall(
            _init_dense_nlfeast_worker!,
            pid,
            plan.key,
            assigned_matrices,
            plan.X_shared,
            plan.R_shared,
            plan.Q₀parts,
            plan.Q₁parts,
            part_index,
            plan.assignments[part_index],
            nodes,
            weights,
            plan.store,
            plan.worker_blas_threads,
            plan.factorizer,
            plan.left_divider,
        )
    end
    for future in plan.futures
        fetch(future)
    end
    nothing
end

function _init_dense_nlfeast_worker!(
    key::Symbol,
    node_matrices,
    X,
    R,
    Q₀parts,
    Q₁parts,
    part_index::Int,
    node_indices::Vector{Int},
    nodes::Vector{ComplexF64},
    weights::Vector{ComplexF64},
    store::Bool,
    worker_blas_threads::Int,
    factorizer,
    left_divider,
)
    old_blas_threads = BLAS.get_num_threads()
    BLAS.set_num_threads(worker_blas_threads)

    N, m₀ = size(X)
    temp = zeros(ComplexF64, N, m₀)
    resolvent = zeros(ComplexF64, m₀)

    Tz = nothing
    lu_ws = nothing
    stored_factors = nothing
    if store
        stored_factors = [factorizer(node_matrices[i]) for i in eachindex(node_indices)]
    else
        T_prototype = node_matrices[1]
        if T_prototype isa StridedMatrix && eltype(T_prototype) <: DenseLapackScalar
            Tz = similar(T_prototype)
            lu_ws = dense_lapack_lu_workspace(Tz)
        end
    end

    _DISTRIBUTED_DENSE_FEAST_WORKSPACES[key] = DenseNonlinearFeastDistributedWorkspace(
        node_matrices,
        X,
        R,
        Q₀parts,
        Q₁parts,
        part_index,
        node_indices,
        nodes,
        weights,
        temp,
        resolvent,
        factorizer,
        left_divider,
        Tz,
        lu_ws,
        stored_factors,
        old_blas_threads,
    )
    nothing
end

function _dense_nlfeast_worker_step!(key::Symbol, Λ::Vector{ComplexF64}, first_iteration::Bool)
    ws = _DISTRIBUTED_DENSE_FEAST_WORKSPACES[key]
    Q₀part = view(ws.Q₀parts, :, :, ws.part_index)
    Q₁part = view(ws.Q₁parts, :, :, ws.part_index)
    fill!(Q₀part, 0)
    fill!(Q₁part, 0)

    rhs = first_iteration ? ws.X : ws.R
    for (local_index, node_index) in pairs(ws.node_indices)
        z = ws.nodes[node_index]
        _dense_nlfeast_solve!(ws.temp, ws, local_index, rhs)

        if first_iteration
            rmul!(ws.temp, ws.weights[node_index])
        else
            fill_resolvent!(ws.resolvent, z, Λ)
            @inbounds for j in axes(ws.temp, 2)
                α = ws.resolvent[j] * ws.weights[node_index]
                for i in axes(ws.temp, 1)
                    ws.temp[i, j] = (ws.X[i, j] - ws.temp[i, j]) * α
                end
            end
        end

        @inbounds for j in axes(ws.temp, 2)
            for i in axes(ws.temp, 1)
                value = ws.temp[i, j]
                Q₀part[i, j] += value
                Q₁part[i, j] += z * value
            end
        end
    end
    nothing
end

function _dense_nlfeast_solve!(Y, ws::DenseNonlinearFeastDistributedWorkspace, local_index::Int, rhs)
    if ws.stored_factors !== nothing
        ws.left_divider(Y, ws.stored_factors[local_index], rhs)
    elseif ws.lu_ws !== nothing
        copyto!(ws.Tz, ws.node_matrices[local_index])
        dense_lapack_linsolve!(Y, ws.Tz, rhs, ws.lu_ws)
    else
        Y .= ws.node_matrices[local_index] \ rhs
    end
    Y
end
