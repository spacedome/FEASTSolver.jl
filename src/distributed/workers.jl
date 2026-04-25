# Worker-local state and remote kernels for dense distributed FEAST.
#
# Each worker owns a stable subset of contour nodes. Persistent workspaces keep
# shifted factors and scratch buffers local to that process across iterations.

const _DISTRIBUTED_DENSE_FEAST_WORKSPACES = Dict{Symbol, Any}()

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

function _prepare_dense_feast_workers!(worker_ids)
    for pid in worker_ids
        remotecall_wait(Main.eval, pid, :(using SharedArrays; using FEASTSolver))
    end
    nothing
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
