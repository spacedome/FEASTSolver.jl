mutable struct DenseFeastDistributedWorkspace
    A
    X
    R
    Qpart::Matrix{ComplexF64}
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

function _init_dense_feast_workers!(plan::DenseDistributedFeastPlan)
    nodes = ComplexF64.(plan.contour.nodes)
    weights = ComplexF64.(plan.contour.weights)
    for (part_index, pid) in enumerate(plan.worker_ids)
        plan.futures[part_index] = remotecall(
            _init_dense_feast_worker!,
            pid,
            plan.key,
            plan.A,
            size(plan.X_buffer, 1),
            size(plan.X_buffer, 2),
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
    N::Int,
    m₀::Int,
    node_indices::Vector{Int},
    nodes::Vector{ComplexF64},
    weights::Vector{ComplexF64},
    store::Bool,
    worker_blas_threads::Int,
)
    old_blas_threads = BLAS.get_num_threads()
    BLAS.set_num_threads(worker_blas_threads)

    X = zeros(ComplexF64, N, m₀)
    R = zeros(ComplexF64, N, m₀)
    Qpart = zeros(ComplexF64, N, m₀)
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
        Qpart,
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

function _dense_feast_worker_step!(key::Symbol, X::AbstractMatrix{ComplexF64}, R::AbstractMatrix{ComplexF64}, Λ::Vector{ComplexF64})
    ws = _DISTRIBUTED_DENSE_FEAST_WORKSPACES[key]
    copyto!(ws.X, X)
    copyto!(ws.R, R)
    Qpart = ws.Qpart
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
    Qpart
end
