mutable struct DenseDualGeneralizedFeastDistributedWorkspace
    A
    B
    Xr
    Xl
    Rr
    Rl
    Qrpart::Matrix{ComplexF64}
    Qlpart::Matrix{ComplexF64}
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
            size(plan.Xr_buffer, 1),
            size(plan.Xr_buffer, 2),
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

function _init_dense_dual_gen_feast_worker!(
    key::Symbol,
    A,
    B,
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

    Xr = zeros(ComplexF64, N, m₀)
    Xl = zeros(ComplexF64, N, m₀)
    Rr = zeros(ComplexF64, N, m₀)
    Rl = zeros(ComplexF64, N, m₀)
    Qrpart = zeros(ComplexF64, N, m₀)
    Qlpart = zeros(ComplexF64, N, m₀)
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
        Qrpart,
        Qlpart,
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

function _dense_dual_gen_feast_worker_step!(
    key::Symbol,
    Xr::AbstractMatrix{ComplexF64},
    Xl::AbstractMatrix{ComplexF64},
    Rr::AbstractMatrix{ComplexF64},
    Rl::AbstractMatrix{ComplexF64},
    Λ::Vector{ComplexF64},
)
    ws = _DISTRIBUTED_DENSE_FEAST_WORKSPACES[key]
    copyto!(ws.Xr, Xr)
    copyto!(ws.Xl, Xl)
    copyto!(ws.Rr, Rr)
    copyto!(ws.Rl, Rl)
    Qrpart = ws.Qrpart
    Qlpart = ws.Qlpart
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
    (; Qrpart, Qlpart)
end
