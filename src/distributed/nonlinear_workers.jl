mutable struct DenseNonlinearFeastDistributedWorkspace
    T
    node_matrices
    X
    R
    Q₀part::Matrix{ComplexF64}
    Q₁part::Matrix{ComplexF64}
    node_indices::Vector{Int}
    nodes::Vector{ComplexF64}
    weights::Vector{ComplexF64}
    temp::Matrix{ComplexF64}
    resolvent::Vector{ComplexF64}
    factorizer
    left_divider
    matrix_update
    matrix_prototype
    Tz
    lu_ws
    sparse_reusable_factor
    stored_factors
    old_blas_threads::Int
end

function _init_dense_nlfeast_workers!(plan::DenseDistributedNonlinearFeastPlan)
    nodes = ComplexF64.(contour_nodes(plan.contour))
    weights = ComplexF64.(contour_weights(plan.contour))
    # Materializing nodes on the master is the most robust path for arbitrary
    # closures. Large no-store runs can disable it so workers build T(z) locally.
    node_matrices = plan.materialize_nodes ? [plan.T(nodes[i]) for i in eachindex(nodes)] : nothing
    worker_T = node_matrices === nothing && plan.matrix_update === nothing ? plan.T : nothing
    for (part_index, pid) in enumerate(plan.worker_ids)
        assigned_matrices = node_matrices === nothing ? nothing : node_matrices[plan.assignments[part_index]]
        plan.futures[part_index] = remotecall(
            _init_dense_nlfeast_worker!,
            pid,
            plan.key,
            worker_T,
            assigned_matrices,
            size(plan.X_buffer, 1),
            size(plan.X_buffer, 2),
            plan.assignments[part_index],
            nodes,
            weights,
            plan.store,
            plan.worker_blas_threads,
            plan.factorizer,
            plan.left_divider,
            plan.matrix_update,
            plan.matrix_prototype,
        )
    end
    for future in plan.futures
        fetch(future)
    end
    nothing
end

function _init_dense_nlfeast_worker!(
    key::Symbol,
    T,
    node_matrices,
    N::Int,
    m₀::Int,
    node_indices::Vector{Int},
    nodes::Vector{ComplexF64},
    weights::Vector{ComplexF64},
    store::Bool,
    worker_blas_threads::Int,
    factorizer,
    left_divider,
    matrix_update,
    matrix_prototype,
)
    old_blas_threads = BLAS.get_num_threads()
    BLAS.set_num_threads(worker_blas_threads)

    X = zeros(ComplexF64, N, m₀)
    R = zeros(ComplexF64, N, m₀)
    Q₀part = zeros(ComplexF64, N, m₀)
    Q₁part = zeros(ComplexF64, N, m₀)
    temp = zeros(ComplexF64, N, m₀)
    resolvent = zeros(ComplexF64, m₀)

    Tz = nothing
    lu_ws = nothing
    sparse_reusable_factor = nothing
    stored_factors = nothing
    if store
        stored_factors = if matrix_update !== nothing
            matrix_prototype === nothing && error("matrix_prototype is required when matrix_update is provided")
            map(node_indices) do i
                Tbuf = similar(matrix_prototype)
                matrix_update(Tbuf, nodes[i])
                factorizer(Tbuf)
            end
        elseif node_matrices === nothing
            [factorizer(T(nodes[i])) for i in node_indices]
        else
            [factorizer(node_matrices[i]) for i in eachindex(node_indices)]
        end
    else
        if matrix_update !== nothing
            matrix_prototype === nothing && error("matrix_prototype is required when matrix_update is provided")
            Tz = similar(matrix_prototype)
            if Tz isa StridedMatrix &&
                    eltype(Tz) <: DenseLapackScalar &&
                    factorizer === lu &&
                    left_divider === ldiv!
                lu_ws = dense_lapack_lu_workspace(Tz)
            elseif Tz isa SparseMatrixCSC &&
                    factorizer === lu &&
                    left_divider === ldiv!
                matrix_update(Tz, nodes[first(node_indices)])
                sparse_reusable_factor = _sparse_factor(SparseDirectSolver(), Tz)
            end
        else
            T_prototype = node_matrices === nothing ? T(nodes[node_indices[1]]) : node_matrices[1]
            if T_prototype isa StridedMatrix &&
                    eltype(T_prototype) <: DenseLapackScalar &&
                    factorizer === lu &&
                    left_divider === ldiv!
                Tz = similar(T_prototype)
                lu_ws = dense_lapack_lu_workspace(Tz)
            end
        end
    end

    _DISTRIBUTED_DENSE_FEAST_WORKSPACES[key] = DenseNonlinearFeastDistributedWorkspace(
        T,
        node_matrices,
        X,
        R,
        Q₀part,
        Q₁part,
        node_indices,
        nodes,
        weights,
        temp,
        resolvent,
        factorizer,
        left_divider,
        matrix_update,
        matrix_prototype,
        Tz,
        lu_ws,
        sparse_reusable_factor,
        stored_factors,
        old_blas_threads,
    )
    nothing
end

function _dense_nlfeast_worker_step!(
    key::Symbol,
    X::AbstractMatrix{ComplexF64},
    rhs::AbstractMatrix{ComplexF64},
    Λ::Vector{ComplexF64},
    first_iteration::Bool,
)
    ws = _DISTRIBUTED_DENSE_FEAST_WORKSPACES[key]
    copyto!(ws.X, X)
    copyto!(ws.R, rhs)
    Q₀part = ws.Q₀part
    Q₁part = ws.Q₁part
    fill!(Q₀part, 0)
    fill!(Q₁part, 0)

    solve_ns = UInt64(0)
    materialize_ns = UInt64(0)
    linsolve_ns = UInt64(0)
    accum_ns = UInt64(0)
    for (local_index, node_index) in pairs(ws.node_indices)
        z = ws.nodes[node_index]
        start_ns = time_ns()
        timing = _dense_nlfeast_solve!(ws.temp, ws, local_index, rhs)
        solve_ns += time_ns() - start_ns
        materialize_ns += timing.materialize_ns
        linsolve_ns += timing.linsolve_ns

        start_ns = time_ns()
        if first_iteration
            weight = ws.weights[node_index]
            axpy!(weight, ws.temp, Q₀part)
            axpy!(z * weight, ws.temp, Q₁part)
            accum_ns += time_ns() - start_ns
            continue
        else
            fill_resolvent!(ws.resolvent, z, Λ)
            accumulate_filtered_moments!(Q₀part, Q₁part, ws.X, ws.temp, ws.resolvent, ws.weights[node_index], z)
        end
        accum_ns += time_ns() - start_ns
    end
    (; Q₀part, Q₁part, solve_ns, materialize_ns, linsolve_ns, accum_ns)
end

function _dense_nlfeast_solve!(Y, ws::DenseNonlinearFeastDistributedWorkspace, local_index::Int, rhs)
    materialize_ns = UInt64(0)
    linsolve_ns = UInt64(0)
    if ws.stored_factors !== nothing
        start_ns = time_ns()
        ws.left_divider(Y, ws.stored_factors[local_index], rhs)
        linsolve_ns = time_ns() - start_ns
    elseif ws.matrix_update !== nothing
        start_ns = time_ns()
        z = ws.nodes[ws.node_indices[local_index]]
        ws.matrix_update(ws.Tz, z)
        materialize_ns = time_ns() - start_ns
        start_ns = time_ns()
        if ws.lu_ws !== nothing
            dense_lapack_linsolve!(Y, ws.Tz, rhs, ws.lu_ws)
        elseif ws.sparse_reusable_factor !== nothing
            _sparse_linsolve_reuse_symbolic!(Y, SparseDirectSolver(), ws.sparse_reusable_factor, ws.Tz, rhs)
        else
            Y .= ws.Tz \ rhs
        end
        linsolve_ns = time_ns() - start_ns
    elseif ws.lu_ws !== nothing
        if ws.node_matrices === nothing
            start_ns = time_ns()
            z = ws.nodes[ws.node_indices[local_index]]
            copyto!(ws.Tz, ws.T(z))
            materialize_ns = time_ns() - start_ns
        else
            copyto!(ws.Tz, ws.node_matrices[local_index])
        end
        start_ns = time_ns()
        dense_lapack_linsolve!(Y, ws.Tz, rhs, ws.lu_ws)
        linsolve_ns = time_ns() - start_ns
    else
        start_ns = time_ns()
        matrix = ws.node_matrices === nothing ? ws.T(ws.nodes[ws.node_indices[local_index]]) : ws.node_matrices[local_index]
        materialize_ns = ws.node_matrices === nothing ? time_ns() - start_ns : UInt64(0)
        start_ns = time_ns()
        Y .= matrix \ rhs
        linsolve_ns = time_ns() - start_ns
    end
    (; materialize_ns, linsolve_ns)
end
