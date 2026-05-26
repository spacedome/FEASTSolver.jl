function _nlfeast_operator_storage(T, matrix_update, matrix_prototype)
    if T isa AbstractFeastOperator
        matrix_update === nothing || error("pass either an AbstractFeastOperator or matrix_update, not both")
        matrix_prototype === nothing || error("pass either an AbstractFeastOperator or matrix_prototype, not both")
        return matrix_materializer(T), operator_prototype(T)
    end
    matrix_update === nothing && return nothing, matrix_prototype
    matrix_prototype === nothing && error("matrix_prototype is required when matrix_update is provided")
    matrix_update, matrix_prototype
end

function _nlfeast_materialize_node(T, matrix_update, T_prototype, z)
    if matrix_update === nothing
        return T(z)
    end
    Tz = similar(T_prototype)
    matrix_update(Tz, z)
    Tz
end

function _nlfeast_factor_node(T, matrix_update, T_prototype, z, sparse_path, solver, factorizer)
    Tz = _nlfeast_materialize_node(T, matrix_update, T_prototype, z)
    sparse_path ? _sparse_factor(solver, Tz) : factorizer(Tz)
end

function _nlfeast_store_factors(T, matrix_update, T_prototype, z_nodes, sparse_path, solver, factorizer, debug)
    first_factor = _nlfeast_factor_node(T, matrix_update, T_prototype, z_nodes[1], sparse_path, solver, factorizer)
    facts = Array{typeof(first_factor)}(undef, length(z_nodes))
    facts[1] = first_factor
    for i = 2:length(z_nodes)
        facts[i] = _nlfeast_factor_node(T, matrix_update, T_prototype, z_nodes[i], sparse_path, solver, factorizer)
        debug && print("*")
    end
    debug && println()
    facts
end

function _nlfeast_dense_lapack_workspace(T_prototype, X, store, factorizer, left_divider)
    if !store &&
       factorizer === lu &&
       left_divider === ldiv! &&
       X isa StridedMatrix &&
       T_prototype isa StridedMatrix &&
       eltype(T_prototype) <: DenseLapackScalar
        Tz = similar(T_prototype)
        return Tz, dense_lapack_lu_workspace(Tz)
    end
    nothing, nothing
end

function _nlfeast_update_or_copy!(Tz, T, matrix_update, z)
    if matrix_update === nothing
        copyto!(Tz, T(z))
    else
        matrix_update(Tz, z)
    end
    Tz
end

function _nlfeast_solve_shifted!(
    Y,
    rhs,
    T,
    z,
    node_index,
    store,
    facts,
    matrix_update,
    sparse_path,
    solver,
    left_divider,
    lapack_lu_ws,
    Tz,
    generic_Tz,
    sparse_Tz,
    sparse_reusable_factor,
)
    if store
        if sparse_path
            _sparse_solve_factored!(Y, solver, facts[node_index], rhs)
        else
            left_divider(Y, facts[node_index], rhs)
        end
    elseif lapack_lu_ws !== nothing
        _nlfeast_update_or_copy!(Tz, T, matrix_update, z)
        dense_lapack_linsolve!(Y, Tz, rhs, lapack_lu_ws)
    elseif matrix_update !== nothing
        sparse_reusable_factor = _nlfeast_solve_materialized_update!(
            Y,
            rhs,
            z,
            matrix_update,
            sparse_path,
            solver,
            generic_Tz,
            sparse_Tz,
            sparse_reusable_factor,
        )
    elseif sparse_path
        _sparse_linsolve!(Y, solver, T(z), rhs)
    else
        Y .= T(z) \ rhs
    end
    sparse_reusable_factor
end

function _nlfeast_solve_materialized_update!(
    Y,
    rhs,
    z,
    matrix_update,
    sparse_path,
    solver,
    generic_Tz,
    sparse_Tz,
    sparse_reusable_factor,
)
    if sparse_path
        matrix_update(sparse_Tz, z)
        if solver isa SparseDirectSolver
            if sparse_reusable_factor === nothing
                sparse_reusable_factor = _sparse_factor(solver, sparse_Tz)
                _sparse_solve_factored!(Y, solver, sparse_reusable_factor, rhs)
            else
                _sparse_linsolve_reuse_symbolic!(Y, solver, sparse_reusable_factor, sparse_Tz, rhs)
            end
        else
            _sparse_linsolve!(Y, solver, sparse_Tz, rhs)
        end
    else
        matrix_update(generic_Tz, z)
        Y .= generic_Tz \ rhs
    end
    sparse_reusable_factor
end

function _nlfeast_finalize_shifted_solves!(sparse_path, facts, sparse_reusable_factor)
    if sparse_path && facts !== nothing
        foreach(finalize!, facts)
    end
    if sparse_reusable_factor !== nothing
        finalize!(sparse_reusable_factor)
    end
    nothing
end
