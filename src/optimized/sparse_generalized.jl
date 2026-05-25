function gen_feast!(
    X::AbstractMatrix,
    A::AbstractSparseMatrix,
    B::Union{AbstractSparseMatrix,UniformScaling};
    nodes::Integer=8,
    iter::Integer=10,
    c=complex(0.0, 0.0),
    r=1.0,
    debug=false,
    store=false,
    ϵ=1e-12,
    solver::AbstractSparseFeastSolver=SparseDirectSolver(),
    stats::Union{Nothing,DenseFeastStats}=nothing,
    contour=nothing,
)
    contour = contour === nothing ? circular_contour_trapezoidal(c, r, nodes) : contour
    gen_feast!(X, A, B, contour; iter=iter, debug=debug, store=store, ϵ=ϵ, solver=solver, stats=stats)
end

"""
    gen_feast!(X, A::AbstractSparseMatrix, B, contour; solver=SparseDirectSolver(), kwargs...)

Sparse generalized FEAST for `Ax = λBx`. This first pass supports sparse direct
factorization and the experimental BiCGSTAB policy through the same shifted
solve abstraction as sparse standard FEAST.
"""
function gen_feast!(
    X::AbstractMatrix,
    A::AbstractSparseMatrix,
    B::Union{AbstractSparseMatrix,UniformScaling},
    contour::Contour;
    iter::Integer=10,
    debug=false,
    store=false,
    ϵ=1e-12,
    solver::AbstractSparseFeastSolver=SparseDirectSolver(),
    stats::Union{Nothing,DenseFeastStats}=nothing,
)
    store && !_sparse_supports_stored_factors(solver) &&
        error("store=true for sparse FEAST requires a solver with reusable factorizations")

    N, m₀ = size(X)
    if size(A, 1) != size(A, 2)
        error("Incorrect dimensions of A, must be square")
    elseif size(A, 1) != N
        error("Incorrect dimensions of X, must match A")
    elseif !(B isa UniformScaling) && size(A) != size(B)
        error("Incorrect dimensions of A and B, must match")
    end

    Λ, resolvent, res = zeros(ComplexF64, m₀), zeros(ComplexF64, m₀), zeros(m₀)
    inside = falses(m₀)
    temp, R, Q = zeros(ComplexF64, N, m₀), similar(X, ComplexF64), copy(X)
    Aq, Bq, Xq = zeros(ComplexF64, m₀, m₀), zeros(ComplexF64, m₀, m₀), zeros(ComplexF64, m₀, m₀)
    qr_ws = Q isa StridedMatrix && eltype(Q) <: DenseLapackScalar ? QRWs(Q) : nothing
    eigen_ws = GeneralizedEigenWs(Aq, lvecs=true, rvecs=true)
    z_nodes = contour_nodes(contour)
    z_weights = contour_weights(contour)
    node_count = length(z_nodes)
    shift_ws = _sparse_generalized_shift_workspace(A, B)
    reusable_factor = nothing

    stored_factors = nothing
    if store
        first_shift = B isa UniformScaling ?
                      _materialize_sparse_generalized_shift!(shift_ws, z_nodes[1], B) :
                      _materialize_sparse_generalized_shift!(shift_ws, z_nodes[1])
        first_factor = _sparse_factor(solver, first_shift)
        stored_factors = Vector{typeof(first_factor)}(undef, node_count)
        stored_factors[1] = first_factor
        for i in 2:node_count
            shift = B isa UniformScaling ?
                    _materialize_sparse_generalized_shift!(shift_ws, z_nodes[i], B) :
                    _materialize_sparse_generalized_shift!(shift_ws, z_nodes[i])
            stored_factors[i] = _sparse_factor(solver, shift)
        end
        _record_stored_factor_memory!(stats, stored_factors)
    end

    solve_start_ns = time_ns()
    for nit in 0:iter
        iter_start_ns = time_ns()
        orthogonalization_ns = UInt64(0)
        rayleigh_ritz_ns = UInt64(0)
        residual_ns = UInt64(0)
        filter_ns = UInt64(0)

        start_ns = time_ns()
        if qr_ws === nothing
            Q .= Matrix(qr(Q).Q)
        else
            dense_lapack_qr!(Q, qr_ws)
        end
        orthogonalization_ns = time_ns() - start_ns

        start_ns = time_ns()
        mul!(R, A, Q)
        mul!(Aq, Q', R)
        mul!(R, B, Q)
        mul!(Bq, Q', R)
        dense_lapack_generalized_eigen!(Λ, Xq, Aq, Bq, eigen_ws)
        mul!(X, Q, Xq)
        rayleigh_ritz_ns = time_ns() - start_ns

        start_ns = time_ns()
        update_R!(X, R, Λ, A, B, temp)
        residuals!(res, R, Λ, A)
        in_contour!(inside, Λ, contour)
        max_res_inside, contour_nonempty = maximum_masked(res, inside)
        residual_ns = time_ns() - start_ns
        if debug
            iter_debug_print(nit, Λ, res, contour, 1e-5)
        end
        converged = contour_nonempty && max_res_inside < ϵ
        if converged && debug
            println("converged in $nit iteration")
        end

        if !converged && nit < iter
            start_ns = time_ns()
            Q .= 0
            for i in 1:node_count
                fill_resolvent!(resolvent, z_nodes[i], Λ)
                if store
                    _sparse_solve_factored!(temp, solver, stored_factors[i], R)
                else
                    shift = B isa UniformScaling ?
                            _materialize_sparse_generalized_shift!(shift_ws, z_nodes[i], B) :
                            _materialize_sparse_generalized_shift!(shift_ws, z_nodes[i])
                    if solver isa SparseDirectSolver
                        if reusable_factor === nothing
                            reusable_factor = _sparse_factor(solver, shift)
                            _sparse_solve_factored!(temp, solver, reusable_factor, R)
                        else
                            _sparse_linsolve_reuse_symbolic!(temp, solver, reusable_factor, shift, R)
                        end
                    else
                        _sparse_linsolve!(temp, solver, shift, R)
                    end
                end
                accumulate_filtered_columns!(Q, X, temp, resolvent, z_weights[i])
            end
            filter_ns = time_ns() - start_ns
        end

        _record_dense_feast_iteration!(
            stats,
            :sparse_generalized,
            nit,
            res,
            inside,
            max_res_inside,
            contour_nonempty,
            iter_start_ns,
            orthogonalization_ns,
            rayleigh_ritz_ns,
            residual_ns,
            filter_ns,
            debug,
        )
        converged && break
    end
    if stats !== nothing
        stats.solve_total_ns += time_ns() - solve_start_ns
    end
    if stored_factors !== nothing
        foreach(finalize!, stored_factors)
    end
    if reusable_factor !== nothing
        finalize!(reusable_factor)
    end
    in_contour!(inside, Λ, contour)
    _, contour_nonempty = maximum_masked(res, inside)
    if !contour_nonempty
        println("no eigenvalues found in contour!")
    end
    Λ[inside], X[:, inside], res[inside]
end
