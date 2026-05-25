"""
    nlfeast!(op, X, nodes, iter; kwargs...)
    nlfeast!(T!, T_prototype, X, nodes, iter; kwargs...)

Canonical nonlinear FEAST-Beyn hybrid prototype. The method applies the
NLFEAST residual inverse iteration idea directly to Beyn-style contour moments:
the first iteration is a Beyn solve, and subsequent iterations reuse the contour
nodes as fixed shifts to refine the subspace through residual solves.

The preferred interface is an `AbstractFeastOperator`: `operator_prototype(op)`
allocates caller-owned storage, `materialize!(M, op, λ)` writes `T(λ)` into it,
and `mul!(Y, op, λ, V)` applies `T(λ)V`. The older `T(λ)` materializing
interface remains available as a compatibility wrapper, and `T!(M, λ)` can be
passed with an explicit prototype.

For large operators where residuals should be computed by matrix-vector action
rather than by materializing `T(λ)`, pass
`residual_update=(res, X, R, Λ) -> ...`. That hook must normalize/update `X`,
write residual vectors into `R`, and fill `res`.
"""
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

function nlfeast!(T, X::AbstractMatrix{ComplexF64}, nodes::Integer, iter::Integer;
    c=complex(0.0, 0.0), r=1.0, debug=false, ϵ=10e-12, store=true, spurious=1e-5,
    factorizer=lu, left_divider=ldiv!, residual_update=nothing,
    stats::Union{Nothing,DenseFeastStats}=nothing,
    matrix_update=nothing,
    matrix_prototype=nothing,
    solver::AbstractSparseFeastSolver=SparseDirectSolver(),
    contour=nothing)

    contour = contour === nothing ? circular_contour_trapezoidal(c, r, nodes) : contour
    nlfeast!(
        T,
        X,
        contour,
        iter;
        debug=debug,
        ϵ=ϵ,
        store=store,
        spurious=spurious,
        factorizer=factorizer,
        left_divider=left_divider,
        residual_update=residual_update,
        stats=stats,
        matrix_update=matrix_update,
        matrix_prototype=matrix_prototype,
        solver=solver,
    )
end

function nlfeast!(T, X::AbstractMatrix{ComplexF64}, contour::Contour, iter::Integer;
    debug=false, ϵ=10e-12, store=true, spurious=1e-5,
    factorizer=lu, left_divider=ldiv!, residual_update=nothing,
    stats::Union{Nothing,DenseFeastStats}=nothing,
    matrix_update=nothing,
    matrix_prototype=nothing,
    solver::AbstractSparseFeastSolver=SparseDirectSolver())

    matrix_update, matrix_prototype = _nlfeast_operator_storage(T, matrix_update, matrix_prototype)
    N, m₀ = size(X)
    Λ, res = zeros(ComplexF64, m₀), Array{Float64}(undef, m₀)
    resolvent = zeros(ComplexF64, m₀)
    inside = falses(m₀)
    z_nodes = ComplexF64.(contour_nodes(contour))
    z_weights = ComplexF64.(contour_weights(contour))
    nodes = length(z_nodes)
    Q₀, Q₁, R, Tinv = similar(X, ComplexF64), similar(X, ComplexF64), similar(X, ComplexF64), similar(X, ComplexF64)
    A, B, Xq = zeros(ComplexF64, m₀, m₀), zeros(ComplexF64, m₀, m₀), zeros(ComplexF64, m₀, m₀)
    T_prototype = matrix_prototype === nothing ? T(z_nodes[1]) : matrix_prototype
    residual_matrix = T_prototype isa StridedMatrix ? similar(T_prototype) : nothing
    residual_x, residual_y = zeros(ComplexF64, N), zeros(ComplexF64, N)
    sparse_path = T_prototype isa AbstractSparseMatrix
    store && sparse_path && !_sparse_supports_stored_factors(solver) &&
        error("store=true for sparse nonlinear FEAST requires a solver with reusable factorizations")

    qr_ws = X isa StridedMatrix ? QRWs(X) : nothing
    if qr_ws === nothing
        qt, rt = qr!(X)
        X .= Matrix(qt)
    else
        dense_lapack_qr!(X, qr_ws)
    end
    svd_ws = Q₀ isa StridedMatrix ? SVDsddWs(Q₀, job='S') : nothing
    eigen_ws = EigenWs(A, rvecs=true)
    lapack_lu_ws = nothing
    Tz = nothing
    if !store && factorizer === lu && left_divider === ldiv! && X isa StridedMatrix
        if T_prototype isa StridedMatrix && eltype(T_prototype) <: DenseLapackScalar
            Tz = similar(T_prototype)
            lapack_lu_ws = dense_lapack_lu_workspace(Tz)
        end
    end
    generic_Tz = !store && matrix_update !== nothing && lapack_lu_ws === nothing && !sparse_path ? similar(T_prototype) : nothing
    sparse_Tz = !store && matrix_update !== nothing && sparse_path ? similar(T_prototype) : nothing
    sparse_reusable_factor = nothing
    facts = nothing

    if store
        facts1 = if sparse_path
            Tmat = if matrix_update === nothing
                T(z_nodes[1])
            else
                Tbuf = similar(T_prototype)
                matrix_update(Tbuf, z_nodes[1])
                Tbuf
            end
            _sparse_factor(solver, Tmat)
        elseif matrix_update === nothing
            factorizer(T(z_nodes[1]))
        else
            Tbuf = similar(T_prototype)
            matrix_update(Tbuf, z_nodes[1])
            factorizer(Tbuf)
        end
        facts = Array{typeof(facts1)}(undef, nodes)
        facts[1] = facts1
        for i = 2:nodes
            facts[i] = if sparse_path
                Tmat = if matrix_update === nothing
                    T(z_nodes[i])
                else
                    Tbuf = similar(T_prototype)
                    matrix_update(Tbuf, z_nodes[i])
                    Tbuf
                end
                _sparse_factor(solver, Tmat)
            elseif matrix_update === nothing
                factorizer(T(z_nodes[i]))
            else
                Tbuf = similar(T_prototype)
                matrix_update(Tbuf, z_nodes[i])
                factorizer(Tbuf)
            end
            if debug print("*") end
        end
        if debug println() end
        sparse_path && _record_stored_factor_memory!(stats, facts)
    end

    solve_start_ns = time_ns()
    for nit = 0:iter
        iter_start_ns = time_ns()
        rayleigh_ritz_ns = UInt64(0)
        residual_ns = UInt64(0)
        filter_ns = UInt64(0)

        Q₀ .= 0
        Q₁ .= 0

        start_ns = time_ns()
        for i = 1:nodes
            z = z_nodes[i]
            if nit == 0
                if store
                    if sparse_path
                        _sparse_solve_factored!(Tinv, solver, facts[i], X)
                    else
                        left_divider(Tinv, facts[i], X)
                    end
                elseif lapack_lu_ws !== nothing
                    if matrix_update === nothing
                        copyto!(Tz, T(z))
                    else
                        matrix_update(Tz, z)
                    end
                    dense_lapack_linsolve!(Tinv, Tz, X, lapack_lu_ws)
                elseif matrix_update !== nothing
                    if sparse_path
                        matrix_update(sparse_Tz, z)
                        if solver isa SparseDirectSolver
                            if sparse_reusable_factor === nothing
                                sparse_reusable_factor = _sparse_factor(solver, sparse_Tz)
                                _sparse_solve_factored!(Tinv, solver, sparse_reusable_factor, X)
                            else
                                _sparse_linsolve_reuse_symbolic!(Tinv, solver, sparse_reusable_factor, sparse_Tz, X)
                            end
                        else
                            _sparse_linsolve!(Tinv, solver, sparse_Tz, X)
                        end
                    else
                        matrix_update(generic_Tz, z)
                        Tinv .= generic_Tz \ X
                    end
                elseif sparse_path
                    _sparse_linsolve!(Tinv, solver, T(z), X)
                else
                    Tinv .= T(z) \ X
                end
                rmul!(Tinv, z_weights[i])
            else
                fill_resolvent!(resolvent, z, Λ)
                if store
                    if sparse_path
                        _sparse_solve_factored!(Tinv, solver, facts[i], R)
                    else
                        left_divider(Tinv, facts[i], R)
                    end
                elseif lapack_lu_ws !== nothing
                    if matrix_update === nothing
                        copyto!(Tz, T(z))
                    else
                        matrix_update(Tz, z)
                    end
                    dense_lapack_linsolve!(Tinv, Tz, R, lapack_lu_ws)
                elseif matrix_update !== nothing
                    if sparse_path
                        matrix_update(sparse_Tz, z)
                        if solver isa SparseDirectSolver
                            if sparse_reusable_factor === nothing
                                sparse_reusable_factor = _sparse_factor(solver, sparse_Tz)
                                _sparse_solve_factored!(Tinv, solver, sparse_reusable_factor, R)
                            else
                                _sparse_linsolve_reuse_symbolic!(Tinv, solver, sparse_reusable_factor, sparse_Tz, R)
                            end
                        else
                            _sparse_linsolve!(Tinv, solver, sparse_Tz, R)
                        end
                    else
                        matrix_update(generic_Tz, z)
                        Tinv .= generic_Tz \ R
                    end
                elseif sparse_path
                    _sparse_linsolve!(Tinv, solver, T(z), R)
                else
                    Tinv .= T(z) \ R
                end
                accumulate_filtered_moments!(Q₀, Q₁, X, Tinv, resolvent, z_weights[i], z)
                if debug print(".") end
                continue
            end
            Q₀ .+= Tinv
            add_weighted_columns!(Q₁, Tinv, z)
            if debug print(".") end
        end
        filter_ns = time_ns() - start_ns
        if debug println() end

        start_ns = time_ns()
        if svd_ws === nothing
            beyn_svd_step!(Q₀, Q₁, A, B, X, Λ)
        else
            beyn_svd_step!(Q₀, Q₁, A, B, X, Λ, svd_ws, eigen_ws, Xq)
        end
        rayleigh_ritz_ns = time_ns() - start_ns

        start_ns = time_ns()
        if residual_update !== nothing
            # Expert fast path for action-only NEPs that cannot cheaply form T(λ).
            residual_update(res, X, R, Λ)
        elseif residual_matrix === nothing
            update_nonlinear_residuals!(res, X, R, Λ, T, residual_x, residual_y)
        else
            update_nonlinear_residuals!(res, X, R, Λ, T, residual_matrix, residual_x, residual_y)
        end
        in_contour!(inside, Λ, contour)
        max_res_inside, contour_nonempty = maximum_masked(res, inside)
        residual_ns = time_ns() - start_ns

        if debug
            iter_debug_print(nit, Λ, res, contour, spurious)
        end

        converged = contour_nonempty && max_res_inside < ϵ
        max_spurious_res_inside, spurious_found = maximum_below_masked(res, inside, spurious)
        spurious_converged = nit > 1 && spurious_found && max_spurious_res_inside < ϵ
        _record_dense_feast_iteration!(
            stats,
            sparse_path ? :sparse_nonlinear : :nonlinear,
            nit,
            res,
            inside,
            max_res_inside,
            contour_nonempty,
            iter_start_ns,
            UInt64(0),
            rayleigh_ritz_ns,
            residual_ns,
            filter_ns,
            debug,
        )
        if converged
            break
        end
        if spurious_converged
            break
        end
    end
    if stats !== nothing
        stats.solve_total_ns += time_ns() - solve_start_ns
    end
    if sparse_path && facts !== nothing
        foreach(finalize!, facts)
    end
    if sparse_reusable_factor !== nothing
        finalize!(sparse_reusable_factor)
    end

    normalize!(X)
    Λ, X, res
end

function nlfeast!(
    matrix_update,
    matrix_prototype::AbstractMatrix,
    X::AbstractMatrix{ComplexF64},
    nodes::Integer,
    iter::Integer;
    c=complex(0.0, 0.0),
    r=1.0,
    contour=nothing,
    kwargs...,
)
    contour = contour === nothing ? circular_contour_trapezoidal(c, r, nodes) : contour
    nlfeast!(matrix_update, matrix_prototype, X, contour, iter; kwargs...)
end

function nlfeast!(
    matrix_update,
    matrix_prototype::AbstractMatrix,
    X::AbstractMatrix{ComplexF64},
    contour::Contour,
    iter::Integer;
    kwargs...,
)
    materializing_T = z -> begin
        Tz = similar(matrix_prototype)
        matrix_update(Tz, z)
        Tz
    end
    nlfeast!(
        materializing_T,
        X,
        contour,
        iter;
        matrix_update=matrix_update,
        matrix_prototype=matrix_prototype,
        kwargs...,
    )
end
