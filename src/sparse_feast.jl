abstract type AbstractSparseFeastSolver end

struct SparseStandardShiftWorkspace{Tv,Ti}
    A::SparseMatrixCSC{Tv,Ti}
    shifted::SparseMatrixCSC{ComplexF64,Ti}
    a_positions::Vector{Int}
    diag_positions::Vector{Int}
end

"""
    SparseDirectSolver()

Sparse direct linear solver policy for sparse FEAST. The default path uses
SuiteSparse/UMFPACK through Julia's `lu(::SparseMatrixCSC)` and supports
`store=true` by keeping one factorization per contour node.
"""
struct SparseDirectSolver <: AbstractSparseFeastSolver end

"""
    SparseBiCGSTABSolver(; l=2, abstol=0.0, reltol=1e-10, max_mv_products=0)

First-pass iterative solver policy for sparse FEAST shifted systems. This uses
IterativeSolvers.jl one right-hand side at a time. It is useful for experiments,
but does not yet provide the persistent workspace model we want long term.
"""
struct SparseBiCGSTABSolver{R<:Real} <: AbstractSparseFeastSolver
    l::Int
    abstol::R
    reltol::R
    max_mv_products::Int
end

function SparseBiCGSTABSolver(; l::Integer=2, abstol::Real=0.0, reltol::Real=1e-10, max_mv_products::Integer=0)
    SparseBiCGSTABSolver(Int(l), promote(abstol, reltol)..., Int(max_mv_products))
end

_sparse_supports_stored_factors(::SparseDirectSolver) = true
_sparse_supports_stored_factors(::AbstractSparseFeastSolver) = false

function _sparse_standard_shift_workspace(A::AbstractSparseMatrix)
    size(A, 1) == size(A, 2) || error("Incorrect dimensions of A, must be square")
    Ac = sparse(A)
    n = size(Ac, 1)
    shifted = _sparse_standard_shift_pattern(Ac)
    a_positions = _sparse_pattern_positions(Ac, shifted)
    diag_positions = Vector{Int}(undef, n)
    @inbounds for j in 1:n
        diag_positions[j] = _find_sparse_row(shifted, j, j)
    end
    SparseStandardShiftWorkspace(Ac, shifted, a_positions, diag_positions)
end

function _sparse_standard_shift_pattern(A::SparseMatrixCSC)
    n = size(A, 1)
    colptr = Vector{eltype(A.colptr)}(undef, n + 1)
    rows = Vector{eltype(A.rowval)}()
    sizehint!(rows, nnz(A) + n)
    colptr[1] = 1
    @inbounds for j in 1:n
        inserted_diag = false
        for p in A.colptr[j]:(A.colptr[j + 1] - 1)
            row = A.rowval[p]
            if !inserted_diag && j < row
                push!(rows, j)
                inserted_diag = true
            end
            push!(rows, row)
            inserted_diag |= row == j
        end
        if !inserted_diag
            push!(rows, j)
        end
        colptr[j + 1] = length(rows) + 1
    end
    SparseMatrixCSC(n, n, colptr, rows, zeros(ComplexF64, length(rows)))
end

function _sparse_pattern_positions(A::SparseMatrixCSC, pattern::SparseMatrixCSC)
    positions = Vector{Int}(undef, nnz(A))
    @inbounds for j in axes(A, 2)
        q = pattern.colptr[j]
        q_stop = pattern.colptr[j + 1] - 1
        for p in A.colptr[j]:(A.colptr[j + 1] - 1)
            row = A.rowval[p]
            while q <= q_stop && pattern.rowval[q] < row
                q += 1
            end
            q <= q_stop && pattern.rowval[q] == row ||
                error("internal sparse shift pattern is missing an entry from A")
            positions[p] = q
        end
    end
    positions
end

function _find_sparse_row(A::SparseMatrixCSC, row::Integer, col::Integer)
    @inbounds for p in A.colptr[col]:(A.colptr[col + 1] - 1)
        A.rowval[p] == row && return p
    end
    error("internal sparse shift pattern is missing diagonal entry $row")
end

function _materialize_sparse_standard_shift!(ws::SparseStandardShiftWorkspace, z)
    fill!(ws.shifted.nzval, 0)
    @inbounds for p in eachindex(ws.A.nzval)
        ws.shifted.nzval[ws.a_positions[p]] = ws.A.nzval[p]
    end
    @inbounds for p in ws.diag_positions
        ws.shifted.nzval[p] -= z
    end
    ws.shifted
end

function _sparse_factor(solver::SparseDirectSolver, A::AbstractSparseMatrix)
    lu(A)
end

function _sparse_solve_factored!(Y, solver::SparseDirectSolver, factor, X)
    ldiv!(Y, factor, X)
    Y
end

function _sparse_linsolve!(Y, solver::SparseDirectSolver, A::AbstractSparseMatrix, X)
    factor = _sparse_factor(solver, A)
    _sparse_solve_factored!(Y, solver, factor, X)
    finalize!(factor)
    Y
end

function _sparse_linsolve_reuse_symbolic!(Y, solver::SparseDirectSolver, factor, A::SparseMatrixCSC, X)
    lu!(factor, A; reuse_symbolic=true)
    _sparse_solve_factored!(Y, solver, factor, X)
end

function _sparse_linsolve!(Y, solver::SparseBiCGSTABSolver, A::AbstractSparseMatrix, X)
    max_mv_products = solver.max_mv_products > 0 ? solver.max_mv_products : size(A, 2)
    @inbounds for j in axes(X, 2)
        yj = view(Y, :, j)
        fill!(yj, zero(eltype(Y)))
        bicgstabl!(
            yj,
            A,
            view(X, :, j),
            solver.l;
            abstol=solver.abstol,
            reltol=solver.reltol,
            max_mv_products=max_mv_products,
            initial_zero=true,
        )
    end
    Y
end

"""
    feast!(X, A::AbstractSparseMatrix; solver=SparseDirectSolver(), kwargs...)

Sparse standard FEAST for `Ax = λx`. The Rayleigh-Ritz stages still operate on
dense subspaces, while contour solves use the configured sparse linear solver.
"""
function feast!(
    X::AbstractMatrix,
    A::AbstractSparseMatrix;
    nodes::Integer=8,
    iter::Integer=10,
    c=complex(0.0, 0.0),
    r=1.0,
    ϵ=1e-12,
    debug=false,
    store=false,
    mixed_prec=false,
    solver::AbstractSparseFeastSolver=SparseDirectSolver(),
    stats::Union{Nothing,DenseFeastStats}=nothing,
    contour=nothing,
)
    contour = contour === nothing ? circular_contour_trapezoidal(c, r, nodes) : contour
    feast!(X, A, contour; iter=iter, debug=debug, ϵ=ϵ, store=store, mixed_prec=mixed_prec, solver=solver, stats=stats)
end

function feast!(
    X::AbstractMatrix,
    A::AbstractSparseMatrix,
    contour::Contour;
    iter::Integer=10,
    ϵ=1e-12,
    debug=false,
    store=false,
    mixed_prec=false,
    solver::AbstractSparseFeastSolver=SparseDirectSolver(),
    stats::Union{Nothing,DenseFeastStats}=nothing,
)
    mixed_prec && error("mixed_prec is only supported by dense FEAST")
    store && !_sparse_supports_stored_factors(solver) &&
        error("store=true for sparse FEAST requires a solver with reusable factorizations")

    N, m₀ = size(X)
    if size(A, 1) != size(A, 2)
        error("Incorrect dimensions of A, must be square")
    elseif size(A, 1) != N
        error("Incorrect dimensions of X, must match A")
    end

    Λ, resolvent, res = zeros(ComplexF64, m₀), zeros(ComplexF64, m₀), zeros(m₀)
    inside = falses(m₀)
    temp, R, Q = zeros(ComplexF64, N, m₀), similar(X, ComplexF64), copy(X)
    Aq, Xq = zeros(ComplexF64, m₀, m₀), zeros(ComplexF64, m₀, m₀)
    qr_ws = Q isa StridedMatrix && eltype(Q) <: DenseLapackScalar ? QRWs(Q) : nothing
    eigen_ws = EigenWs(Aq, rvecs=true)
    z_nodes = contour_nodes(contour)
    z_weights = contour_weights(contour)
    node_count = length(z_nodes)
    shift_ws = _sparse_standard_shift_workspace(A)
    reusable_factor = nothing

    stored_factors = nothing
    if store
        first_shift = _materialize_sparse_standard_shift!(shift_ws, z_nodes[1])
        first_factor = _sparse_factor(solver, first_shift)
        stored_factors = Vector{typeof(first_factor)}(undef, node_count)
        stored_factors[1] = first_factor
        for i in 2:node_count
            shift = _materialize_sparse_standard_shift!(shift_ws, z_nodes[i])
            stored_factors[i] = _sparse_factor(solver, shift)
        end
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
        dense_lapack_eigen!(Λ, Xq, Aq, eigen_ws)
        mul!(X, Q, Xq)
        rayleigh_ritz_ns = time_ns() - start_ns

        start_ns = time_ns()
        update_R!(X, R, Λ, A)
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
                elseif solver isa SparseDirectSolver
                    shift = _materialize_sparse_standard_shift!(shift_ws, z_nodes[i])
                    if reusable_factor === nothing
                        reusable_factor = _sparse_factor(solver, shift)
                        _sparse_solve_factored!(temp, solver, reusable_factor, R)
                    else
                        _sparse_linsolve_reuse_symbolic!(temp, solver, reusable_factor, shift, R)
                    end
                else
                    shift = _materialize_sparse_standard_shift!(shift_ws, z_nodes[i])
                    _sparse_linsolve!(temp, solver, shift, R)
                end
                accumulate_filtered_columns!(Q, X, temp, resolvent, z_weights[i])
            end
            filter_ns = time_ns() - start_ns
        end

        _record_dense_feast_iteration!(
            stats,
            :sparse_standard,
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
