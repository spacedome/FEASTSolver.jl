abstract type AbstractSparseFeastSolver end

struct SparseStandardShiftWorkspace{Tv,Ti}
    A::SparseMatrixCSC{Tv,Ti}
    shifted::SparseMatrixCSC{ComplexF64,Ti}
    a_positions::Vector{Int}
    diag_positions::Vector{Int}
end

struct SparseGeneralizedShiftWorkspace{Ta,Tb,Ti}
    A::SparseMatrixCSC{Ta,Ti}
    B::SparseMatrixCSC{Tb,Ti}
    shifted::SparseMatrixCSC{ComplexF64,Ti}
    a_positions::Vector{Int}
    b_positions::Vector{Int}
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

function _sparse_union_pattern(A::SparseMatrixCSC, B::SparseMatrixCSC)
    size(A) == size(B) || error("Incorrect dimensions of A and B, must match")
    n, m = size(A)
    colptr = Vector{promote_type(eltype(A.colptr), eltype(B.colptr))}(undef, m + 1)
    rows = Vector{promote_type(eltype(A.rowval), eltype(B.rowval))}()
    sizehint!(rows, nnz(A) + nnz(B))
    colptr[1] = 1
    @inbounds for j in 1:m
        ap = A.colptr[j]
        bp = B.colptr[j]
        astop = A.colptr[j + 1] - 1
        bstop = B.colptr[j + 1] - 1
        while ap <= astop || bp <= bstop
            arow = ap <= astop ? A.rowval[ap] : n + 1
            brow = bp <= bstop ? B.rowval[bp] : n + 1
            row = min(arow, brow)
            push!(rows, row)
            while ap <= astop && A.rowval[ap] == row
                ap += 1
            end
            while bp <= bstop && B.rowval[bp] == row
                bp += 1
            end
        end
        colptr[j + 1] = length(rows) + 1
    end
    SparseMatrixCSC(n, m, colptr, rows, zeros(ComplexF64, length(rows)))
end

function _sparse_generalized_shift_workspace(A::AbstractSparseMatrix, B::AbstractSparseMatrix)
    size(A, 1) == size(A, 2) || error("Incorrect dimensions of A, must be square")
    size(A) == size(B) || error("Incorrect dimensions of A and B, must match")
    Ac = sparse(A)
    Bc = sparse(B)
    shifted = _sparse_union_pattern(Ac, Bc)
    a_positions = _sparse_pattern_positions(Ac, shifted)
    b_positions = _sparse_pattern_positions(Bc, shifted)
    SparseGeneralizedShiftWorkspace(Ac, Bc, shifted, a_positions, b_positions)
end

function _sparse_generalized_shift_workspace(A::AbstractSparseMatrix, B::UniformScaling)
    ws = _sparse_standard_shift_workspace(A)
    size(ws.A, 1) == size(ws.A, 2) || error("Incorrect dimensions of A, must be square")
    ws
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

function _materialize_sparse_generalized_shift!(ws::SparseGeneralizedShiftWorkspace, z)
    fill!(ws.shifted.nzval, 0)
    @inbounds for p in eachindex(ws.A.nzval)
        ws.shifted.nzval[ws.a_positions[p]] = ws.A.nzval[p]
    end
    @inbounds for p in eachindex(ws.B.nzval)
        ws.shifted.nzval[ws.b_positions[p]] -= z * ws.B.nzval[p]
    end
    ws.shifted
end

function _materialize_sparse_generalized_shift!(ws::SparseStandardShiftWorkspace, z, B::UniformScaling)
    fill!(ws.shifted.nzval, 0)
    @inbounds for p in eachindex(ws.A.nzval)
        ws.shifted.nzval[ws.a_positions[p]] = ws.A.nzval[p]
    end
    @inbounds for p in ws.diag_positions
        ws.shifted.nzval[p] -= z * B.λ
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
