# Worker-local state registry and shared remote helpers for dense distributed FEAST.
#
# Variant-specific worker kernels live in `distributed/*_workers.jl`. This file
# keeps only shared process setup, reductions, cleanup, and plan closing.

const _DISTRIBUTED_DENSE_FEAST_WORKSPACES = Dict{Symbol, Any}()

function _prepare_dense_feast_workers!(worker_ids)
    for pid in worker_ids
        remotecall_wait(Main.eval, pid, :(using LinearAlgebra; using FEASTSolver))
    end
    nothing
end

function _sum_dense_feast_qparts!(Q::AbstractMatrix, Qparts)
    if isempty(Qparts)
        fill!(Q, 0)
        return Q
    end
    copyto!(Q, first(Qparts))
    for Qpart in Iterators.drop(Qparts, 1)
        axpy!(one(eltype(Q)), Qpart, Q)
    end
    Q
end

function _cleanup_dense_feast_worker!(key::Symbol)
    ws = pop!(_DISTRIBUTED_DENSE_FEAST_WORKSPACES, key, nothing)
    if ws !== nothing
        if hasproperty(ws, :stored_factors) && ws.stored_factors !== nothing
            foreach(finalize!, ws.stored_factors)
        end
        if hasproperty(ws, :sparse_reusable_factor) && ws.sparse_reusable_factor !== nothing
            finalize!(ws.sparse_reusable_factor)
        end
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
