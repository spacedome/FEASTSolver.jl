function distributed_feast!(
    X::AbstractMatrix,
    A::AbstractMatrix;
    nodes::Integer=8,
    iter::Integer=10,
    c=complex(0.0, 0.0),
    r=1.0,
    contour=nothing,
    ϵ=1e-12,
    debug=false,
    store=false,
    worker_ids=_default_feast_worker_ids(),
    worker_blas_threads::Integer=1,
    stats::Union{Nothing, DenseDistributedFeastStats}=nothing,
)
    contour = contour === nothing ? circular_contour_trapezoidal(c, r, nodes) : contour
    distributed_feast!(
        X,
        A,
        contour;
        iter=iter,
        ϵ=ϵ,
        debug=debug,
        store=store,
        worker_ids=worker_ids,
        worker_blas_threads=worker_blas_threads,
        stats=stats,
    )
end

function distributed_feast!(
    X::AbstractMatrix,
    A::AbstractMatrix,
    contour::Contour;
    iter::Integer=10,
    ϵ=1e-12,
    debug=false,
    store=false,
    worker_ids=_default_feast_worker_ids(),
    worker_blas_threads::Integer=1,
    stats::Union{Nothing, DenseDistributedFeastStats}=nothing,
)
    plan = DenseDistributedFeastPlan(
        A,
        size(X, 2),
        contour;
        store=store,
        worker_ids=worker_ids,
        worker_blas_threads=worker_blas_threads,
        stats=stats,
    )
    try
        distributed_feast!(X, plan; iter=iter, ϵ=ϵ, debug=debug, stats=stats)
    finally
        close(plan)
    end
end

function distributed_feast!(
    X::AbstractMatrix,
    plan::DenseDistributedFeastPlan;
    iter::Integer=10,
    ϵ=1e-12,
    debug=false,
    stats::Union{Nothing, DenseDistributedFeastStats}=nothing,
)
    plan.closed && error("DenseDistributedFeastPlan is closed")
    stats = stats === nothing ? plan.stats : stats
    N, m₀ = size(plan.X_buffer)
    size(X) == (N, m₀) || error("Incorrect dimensions of X, must match planned A and subspace dimension")

    A = plan.A
    contour = plan.contour
    Λ = plan.Λ
    res = plan.res
    inside = plan.inside
    R = plan.R
    Q = plan.Q
    Aq = plan.Aq
    Xq = plan.Xq

    solve_start_ns = time_ns()
    copyto!(Q, X)
    for nit=0:iter
        iter_start_ns = time_ns()
        qr_ns = UInt64(0)
        rayleigh_ritz_ns = UInt64(0)
        residual_ns = UInt64(0)
        input_transfer_ns = UInt64(0)
        worker_step_ns = UInt64(0)
        reduce_ns = UInt64(0)

        if stats !== nothing
            stats.iterations += 1
        end
        start_ns = time_ns()
        dense_lapack_qr!(Q, plan.qr_ws)
        qr_ns = time_ns() - start_ns
        _add_ns!(stats, :qr_ns, qr_ns)

        start_ns = time_ns()
        mul!(R, A, Q)
        mul!(Aq, Q', R)
        dense_lapack_eigen!(Λ, Xq, Aq, plan.eigen_ws)
        rayleigh_ritz_ns = time_ns() - start_ns
        _add_ns!(stats, :rayleigh_ritz_ns, rayleigh_ritz_ns)

        start_ns = time_ns()
        mul!(X, Q, Xq)
        update_R!(X, R, Λ, A)
        residuals!(res, R, Λ, A)
        in_contour!(inside, Λ, contour)
        max_res_inside, contour_nonempty = maximum_masked(res, inside)
        residual_ns = time_ns() - start_ns
        _add_ns!(stats, :residual_ns, residual_ns)

        if debug
            iter_debug_print(nit, Λ, res, contour, 1e-5)
        end
        converged = contour_nonempty && max_res_inside < ϵ
        if converged
            if debug
                println("converged in $nit iteration")
            end
        end
        if !converged && nit < iter
            start_ns = time_ns()
            for (i, pid) in enumerate(plan.worker_ids)
                plan.futures[i] = remotecall(_dense_feast_worker_step!, pid, plan.key, X, R, Λ)
            end
            for (i, future) in enumerate(plan.futures)
                copyto!(plan.Qparts[i], fetch(future))
            end
            worker_step_ns = time_ns() - start_ns
            _add_ns!(stats, :worker_step_ns, worker_step_ns)

            start_ns = time_ns()
            _sum_dense_feast_qparts!(Q, plan.Qparts)
            reduce_ns = time_ns() - start_ns
            _add_ns!(stats, :reduce_ns, reduce_ns)
        end

        _record_distributed_feast_iteration!(
            stats,
            nit,
            res,
            inside,
            max_res_inside,
            contour_nonempty,
            iter_start_ns,
            qr_ns,
            rayleigh_ritz_ns,
            residual_ns,
            input_transfer_ns,
            worker_step_ns,
            reduce_ns,
            debug,
        )

        if converged
            break
        end
    end
    _add_elapsed!(stats, :solve_total_ns, solve_start_ns)

    in_contour!(inside, Λ, contour)
    _, contour_nonempty = maximum_masked(res, inside)
    if !contour_nonempty
        println("no eigenvalues found in contour!")
    end
    Λ[inside], X[:, inside], res[inside]
end
