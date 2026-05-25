function distributed_dual_gen_feast!(
    Xr::AbstractMatrix,
    Xl::AbstractMatrix,
    A::AbstractMatrix,
    B;
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
    distributed_dual_gen_feast!(
        Xr,
        Xl,
        A,
        B,
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

function distributed_dual_gen_feast!(
    Xr::AbstractMatrix,
    Xl::AbstractMatrix,
    A::AbstractMatrix,
    B,
    contour::Contour;
    iter::Integer=10,
    ϵ=1e-12,
    debug=false,
    store=false,
    worker_ids=_default_feast_worker_ids(),
    worker_blas_threads::Integer=1,
    stats::Union{Nothing, DenseDistributedFeastStats}=nothing,
)
    plan = DenseDistributedDualGeneralizedFeastPlan(
        A,
        B,
        size(Xr, 2),
        contour;
        store=store,
        worker_ids=worker_ids,
        worker_blas_threads=worker_blas_threads,
        stats=stats,
    )
    try
        distributed_dual_gen_feast!(Xr, Xl, plan; iter=iter, ϵ=ϵ, debug=debug, stats=stats)
    finally
        close(plan)
    end
end

function distributed_dual_gen_feast!(
    Xr::AbstractMatrix,
    Xl::AbstractMatrix,
    plan::DenseDistributedDualGeneralizedFeastPlan;
    iter::Integer=10,
    ϵ=1e-12,
    debug=false,
    stats::Union{Nothing, DenseDistributedFeastStats}=nothing,
)
    plan.closed && error("DenseDistributedDualGeneralizedFeastPlan is closed")
    stats = stats === nothing ? plan.stats : stats
    N, m₀ = size(plan.Xr_buffer)
    size(Xr) == (N, m₀) || error("Incorrect dimensions of Xr, must match planned A and subspace dimension")
    size(Xl) == (N, m₀) || error("Incorrect dimensions of Xl, must match planned A and subspace dimension")

    A = plan.A
    B = plan.B
    contour = plan.contour
    Λ = plan.Λ
    resr = plan.resr
    inside = plan.inside
    Rr = plan.Rr
    Rl = plan.Rl
    Qr = plan.Qr
    Ql = plan.Ql
    Aq = plan.Aq
    Bq = plan.Bq
    Xql = plan.Xql
    Xqr = plan.Xqr

    solve_start_ns = time_ns()
    copyto!(Qr, Xr)
    copyto!(Ql, Xl)
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
        mul!(Rr, B, Qr)
        mul!(Bq, Ql', Rr)
        U, S, Vt = dense_lapack_svd!(Bq, plan.svd_ws)
        mul!(Rr, Qr, adjoint(Vt))
        Qr .= Rr
        inv_scale_columns!(Qr, S)
        mul!(Rl, Ql, U)
        Ql .= Rl
        inv_scale_columns!(Ql, S)
        qr_ns = time_ns() - start_ns
        _add_ns!(stats, :qr_ns, qr_ns)

        start_ns = time_ns()
        mul!(Rr, A, Qr)
        mul!(Aq, Ql', Rr)
        mul!(Rr, B, Qr)
        mul!(Bq, Ql', Rr)
        dense_lapack_generalized_eigen!(Λ, Xql, Xqr, Aq, Bq, plan.eigen_ws)
        mul!(Xr, Qr, Xqr)
        mul!(Xl, Ql, Xql)
        rayleigh_ritz_ns = time_ns() - start_ns
        _add_ns!(stats, :rayleigh_ritz_ns, rayleigh_ritz_ns)

        start_ns = time_ns()
        update_R_shifted!(Xr, Rr, Λ, A, B, plan.residual_shift, plan.residual_x, plan.residual_y)
        update_R_shifted!(Xl, Rl, Λ, A', B', plan.residual_shift, plan.residual_x, plan.residual_y)
        residuals!(resr, Rr, Λ, A)
        in_contour!(inside, Λ, contour)
        max_res_inside, contour_nonempty = maximum_masked(resr, inside)
        residual_ns = time_ns() - start_ns
        _add_ns!(stats, :residual_ns, residual_ns)

        if debug
            iter_debug_print(nit, Λ, resr, contour, 1e-5)
        end
        converged = contour_nonempty && max_res_inside < ϵ
        if converged && debug
            println("converged in $nit iteration")
        end
        if !converged && nit < iter
            start_ns = time_ns()
            for (i, pid) in enumerate(plan.worker_ids)
                plan.futures[i] = remotecall(_dense_dual_gen_feast_worker_step!, pid, plan.key, Xr, Xl, Rr, Rl, Λ)
            end
            for (i, future) in enumerate(plan.futures)
                worker_qparts = fetch(future)
                copyto!(plan.Qrparts[i], worker_qparts.Qrpart)
                copyto!(plan.Qlparts[i], worker_qparts.Qlpart)
            end
            worker_step_ns = time_ns() - start_ns
            _add_ns!(stats, :worker_step_ns, worker_step_ns)

            start_ns = time_ns()
            _sum_dense_feast_qparts!(Qr, plan.Qrparts)
            _sum_dense_feast_qparts!(Ql, plan.Qlparts)
            reduce_ns = time_ns() - start_ns
            _add_ns!(stats, :reduce_ns, reduce_ns)
        end

        _record_distributed_feast_iteration!(
            stats,
            nit,
            resr,
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
    _, contour_nonempty = maximum_masked(resr, inside)
    if !contour_nonempty
        println("no eigenvalues found in contour!")
    end
    Λ[inside], Xr[:, inside], Xl[:, inside], resr[inside]
end
