function distributed_nlfeast!(
    T,
    X::AbstractMatrix{ComplexF64},
    nodes::Integer,
    iter::Integer;
    c=complex(0.0, 0.0),
    r=1.0,
    contour=nothing,
    kwargs...,
)
    contour = contour === nothing ? circular_contour_trapezoidal(c, r, nodes) : contour
    distributed_nlfeast!(T, X, contour, iter; kwargs...)
end

function distributed_nlfeast!(
    T,
    X::AbstractMatrix{ComplexF64},
    contour::Contour,
    iter::Integer;
    store=true,
    materialize_nodes=true,
    matrix_update=nothing,
    matrix_prototype=nothing,
    worker_ids=_default_feast_worker_ids(),
    worker_blas_threads::Integer=1,
    factorizer=lu,
    left_divider=ldiv!,
    stats::Union{Nothing, DenseDistributedFeastStats}=nothing,
    kwargs...,
)
    plan = DenseDistributedNonlinearFeastPlan(
        T,
        size(X, 1),
        size(X, 2),
        contour;
        store=store,
        materialize_nodes=materialize_nodes,
        matrix_update=matrix_update,
        matrix_prototype=matrix_prototype,
        worker_ids=worker_ids,
        worker_blas_threads=worker_blas_threads,
        factorizer=factorizer,
        left_divider=left_divider,
        stats=stats,
    )
    try
        distributed_nlfeast!(X, plan, iter; stats=stats, kwargs...)
    finally
        close(plan)
    end
end

function distributed_nlfeast!(
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
    distributed_nlfeast!(matrix_update, matrix_prototype, X, contour, iter; kwargs...)
end

function distributed_nlfeast!(
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
    distributed_nlfeast!(
        materializing_T,
        X,
        contour,
        iter;
        matrix_update=matrix_update,
        matrix_prototype=matrix_prototype,
        kwargs...,
    )
end

function distributed_nlfeast!(
    X::AbstractMatrix{ComplexF64},
    plan::DenseDistributedNonlinearFeastPlan,
    iter::Integer;
    ϵ=10e-12,
    debug=false,
    spurious=1e-5,
    residual_update=nothing,
    stats::Union{Nothing, DenseDistributedFeastStats}=nothing,
)
    plan.closed && error("DenseDistributedNonlinearFeastPlan is closed")
    stats = stats === nothing ? plan.stats : stats
    N, m₀ = size(plan.X_buffer)
    size(X) == (N, m₀) || error("Incorrect dimensions of X, must match planned problem and subspace dimension")

    T = plan.T
    contour = plan.contour
    Λ = plan.Λ
    res = plan.res
    inside = plan.inside
    R = plan.R
    Q₀ = plan.Q₀
    Q₁ = plan.Q₁
    A = plan.A
    B = plan.B
    Xq = plan.Xq

    dense_lapack_qr!(X, plan.qr_ws)

    solve_start_ns = time_ns()
    for nit=0:iter
        iter_start_ns = time_ns()
        rayleigh_ritz_ns = UInt64(0)
        residual_ns = UInt64(0)
        input_transfer_ns = UInt64(0)
        worker_step_ns = UInt64(0)
        worker_solve_ns = UInt64(0)
        worker_materialize_ns = UInt64(0)
        worker_linsolve_ns = UInt64(0)
        worker_accum_ns = UInt64(0)
        reduce_ns = UInt64(0)

        if stats !== nothing
            stats.iterations += 1
        end

        start_ns = time_ns()
        first_iteration = nit == 0
        for (i, pid) in enumerate(plan.worker_ids)
            rhs = first_iteration ? X : R
            plan.futures[i] = remotecall(_dense_nlfeast_worker_step!, pid, plan.key, X, rhs, Λ, first_iteration)
        end
        for (i, future) in enumerate(plan.futures)
            worker_report = fetch(future)
            copyto!(plan.Q₀parts[i], worker_report.Q₀part)
            copyto!(plan.Q₁parts[i], worker_report.Q₁part)
            worker_solve_ns += worker_report.solve_ns
            worker_materialize_ns += worker_report.materialize_ns
            worker_linsolve_ns += worker_report.linsolve_ns
            worker_accum_ns += worker_report.accum_ns
        end
        worker_step_ns = time_ns() - start_ns
        _add_ns!(stats, :worker_step_ns, worker_step_ns)
        _add_ns!(stats, :worker_solve_ns, worker_solve_ns)
        _add_ns!(stats, :worker_materialize_ns, worker_materialize_ns)
        _add_ns!(stats, :worker_linsolve_ns, worker_linsolve_ns)
        _add_ns!(stats, :worker_accum_ns, worker_accum_ns)

        start_ns = time_ns()
        _sum_dense_feast_qparts!(Q₀, plan.Q₀parts)
        _sum_dense_feast_qparts!(Q₁, plan.Q₁parts)
        reduce_ns = time_ns() - start_ns
        _add_ns!(stats, :reduce_ns, reduce_ns)

        start_ns = time_ns()
        beyn_svd_step!(Q₀, Q₁, A, B, X, Λ, plan.svd_ws, plan.eigen_ws, Xq)
        rayleigh_ritz_ns = time_ns() - start_ns
        _add_ns!(stats, :rayleigh_ritz_ns, rayleigh_ritz_ns)

        start_ns = time_ns()
        if residual_update !== nothing
            # Expert fast path for action-only NEPs that cannot cheaply form T(λ).
            residual_update(res, X, R, Λ)
        elseif plan.residual_matrix === nothing
            Tλ = T(first(Λ))
            if Tλ isa StridedMatrix
                plan.residual_matrix = similar(Tλ)
                update_nonlinear_residuals!(res, X, R, Λ, T, plan.residual_matrix, plan.residual_x, plan.residual_y)
            else
                update_nonlinear_residuals!(res, X, R, Λ, T, plan.residual_x, plan.residual_y)
            end
        else
            update_nonlinear_residuals!(res, X, R, Λ, T, plan.residual_matrix, plan.residual_x, plan.residual_y)
        end
        in_contour!(inside, Λ, contour)
        max_res_inside, contour_nonempty = maximum_masked(res, inside)
        residual_ns = time_ns() - start_ns
        _add_ns!(stats, :residual_ns, residual_ns)

        if debug
            iter_debug_print(nit, Λ, res, contour, spurious)
        end

        converged = contour_nonempty && max_res_inside < ϵ
        max_spurious_res_inside, spurious_found = maximum_below_masked(res, inside, spurious)
        spurious_converged = nit > 1 && spurious_found && max_spurious_res_inside < ϵ
        _record_distributed_feast_iteration!(
            stats,
            nit,
            res,
            inside,
            max_res_inside,
            contour_nonempty,
            iter_start_ns,
            UInt64(0),
            rayleigh_ritz_ns,
            residual_ns,
            input_transfer_ns,
            worker_step_ns,
            reduce_ns,
            debug,
        )

        if converged || spurious_converged
            break
        end
    end
    _add_elapsed!(stats, :solve_total_ns, solve_start_ns)

    normalize!(X)
    Λ, X, res
end
