function dual_gen_feast!(Xr::AbstractMatrix, Xl::AbstractMatrix, A::AbstractMatrix, B;
                    nodes::Integer=8, iter::Integer=10, c=complex(0.0,0.0), r=1.0,
                    debug=false, store=false, ϵ=1e-12, factorizer=lu, left_divider=ldiv!,
                    stats::Union{Nothing,DenseFeastStats}=nothing, contour=nothing)
    contour = contour === nothing ? circular_contour_trapezoidal(c, r, nodes) : contour
    dual_gen_feast!(Xr, Xl, A, B, contour; iter=iter, debug=debug, store=store, ϵ=ϵ, factorizer=factorizer, left_divider=left_divider, stats=stats)
end

"""
    dual_gen_feast!(Xr, Xl, A, B, contour; kwargs...)

Bi-orthogonal generalized FEAST for non-normal pencils. Right and left
subspaces are filtered together, then paired through a small SVD so the reduced
generalized eigenproblem is well conditioned.
"""
function dual_gen_feast!(Xr::AbstractMatrix, Xl::AbstractMatrix, A::AbstractMatrix, B, contour::Contour;
                    iter::Integer=10, debug=false, store=false, ϵ=1e-12, factorizer=lu, left_divider=ldiv!,
                    stats::Union{Nothing,DenseFeastStats}=nothing)
    N, m₀ = size(Xl)
    if size(A, 1) != size(A, 2)
        error("Incorrect dimensions of A, must be square")
    elseif size(A,1) != N
        error("Incorrect dimensions of X, must match A")
    end

    Λ, resolvent, resr, resl = zeros(ComplexF64, m₀), zeros(ComplexF64, m₀), zeros(m₀), zeros(m₀)
    inside = falses(m₀)
    temp, Rr, Rl, Ql, Qr = zeros(ComplexF64, N, m₀), similar(Xr, ComplexF64), similar(Xr, ComplexF64), copy(Xl), copy(Xr)
    Aq, Bq = zeros(ComplexF64, m₀, m₀), zeros(ComplexF64, m₀, m₀)
    Xql, Xqr = zeros(ComplexF64, m₀, m₀), zeros(ComplexF64, m₀, m₀)
    ZmA = similar(A, ComplexF64)
    lapack_lu_ws = use_dense_lapack_lu(A, Xr, store, factorizer, left_divider, ComplexF64) ?
                   dense_lapack_lu_workspace(ZmA) : nothing
    share_adjoint_factors = factorizer === lu && left_divider === ldiv!
    eigen_ws = GeneralizedEigenWs(Aq, lvecs=true, rvecs=true)
    svd_ws = SVDsddWs(Bq, job='A')
    residual_shift = similar(A, ComplexF64)
    residual_x = zeros(ComplexF64, N)
    residual_y = zeros(ComplexF64, N)
    nodes = size(contour.nodes, 1)

    use_dense_lapack_store = store && factorizer === lu && left_divider === ldiv! &&
                             A isa StridedMatrix && B isa Union{StridedMatrix, UniformScaling}
    stored_shifts = nothing
    stored_lu_ws = nothing
    rfacts = nothing
    lfacts = nothing
    if store
         if use_dense_lapack_store
             stored_shifts = [similar(ZmA) for _ in 1:nodes]
             stored_lu_ws = [dense_lapack_lu_workspace(stored_shifts[i]) for i in 1:nodes]
             for i in 1:nodes
                   materialize_generalized_shift!(stored_shifts[i], A, B, contour.nodes[i])
                   dense_lapack_factor!(stored_shifts[i], stored_lu_ws[i])
             end
         else
             materialize_generalized_shift!(ZmA, A, B, contour.nodes[1])
             rfacts1 = factorizer(ZmA)
             rfacts = Array{typeof(rfacts1)}(undef, nodes)
             rfacts[1] = rfacts1
             if !share_adjoint_factors
                 materialize_adjoint_generalized_shift!(ZmA, A, B, contour.nodes[1])
                 lfacts1 = factorizer(ZmA)
                 lfacts = Array{typeof(lfacts1)}(undef, nodes)
                 lfacts[1] = lfacts1
             end

             for i=2:nodes
                   local_shift = similar(ZmA)
                   materialize_generalized_shift!(local_shift, A, B, contour.nodes[i])
                   rfacts[i] = factorizer(local_shift)
                   if !share_adjoint_factors
                       local_adjoint_shift = similar(ZmA)
                       materialize_adjoint_generalized_shift!(local_adjoint_shift, A, B, contour.nodes[i])
                       lfacts[i] = factorizer(local_adjoint_shift)
                   end
             end
         end
    end

    solve_start_ns = time_ns()
    for nit=0:iter
        iter_start_ns = time_ns()
        orthogonalization_ns = UInt64(0)
        rayleigh_ritz_ns = UInt64(0)
        residual_ns = UInt64(0)
        filter_ns = UInt64(0)

        # Bi-orthogonalize the left/right subspaces through the B inner product.
        start_ns = time_ns()
        mul!(Rr, B, Qr)
        mul!(Bq, Ql', Rr)
        U, S, Vt = dense_lapack_svd!(Bq, svd_ws)
        mul!(Rr, Qr, adjoint(Vt))
        Qr .= Rr
        inv_scale_columns!(Qr, S)
        mul!(Rl, Ql, U)
        Ql .= Rl
        inv_scale_columns!(Ql, S)
        orthogonalization_ns = time_ns() - start_ns

        # Rayleigh-Ritz extraction for the paired left/right subspaces.
        start_ns = time_ns()
        mul!(Rr, A, Qr)
        mul!(Aq, Ql', Rr)
        mul!(Rr, B, Qr)
        mul!(Bq, Ql', Rr)
        dense_lapack_generalized_eigen!(Λ, Xql, Xqr, Aq, Bq, eigen_ws)
        mul!(Xr, Qr, Xqr)
        mul!(Xl, Ql, Xql)
        rayleigh_ritz_ns = time_ns() - start_ns

        # Right and left residuals drive the next rational filter update.
        start_ns = time_ns()
        update_R_shifted!(Xr, Rr, Λ, A, B, residual_shift, residual_x, residual_y)
        update_R_shifted!(Xl, Rl, Λ, A', B', residual_shift, residual_x, residual_y)
        residuals!(resr, Rr, Λ, A)
        in_contour!(inside, Λ, contour)
        max_res_inside, contour_nonempty = maximum_masked(resr, inside)
        residual_ns = time_ns() - start_ns
        if debug iter_debug_print(nit, Λ, resr, contour, 1e-5) end
        converged = contour_nonempty && max_res_inside < ϵ
        if converged
              if debug println("converged in $nit iteration") end
        end
        if !converged && nit < iter
            # Filter the right subspace with `A - zB` and the left subspace
            # with its adjoint. Standard LU factors can be reused via `trans='C'`.
            start_ns = time_ns()
            Qr .= 0.00
            Ql .= 0.00
            for i=1:nodes
                fill_resolvent!(resolvent, contour.nodes[i], Λ)
                if use_dense_lapack_store
                    dense_lapack_solve_factored!(temp, stored_shifts[i], Rr, stored_lu_ws[i], 'N')
                elseif store
                    left_divider(temp, rfacts[i], Rr)
                elseif lapack_lu_ws !== nothing
                    materialize_generalized_shift!(ZmA, A, B, contour.nodes[i])
                    dense_lapack_factor!(ZmA, lapack_lu_ws)
                    dense_lapack_solve_factored!(temp, ZmA, Rr, lapack_lu_ws, 'N')
                else
                    materialize_generalized_shift!(ZmA, A, B, contour.nodes[i])
                    linsolve!(temp, ZmA, Rr, factorizer, left_divider)
                end
                accumulate_filtered_columns!(Qr, Xr, temp, resolvent, contour.weights[i])

                fill_adjoint_resolvent!(resolvent, contour.nodes[i], Λ)
                if use_dense_lapack_store
                    dense_lapack_solve_factored!(temp, stored_shifts[i], Rl, stored_lu_ws[i], 'C')
                elseif store
                    if share_adjoint_factors
                        left_divider(temp, adjoint(rfacts[i]), Rl)
                    else
                        left_divider(temp, lfacts[i], Rl)
                    end
                elseif lapack_lu_ws !== nothing
                    dense_lapack_solve_factored!(temp, ZmA, Rl, lapack_lu_ws, 'C')
                else
                    materialize_adjoint_generalized_shift!(ZmA, A, B, contour.nodes[i])
                    linsolve!(temp, ZmA, Rl, factorizer, left_divider)
                end
                accumulate_filtered_columns!(Ql, Xl, temp, resolvent, conj(contour.weights[i]))
            end
            filter_ns = time_ns() - start_ns
        end
        _record_dense_feast_iteration!(
            stats,
            :dual_generalized,
            nit,
            resr,
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
        if converged
            break
        end
    end
    if stats !== nothing
        stats.solve_total_ns += time_ns() - solve_start_ns
    end
    if rfacts !== nothing
        foreach(finalize!, rfacts)
        if lfacts !== nothing
            foreach(finalize!, lfacts)
        end
    end
    in_contour!(inside, Λ, contour)
    _, contour_nonempty = maximum_masked(resr, inside)
    if !contour_nonempty println("no eigenvalues found in contour!") end
    Λ[inside], Xr[:, inside], Xl[:, inside], resr[inside]
end
