function gen_feast!(X::AbstractMatrix, A::AbstractMatrix, B;
                    nodes::Integer=8, iter::Integer=10, c=complex(0.0,0.0), r=1.0,
                    debug=false, store=false, ϵ=1e-12, factorizer=lu, left_divider=ldiv!,
                    stats::Union{Nothing,DenseFeastStats}=nothing, contour=nothing)
    contour = contour === nothing ? circular_contour_trapezoidal(c, r, nodes) : contour
    gen_feast!(X, A, B, contour; iter=iter, debug=debug, store=store, ϵ=ϵ, factorizer=factorizer, left_divider=left_divider, stats=stats)
end

"""
    gen_feast!(X, A, B, contour; kwargs...)

Dense generalized FEAST for `Ax = λBx`. This mirrors `feast!`, but the reduced
problem is `Q'AQ y = λ Q'BQ y` and each contour solve uses `A - zB`.
"""
function gen_feast!(X::AbstractMatrix, A::AbstractMatrix, B, contour::Contour;
                    iter::Integer=10, debug=false, store=false, ϵ=1e-12, factorizer=lu, left_divider=ldiv!,
                    stats::Union{Nothing,DenseFeastStats}=nothing)
    N, m₀ = size(X)
    if size(A, 1) != size(A, 2)
        error("Incorrect dimensions of A, must be square")
    elseif size(A,1) != N
        error("Incorrect dimensions of X, must match A")
    end

    Λ, resolvent, res = zeros(ComplexF64, m₀), zeros(ComplexF64, m₀), zeros(m₀)
    inside = falses(m₀)
    temp, R, Q = zeros(ComplexF64, N, m₀), similar(X, ComplexF64), copy(X)
    Aq, Bq, Xq = zeros(ComplexF64, m₀, m₀), zeros(ComplexF64, m₀, m₀), zeros(ComplexF64, m₀, m₀)
    ZmA = similar(A, ComplexF64)
    lapack_lu_ws = use_dense_lapack_lu(A, X, store, factorizer, left_divider, ComplexF64) ?
                   dense_lapack_lu_workspace(ZmA) : nothing
    qr_ws = Q isa StridedMatrix ? QRWs(Q) : nothing
    eigen_ws = GeneralizedEigenWs(Aq, lvecs=true, rvecs=true)
    nodes = size(contour.nodes, 1)

    use_dense_lapack_store = store && factorizer === lu && left_divider === ldiv! &&
                             A isa StridedMatrix && B isa Union{StridedMatrix, UniformScaling}
    stored_shifts = nothing
    stored_lu_ws = nothing
    facts = nothing
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
             facts1 = factorizer(ZmA)
             facts = Array{typeof(facts1)}(undef, nodes)
             facts[1] = facts1

             for i=2:nodes
                   local_shift = similar(ZmA)
                   materialize_generalized_shift!(local_shift, A, B, contour.nodes[i])
                   facts[i] = factorizer(local_shift)
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

        # Rayleigh-Ritz extraction for the projected generalized pencil.
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

        # R holds `(A - λB)x` and is reused as the right-hand side in RII.
        start_ns = time_ns()
        update_R!(X, R, Λ, A, B, temp)
        residuals!(res, R, Λ, A)
        in_contour!(inside, Λ, contour)
        max_res_inside, contour_nonempty = maximum_masked(res, inside)
        residual_ns = time_ns() - start_ns
        if debug iter_debug_print(nit, Λ, res, contour, 1e-5) end
        converged = contour_nonempty && max_res_inside < ϵ
        if converged
              if debug println("converged in $nit iteration") end
        end
        if !converged && nit < iter
            # Rational inverse iteration update with shifted pencil `A - zB`.
            start_ns = time_ns()
            Q .= 0.00
            for i=1:nodes
                fill_resolvent!(resolvent, contour.nodes[i], Λ)
                if use_dense_lapack_store
                    dense_lapack_solve_factored!(temp, stored_shifts[i], R, stored_lu_ws[i])
                elseif store
                    left_divider(temp, facts[i], R)
                elseif lapack_lu_ws !== nothing
                    materialize_generalized_shift!(ZmA, A, B, contour.nodes[i])
                    dense_lapack_linsolve!(temp, ZmA, R, lapack_lu_ws)
                else
                    materialize_generalized_shift!(ZmA, A, B, contour.nodes[i])
                    linsolve!(temp, ZmA, R, factorizer, left_divider)
                end
                accumulate_filtered_columns!(Q, X, temp, resolvent, contour.weights[i])
            end
            filter_ns = time_ns() - start_ns
        end
        _record_dense_feast_iteration!(
            stats,
            :generalized,
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
        if converged
            break
        end
    end
    if stats !== nothing
        stats.solve_total_ns += time_ns() - solve_start_ns
    end
    if facts !== nothing
        foreach(finalize!, facts)
    end
    in_contour!(inside, Λ, contour)
    _, contour_nonempty = maximum_masked(res, inside)
    if !contour_nonempty println("no eigenvalues found in contour!") end
    Λ[inside], X[:, inside], res[inside]
end
