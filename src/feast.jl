import LinearAlgebra: qr, lu, eigen

function feast!(X::AbstractMatrix, A::AbstractMatrix;
                nodes::Integer=8, iter::Integer=10, c=complex(0.0,0.0), r=1.0, ϵ=1e-12,
                debug=false, store=false, mixed_prec=false, factorizer=lu, left_divider=ldiv!)
    contour = circular_contour_trapezoidal(c, r, nodes)
    feast!(X, A, contour; iter=iter, debug=debug, ϵ=ϵ, store=store, mixed_prec=mixed_prec, factorizer=factorizer, left_divider=left_divider)
end

function feast!(X::AbstractMatrix, A::AbstractMatrix, contour::Contour;
                     iter::Integer=10, ϵ=1e-12, debug=false, store=false, mixed_prec=false, factorizer=lu, left_divider=ldiv!)
    N, m₀ = size(X)
    if size(A, 1) != size(A, 2)
         error("Incorrect dimensions of A, must be square")
    elseif size(A,1) != N
         error("Incorrect dimensions of X, must match A")
    end

    Ctype = if mixed_prec ComplexF32 else ComplexF64 end

    Λ, resolvent, res = zeros(ComplexF64, m₀), zeros(ComplexF64, m₀), zeros(m₀)
    inside = falses(m₀)
    temp, R, Q = zeros(Ctype, N, m₀), similar(X, ComplexF64), copy(X)
    Aq, Xq = zeros(ComplexF64, m₀, m₀), zeros(ComplexF64, m₀, m₀)

    ZmA = similar(A, Ctype)
    lapack_lu_ws = use_dense_lapack_lu(A, X, store, factorizer, left_divider, Ctype) ?
                   dense_lapack_lu_workspace(ZmA) : nothing
    qr_ws = Q isa StridedMatrix && eltype(Q) <: DenseLapackScalar ? QRWs(Q) : nothing
    eigen_ws = EigenWs(Aq, rvecs=true)
    nodes = size(contour.nodes, 1)

    if store
        materialize_standard_shift!(ZmA, A, contour.nodes[1])
        facts1 = factorizer(ZmA)
        facts = Array{typeof(facts1)}(undef, nodes)
        facts[1] = facts1

        Threads.@threads for i=2:nodes
              local_shift = similar(ZmA)
              materialize_standard_shift!(local_shift, A, contour.nodes[i])
              facts[i] = factorizer(local_shift)
        end
    end

    for nit=0:iter
        if qr_ws === nothing
            Q .= Matrix(qr(Q).Q)
        else
            dense_lapack_qr!(Q, qr_ws)
        end
        mul!(R, A, Q) ## why does this one allocate?
        mul!(Aq, Q', R) ### Aq = Q' * A * Q
        # mul!(Bq, Q', Q) ### Bq = Q' * Q = I
        dense_lapack_eigen!(Λ, Xq, Aq, eigen_ws)
        mul!(X, Q, Xq) ### Recover eigenvectors from Ritz vectors ( X = Q * Xq )
        update_R!(X, R, Λ, A) ### compute residual vectors R for RII update
        residuals!(res, R, Λ, A) ### compute actual residuals
        in_contour!(inside, Λ, contour)
        max_res_inside, contour_nonempty = maximum_masked(res, inside)
        if debug iter_debug_print(nit, Λ, res, contour, 1e-5) end
        if contour_nonempty && max_res_inside < ϵ
            if debug println("converged in $nit iteration") end
            break
        end
        if nit < iter ### Do not solve linear systems / form Q on last iteration
            Q .= 0.00
            for i=1:nodes
                fill_resolvent!(resolvent, contour.nodes[i], Λ)
                if store
                    left_divider(temp, facts[i], R)
                elseif lapack_lu_ws !== nothing
                    materialize_standard_shift!(ZmA, A, contour.nodes[i])
                    dense_lapack_linsolve!(temp, ZmA, R, lapack_lu_ws)
                else
                    materialize_standard_shift!(ZmA, A, contour.nodes[i])
                    linsolve!(temp, ZmA, R, factorizer, left_divider)
                end

                temp .= X .- temp
                scale_columns!(temp, resolvent, contour.weights[i])
                Q .+= temp
            end
        end
    end
    if store
        foreach(finalize!, facts)
    end
    in_contour!(inside, Λ, contour)
    _, contour_nonempty = maximum_masked(res, inside)
    if !contour_nonempty println("no eigenvalues found in contour!") end
    Λ[inside], X[:, inside], res[inside]
end

function gen_feast!(X::AbstractMatrix, A::AbstractMatrix, B;
                    nodes::Integer=8, iter::Integer=10, c=complex(0.0,0.0), r=1.0,
                    debug=false, store=false, ϵ=1e-12, factorizer=lu, left_divider=ldiv!)
    contour = circular_contour_trapezoidal(c, r, nodes)
    gen_feast!(X, A, B, contour; iter=iter, debug=debug, ϵ=ϵ, factorizer=factorizer, left_divider=left_divider)
end

function gen_feast!(X::AbstractMatrix, A::AbstractMatrix, B, contour::Contour;
                    iter::Integer=10, debug=false, store=false, ϵ=1e-12, factorizer=lu, left_divider=ldiv!)
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

    if store
         materialize_generalized_shift!(ZmA, A, B, contour.nodes[1])
         facts1 = factorizer(ZmA)
         facts = Array{typeof(facts1)}(undef, nodes)
         facts[1] = facts1

         Threads.@threads for i=2:nodes
               local_shift = similar(ZmA)
               materialize_generalized_shift!(local_shift, A, B, contour.nodes[i])
               facts[i] = factorizer(local_shift)
         end
    end

    for nit=0:iter
        if qr_ws === nothing
            Q .= Matrix(qr(Q).Q)
        else
            dense_lapack_qr!(Q, qr_ws)
        end
        mul!(R, A, Q) ## why does this one allocate?
        mul!(Aq, Q', R) ### Aq = Q' * A * Q
        mul!(R, B, Q)
        mul!(Bq, Q', R) ### Bq = Q' * Q = I
        dense_lapack_generalized_eigen!(Λ, Xq, Aq, Bq, eigen_ws)
        mul!(X, Q, Xq) ### Recover eigenvectors from Ritz vectors ( X = Q * Xq )
        update_R!(X, R, Λ, A, B, temp) ### compute residual vectors R for RII update
        residuals!(res, R, Λ, A) ### compute actual residuals
        in_contour!(inside, Λ, contour)
        max_res_inside, contour_nonempty = maximum_masked(res, inside)
        if debug iter_debug_print(nit, Λ, res, contour, 1e-5) end
        if contour_nonempty && max_res_inside < ϵ
              if debug println("converged in $nit iteration") end
              break
        end
        if nit < iter ### Do not solve linear systems / form Q on last iteration
            Q .= 0.00
            for i=1:nodes
                fill_resolvent!(resolvent, contour.nodes[i], Λ)
                if store
                    left_divider(temp, facts[i], R)
                elseif lapack_lu_ws !== nothing
                    materialize_generalized_shift!(ZmA, A, B, contour.nodes[i])
                    dense_lapack_linsolve!(temp, ZmA, R, lapack_lu_ws)
                else
                    materialize_generalized_shift!(ZmA, A, B, contour.nodes[i])
                    linsolve!(temp, ZmA, R, factorizer, left_divider)
                end
                temp .= X .- temp
                scale_columns!(temp, resolvent, contour.weights[i])
                Q .+= temp
            end
        end
    end
    if store
        foreach(finalize!, facts)
    end
    in_contour!(inside, Λ, contour)
    _, contour_nonempty = maximum_masked(res, inside)
    if !contour_nonempty println("no eigenvalues found in contour!") end
    Λ[inside], X[:, inside], res[inside]
end

function dual_gen_feast!(Xr::AbstractMatrix, Xl::AbstractMatrix, A::AbstractMatrix, B;
                    nodes::Integer=8, iter::Integer=10, c=complex(0.0,0.0), r=1.0,
                    debug=false, store=false, ϵ=1e-12, factorizer=lu, left_divider=ldiv!)
    contour = circular_contour_trapezoidal(c, r, nodes)
    dual_gen_feast!(Xr, Xl, A, B, contour; iter=iter, debug=debug, ϵ=ϵ, factorizer=factorizer, left_divider=left_divider)
end

function dual_gen_feast!(Xr::AbstractMatrix, Xl::AbstractMatrix, A::AbstractMatrix, B, contour::Contour;
                    iter::Integer=10, debug=false, store=false, ϵ=1e-12, factorizer=lu, left_divider=ldiv!)
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

    if store
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

         Threads.@threads for i=2:nodes
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

    for nit=0:iter
        mul!(Rr, B, Qr)
        mul!(Bq, Ql', Rr)
        U, S, Vt = dense_lapack_svd!(Bq, svd_ws)
        mul!(Rr, Qr, adjoint(Vt))
        Qr .= Rr
        inv_scale_columns!(Qr, S)
        mul!(Rl, Ql, U)
        Ql .= Rl
        inv_scale_columns!(Ql, S)
        mul!(Rr, A, Qr) ## why does this one allocate?
        mul!(Aq, Ql', Rr) ### Aq = Q' * A * Q
        mul!(Rr, B, Qr)
        mul!(Bq, Ql', Rr) ### Bq = Q' * Q = I
        dense_lapack_generalized_eigen!(Λ, Xql, Xqr, Aq, Bq, eigen_ws)
        mul!(Xr, Qr, Xqr) ### Recover eigenvectors from Ritz vectors ( X = Q * Xq )
        mul!(Xl, Ql, Xql) ### Recover left eigenvectors from paired left Ritz vectors
        update_R_shifted!(Xr, Rr, Λ, A, B, residual_shift, residual_x, residual_y) ### compute residual vectors R for RII update
        update_R_shifted!(Xl, Rl, Λ, A', B', residual_shift, residual_x, residual_y) ### compute residual vectors R for RII update
        residuals!(resr, Rr, Λ, A) ### compute actual residuals
        in_contour!(inside, Λ, contour)
        max_res_inside, contour_nonempty = maximum_masked(resr, inside)
        if debug iter_debug_print(nit, Λ, resr, contour, 1e-5) end
        if contour_nonempty && max_res_inside < ϵ
              if debug println("converged in $nit iteration") end
              break
        end
        if nit < iter ### Do not solve linear systems / form Q on last iteration
            Qr .= 0.00
            Ql .= 0.00
            for i=1:nodes
                fill_resolvent!(resolvent, contour.nodes[i], Λ)
                if store
                    left_divider(temp, rfacts[i], Rr)
                elseif lapack_lu_ws !== nothing
                    materialize_generalized_shift!(ZmA, A, B, contour.nodes[i])
                    dense_lapack_factor!(ZmA, lapack_lu_ws)
                    dense_lapack_solve_factored!(temp, ZmA, Rr, lapack_lu_ws, 'N')
                else
                    materialize_generalized_shift!(ZmA, A, B, contour.nodes[i])
                    linsolve!(temp, ZmA, Rr, factorizer, left_divider)
                end
                temp .= Xr .- temp
                scale_columns!(temp, resolvent, contour.weights[i])
                Qr .+= temp

                fill_adjoint_resolvent!(resolvent, contour.nodes[i], Λ)
                if store
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
                temp .= Xl .- temp
                scale_columns!(temp, resolvent, conj(contour.weights[i]))
                Ql .+= temp
            end
        end
    end
    if store
        foreach(finalize!, rfacts)
        if !share_adjoint_factors
            foreach(finalize!, lfacts)
        end
    end
    in_contour!(inside, Λ, contour)
    _, contour_nonempty = maximum_masked(resr, inside)
    if !contour_nonempty println("no eigenvalues found in contour!") end
    Λ[inside], Xr[:, inside], Xl[:, inside], resr[inside]
end
