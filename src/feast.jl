import LinearAlgebra: qr, lu

function feast!(X::AbstractMatrix, A::AbstractMatrix;
                nodes::Integer=8, iter::Integer=10, c=complex(0.0,0.0), r=1.0, ϵ=1e-12,
                debug=false, store=false, mixed_prec=false, linsolve=lu_solve!)
    contour = circular_contour_trapezoidal(c, r, nodes)
    feast!(X, A, contour; iter=iter, debug=debug, ϵ=ϵ, store=store, mixed_prec=mixed_prec, linsolve=linsolve)
end


function feast!(X::AbstractMatrix, A::AbstractMatrix, contour::Contour;
                iter::Integer=10, ϵ=1e-12, debug=false, store=false, mixed_prec=false, linsolve=lu_solve!)
    N, m₀ = size(X)
    if size(A, 1) != size(A, 2)
        error("Incorrect dimensions of A, must be square")
    elseif size(A,1) != N
        error("Incorrect dimensions of X, must match A")
    end

    Ctype = if mixed_prec ComplexF32 else ComplexF64 end

    Λ, resolvent, res = zeros(ComplexF64, m₀), zeros(ComplexF64, m₀), zeros(m₀)
    temp, R, Q = zeros(Ctype, N, m₀), similar(X, ComplexF64), copy(X)
    Aq, Xq = zeros(ComplexF64, m₀, m₀), zeros(ComplexF64, m₀, m₀)

    ZmA = similar(A, Ctype)
    nodes = size(contour.nodes, 1)

    facts = Array{Factorization}(undef, nodes)
    if store
        Threads.@threads for i=1:nodes
            ZmA .= (A - I*contour.nodes[i])
            facts[i] = lu(A - I*contour.nodes[i])
        end
    end

    for nit=0:iter
        Q .= Matrix(qr(Q).Q)
        mul!(R, A, Q) ## why does this one allocate?
        mul!(Aq, Q', R) ### Aq = Q' * A * Q
        # mul!(Bq, Q', Q) ### Bq = Q' * Q = I
        F = eigen!(Aq)
        Λ .= F.values
        Xq .= F.vectors
        mul!(X, Q, Xq) ### Recover eigenvectors from Ritz vectors ( X = Q * Xq )
        update_R!(X, R, Λ, A) ### compute residual vectors R for RII update
        residuals!(res, R, Λ, A) ### compute actual residuals
        contour_nonempty = reduce(|, in_contour(Λ, contour))
        if debug iter_debug_print(nit, Λ, res, contour, 1e-5) end
        if contour_nonempty && maximum(res[in_contour(Λ, contour)]) < ϵ
            if debug println("converged in $nit iteration") end
            break
        end
        if nit < iter ### Do not solve linear systems / form Q on last iteration
            Q .= 0.00
            for i=1:nodes
                resolvent .= 1.0 ./(contour.nodes[i] .- Λ)
                if store
                    ldiv!(temp, facts[i], R)
                else
                    ZmA .= A - I*contour.nodes[i]
                    linsolve(temp, ZmA, R)
                end

                R .= X - temp # using R as work array
                rmul!(R, Diagonal(resolvent .* contour.weights[i]))
                Q .+= R
            end
        end
    end
    contour_nonempty = reduce(|, in_contour(Λ, contour))
    if !contour_nonempty println("no eigenvalues found in contour!") end
    Λ[in_contour(Λ, contour)], X[:,in_contour(Λ, contour)], res[in_contour(Λ, contour)]
end

function gen_feast!(X::AbstractMatrix, A::AbstractMatrix, B::AbstractMatrix;
                    nodes::Integer=8, iter::Integer=10, c=complex(0.0,0.0), r=1.0,
                    debug=false, ϵ=1e-12, linsolve=lu_solve!)
    contour = circular_contour_trapezoidal(c, r, nodes)
    gen_feast!(X, A, B, contour; iter=iter, debug=debug, ϵ=ϵ, linsolve=linsolve)
end

function gen_feast!(X::AbstractMatrix, A::AbstractMatrix, B::AbstractMatrix, contour::Contour;
                    iter::Integer=10, debug=false, ϵ=1e-12, linsolve=lu_solve!)
    N, m₀ = size(X)
    if size(A, 1) != size(A, 2)
        error("Incorrect dimensions of A, must be square")
    elseif size(A,1) != N
        error("Incorrect dimensions of X, must match A")
    end

    Λ, resolvent, res = zeros(ComplexF64, m₀), zeros(ComplexF64, m₀), zeros(m₀)
    temp, R, Q = zeros(ComplexF64, N, m₀), similar(X, ComplexF64), copy(X)
    Aq, Bq, Xq = zeros(ComplexF64, m₀, m₀), zeros(ComplexF64, m₀, m₀), zeros(ComplexF64, m₀, m₀)
    ZmA = similar(A, ComplexF64)
    nodes = size(contour.nodes, 1)

    for nit=0:iter
        Q .= Matrix(qr(Q).Q)
        mul!(R, A, Q) ## why does this one allocate?
        mul!(Aq, Q', R) ### Aq = Q' * A * Q
        mul!(R, B, Q)
        mul!(Bq, Q', R) ### Bq = Q' * Q = I
        F = eigen!(Aq, Bq)
        Λ .= F.values
        Xq .= F.vectors
        mul!(X, Q, Xq) ### Recover eigenvectors from Ritz vectors ( X = Q * Xq )
        update_R!(X, R, Λ, A, B) ### compute residual vectors R for RII update
        residuals!(res, R, Λ, A) ### compute actual residuals
        contour_nonempty = reduce(|, in_contour(Λ, contour))
        if debug iter_debug_print(nit, Λ, res, contour, 1e-5) end
        if contour_nonempty && maximum(res[in_contour(Λ, contour)]) < ϵ
            if debug println("converged in $nit iteration") end
            break
        end
        if nit < iter ### Do not solve linear systems / form Q on last iteration
            Q .= 0.00
            for i=1:nodes
                resolvent .= (1.0 ./(contour.nodes[i] .- Λ))
                ZmA .= (A - B*contour.nodes[i])
                linsolve(temp, ZmA, R)
                temp .= X - temp
                rmul!(temp, Diagonal(resolvent .* contour.weights[i]))
                Q .+= temp
            end
        end
    end
    contour_nonempty = reduce(|, in_contour(Λ, contour))
    if !contour_nonempty println("no eigenvalues found in contour!") end
    Λ[in_contour(Λ, contour)], X[:,in_contour(Λ, contour)], res[in_contour(Λ, contour)]
end

function lu_solve!(Y, C, X)
    ldiv!(Y, lu(C), X)
end
