
function ifeast!(A::AbstractMatrix, X₀::AbstractMatrix, nodes::Integer, iter::Integer;
                c=complex(0.0,0.0), r=1.0, debug=false, ϵ=0.05)
    N, m₀ = size(X₀)
    if size(A, 1) != size(A, 2)
        error("Incorrect dimensions of A, must be square")
    elseif size(A,1) != N
        error("Incorrect dimensions of X₀, must match A")
    end
    X = deepcopy(X₀)
    Λ = zeros(m₀)
    res = zeros(m₀)
    resvecs = zeros(ComplexF64, N, m₀)
    temp = complex.(X)
    Aq, Bq, Xq = zeros(ComplexF64, m₀, m₀), zeros(ComplexF64, m₀, m₀), zeros(ComplexF64, m₀, m₀)
    θ = LinRange(π/nodes, 2*π-π/nodes, nodes)
    Q = similar(X, ComplexF64)
    ZmA = similar(A, ComplexF64)
    for nit=1:iter
        Q .= 0.0
        for i=1:nodes
            ZmA .= 0.0
                for j=1:N
                    ZmA[j,j] = r*exp(θ[i]*im)+c
                end
            ZmA .-= A
            for m=1:m₀
                temp[:, m] .= bicgstabl(ZmA, X[:, m])
            end
            Q .+= temp .* exp(θ[i]*im)./nodes
        end
        # Q .= Matrix(qr(Q).Q)
        mul!(resvecs, A, Q) ## why does this one allocate?
        mul!(Aq, Q', resvecs) ### Aq = Q' * A * Q
        mul!(Bq, Q', Q) ### Bq = Q' * Q
        F = eigen!(Aq, Bq)
        Λ = F.values
        Xq = F.vectors
        # Λ, Xq = eigen!(Aq, Bq) ### eigen! allocates
        # for i=1:m₀
        #     Xq[:,i] = Xq[:,i] / vecnorm(Xq)
        # end
        mul!(X, Q, Xq) ### X = Q*Xq
        for i=1:m₀
            vn = norm(@view X[:,i]) ## allocates
            for j=1:N
                X[j,i] /= vn
            end
            # res[i] = vecnorm(A*X[:,i]-X[:,i]*Λ[i])/vecnorm(X[:,i]*Λ[i])
        end
        mul!(resvecs, A, X) ## allocates
        mul!(temp, X, Diagonal(Λ)) ## Diagonal allocates
        resvecs .-= temp ### resvecs = A*X-X*Diagonal(Λ)
        for i in 1:m₀
            res[i]=norm(@view resvecs[:,i])
        end
        if debug println(nit) end
    end
    Λ, X, res
end
