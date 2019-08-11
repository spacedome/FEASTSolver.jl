
using LinearAlgebra: ldiv!, lu!, mul!, rmul!, lmul!, eigen!, eigen, svd!, norm, Diagonal
using IterativeSolvers: bicgstabl
using SparseArrays: similar

export feast!
export ifeast!
export gen_feast!, dual_gen_feast!

function in_contour(λ, c, r)
    abs(λ - c) <= r
end


### Overwrites X₀
function feast!(X₀::AbstractMatrix, A::AbstractMatrix;
                nodes::Integer=8, iter::Integer=10, c=complex(0.0,0.0), r=1.0, debug=false, ϵ=1e-12)
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
    temp = zeros(ComplexF64, N, m₀)
    Aq, Bq, Xq = zeros(ComplexF64, m₀, m₀), zeros(ComplexF64, m₀, m₀), zeros(ComplexF64, m₀, m₀)
    θ = LinRange(π/nodes, 2*π-π/nodes, nodes)
    # θ = LinRange(0, 2*π-2*π/nodes, nodes)
    Q = similar(X, ComplexF64)
    ZmA = similar(A, ComplexF64)
    for nit=1:iter
        Q .= 0.00
        for i=1:nodes
            ZmA .= 0.0
                for j=1:N
                    ZmA[j,j] = r*exp(θ[i]*im)+c
                end
            ZmA .-= A
            ldiv!(temp, lu!(ZmA), X)
            Q .+= temp .* exp(θ[i]*im)./nodes
        end
        ### Q .= Matrix(qr(Q).Q)
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
        contour_nonempty = reduce(|, in_contour.(Λ, c, r))
        if contour_nonempty && maximum(res[in_contour.(Λ, c, r)]) < ϵ
            println("converged in $nit iteration")
            break
        end
        if debug println(nit) end
    end
    contour_nonempty = reduce(|, in_contour.(Λ, c, r))
    if !contour_nonempty println("no eigenvalues found in contour!") end
    Λ[in_contour.(Λ, c, r)], X[:,in_contour.(Λ, c, r)], res[in_contour.(Λ, c, r)]
end

function gen_feast!(X::AbstractMatrix, A::AbstractMatrix, B::AbstractMatrix;
                nodes::Integer=8, iter::Integer=10, c=complex(0.0,0.0), r=1.0, debug=false, ϵ=1e-12)
    N, m₀ = size(X)
    if size(A, 1) != size(A, 2)
        error("Incorrect dimensions of A, must be square")
    elseif size(A,1) != N
        error("Incorrect dimensions of X₀, must match A")
    end
    if !(eltype(X) <: Complex)
        # currently not handling this actually so just use complex X for all problems
        error("X must be complex for non-hermitian problems")
    end
    Λ = zeros(ComplexF64, m₀)
    res = zeros(m₀)
    resvecs = zeros(ComplexF64, N, m₀)
    temp = zeros(ComplexF64, N, m₀)
    Aq, Bq, Xq = zeros(ComplexF64, m₀, m₀), zeros(ComplexF64, m₀, m₀), zeros(ComplexF64, m₀, m₀)
    θ = LinRange(π/nodes, 2*π-π/nodes, nodes)
    ZmA = similar(A, ComplexF64)

    Y = similar(X)
    Q = deepcopy(X)
    for nit=1:iter
        ########## Rayleigh-Ritz ##########
        mul!(resvecs, A, Q) ## why does this one allocate?
        mul!(Aq, Q', resvecs) ### Aq = Q' * A * Q
        mul!(resvecs, B, Q) ## why does this one allocate?
        mul!(Bq, Q', resvecs) ### Aq = Q' * A * Q
        ## Will have to manually use BLAS to get this without reallocation
        F = eigen!(Aq, Bq)
        Λ .= F.values
        Xq .= F.vectors
        mul!(X, Q, Xq) ### X = Q*Xq
        ########## Calculate Residuals ##########
        mul!(temp, A, X) ## allocates
        mul!(resvecs, B, X)
        rmul!(resvecs, Diagonal(Λ)) ## Diagonal allocates
        resvecs .-= temp ### resvecs = A*X-X*Diagonal(Λ)
        for i in 1:m₀
            res[i]=norm(@view resvecs[:,i])
        end
        contour_nonempty = reduce(|, in_contour.(Λ, c, r))
        if contour_nonempty && maximum(res[in_contour.(Λ, c, r)]) < ϵ
            println("converged in $nit iteration")
            break
        end
        ########## Solve Linear Systems & Form Q ##########
        Q .= 0.0
        for i=1:nodes
            ZmA .= 0.0
            for j=1:N
                ZmA[j,j] = r*exp(θ[i]*im)+c
            end
            rmul!(ZmA, B)
            ZmA .-= A
            ZmA .*= 1.0/(exp(θ[i]*im)/nodes) # quadrature weight
            ldiv!(Y, lu!(ZmA), resvecs)
            Q .+= (exp(θ[i]*im) / nodes) .* (X .- Y) * Diagonal(inv.((r*exp(θ[i]*im)+c) .- Λ ))
        end
        if debug println(nit) end
    end
    contour_nonempty = reduce(|, in_contour.(Λ, c, r))
    if !contour_nonempty println("no eigenvalues found in contour!") end
    Λ[in_contour.(Λ, c, r)], X[:,in_contour.(Λ, c, r)], res[in_contour.(Λ, c, r)]
end

function dual_gen_feast!(X::AbstractMatrix, A::AbstractMatrix, B::AbstractMatrix;
                nodes::Integer=8, iter::Integer=10, c=complex(0.0,0.0), r=1.0, debug=false, ϵ=1e-12)
    N, m₀ = size(X)
    if size(A, 1) != size(A, 2)
        error("Incorrect dimensions of A, must be square")
    elseif size(A,1) != N
        error("Incorrect dimensions of X₀, must match A")
    end
    if !(eltype(X) <: Complex)
        # currently not handling this actually so just use complex X for all problems
        error("X must be complex for non-hermitian problems")
    end
    Λ, ΛH = zeros(ComplexF64, m₀), zeros(ComplexF64, m₀)
    res = zeros(m₀)
    lres = zeros(m₀)
    resvecs = zeros(ComplexF64, N, m₀)
    lresvecs = zeros(ComplexF64, N, m₀)
    temp = zeros(ComplexF64, N, m₀)
    Aq, Bq, Xqr, Xql = zeros(ComplexF64, m₀, m₀), zeros(ComplexF64, m₀, m₀), zeros(ComplexF64, m₀, m₀), zeros(ComplexF64, m₀, m₀)
    θ = LinRange(π/nodes, 2*π-π/nodes, nodes)
    ZmA = similar(A, ComplexF64)

    XL = rand(ComplexF64, size(X))
    QR, QL = deepcopy(X), deepcopy(XL)
    YR, YL = similar(X), similar(XL)
    for nit=1:iter
        ########## B-Biorthogonalize ##########
        mul!(temp, B, QR) ## why does this one allocate?
        mul!(Bq, QL', temp) ### Aq = Q' * A * Q
        F = svd!(Bq)
        QR .= QR * F.V * Diagonal(inv.(sqrt.(F.S)))
        QL .= QL * F.U * Diagonal(inv.(sqrt.(F.S)))
        ########## Rayleigh-Ritz ##########
        mul!(temp, A, QR) ## why does this one allocate?
        mul!(Aq, QL', temp) ### Aq = Q' * A * Q
        mul!(temp, B, QR) ## why does this one allocate?
        mul!(Bq, QL', temp) ### Aq = Q' * A * Q
        ## Will have to manually use BLAS to get this without reallocation
        F = eigen(Aq, Bq)
        Λ .= F.values
        Xqr .= F.vectors
        mul!(X, QR, Xqr) ### X = Q*Xq
        FH = eigen(copy(Aq'), copy(Bq'))
        ΛH .= FH.values
        Xql .= FH.vectors
        mul!(XL, QL, Xql) ### X = Q*Xq
        ########## Calculate Residuals ##########
        mul!(temp, A, X) ## allocates
        mul!(resvecs, B, X)
        rmul!(resvecs, Diagonal(Λ)) ## Diagonal allocates
        resvecs .-= temp ### resvecs = A*X-X*Diagonal(Λ)
        for i in 1:m₀
            res[i]=norm(@view resvecs[:,i])
        end
        mul!(temp, A', XL) ## allocates
        mul!(lresvecs, B', XL)
        rmul!(lresvecs, Diagonal(ΛH)) ## Diagonal allocates
        lresvecs .-= temp ### resvecs = A*X-X*Diagonal(Λ)
        for i in 1:m₀
            lres[i]=norm(@view lresvecs[:,i])
        end
        contour_nonempty = reduce(|, in_contour.(Λ, c, r))
        if contour_nonempty && maximum(res[in_contour.(Λ, c, r)]) < ϵ
            println("converged in $nit iteration")
            break
        end
        ########## Solve Linear Systems & Form Q ##########
        QR .= 0.0
        QL .= 0.0
        for i=1:nodes
            ZmA .= 0.0
            for j=1:N
                ZmA[j,j] = r*exp(θ[i]*im)+c
            end
            rmul!(ZmA, B)
            ZmA .-= A
            YR .= (ZmA ./(exp(θ[i]*im)/nodes)) \ resvecs # ldiv!(YR, lu!(ZmA), resvecs)
            YL .= (ZmA' ./(exp(θ[i]*im)/nodes)) \ lresvecs # ldiv!(YL, lu!(ZmA'), lresvecs)
            DR = Diagonal(inv.((r*exp(θ[i]*im)+c) .- Λ ))
            DL = Diagonal(inv.((r*exp(θ[i]*im)+c)' .- ΛH ))
            QR .+= (exp(θ[i]*im) / nodes) .* ((X .- YR) * DR)
            QL .+= (exp(θ[i]*im) / nodes)' .* ((XL .- YL) * DL)
        end
        if debug println(nit) end
    end
    contour_nonempty = reduce(|, in_contour.(Λ, c, r))
    if !contour_nonempty println("no eigenvalues found in contour!") end
    Λ[in_contour.(Λ, c, r)], X[:,in_contour.(Λ, c, r)], res[in_contour.(Λ, c, r)]
end


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
