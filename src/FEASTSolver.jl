__precompile__(true)

module FEASTSolver

using LinearAlgebra: ldiv!, lu!, mul!, eigen!, svd!, norm, Diagonal

import LinearAlgebra
import FastGaussQuadrature

export feast!
export beyn

function beyn(T::Function, A::AbstractMatrix, X::AbstractMatrix, nodes::Integer;
                c=complex(0.0,0.0), r=1.0)
    N, m₀ = size(X)
    if size(A, 1) != size(A, 2)
        error("Incorrect dimensions of A, must be square")
    elseif size(A,1) != N
        error("Incorrect dimensions of X₀, must match A")
    end
    res = zeros(m₀)
    temp = zeros(ComplexF64, N, m₀)

    Q₀, Q₁ = zeros(ComplexF64, N, m₀), zeros(ComplexF64, N, m₀)
    ZmA = similar(A, ComplexF64)
    θ = LinRange(π/nodes, 2*π-π/nodes, nodes)
    for i=1:nodes
        ZmA .= T(r*exp(θ[i]*im)+c)
        ldiv!(temp, lu!(ZmA), X)
        Q₀ .+= temp .* exp(θ[i]*im)./nodes
        Q₁ .+= (r*exp(θ[i]*im)+c) .* temp .* exp(θ[i]*im)./nodes
    end
    S = svd!(Q₀)
    Aq = S.U' * Q₁ * S.V * inv(Diagonal(S.S))
    Λ, Xq = eigen!(Aq)
    X = S.U * Xq
    for i=1:m₀
#             res[i] = norm(T(Λ[i])*X[:,i])/norm(T(c)*X[:,i])
        res[i] = norm(T(Λ[i])*X[:,i])
    end
    p = sortperm(res)
    Λ[p], X[:, p], res[p]
end

### Overwrites X₀
function feast!(A::AbstractMatrix, X₀::AbstractMatrix, nodes::Integer, iter::Integer;
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
    temp = zeros(ComplexF64, N, m₀)
    Aq, Bq, Xq = zeros(ComplexF64, m₀, m₀), zeros(ComplexF64, m₀, m₀), zeros(ComplexF64, m₀, m₀)
    θ = LinRange(π/nodes, 2*π-π/nodes, nodes)
#     θ = LinRange(0, 2*π-2*π/nodes, nodes)
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
            ldiv!(temp, lu!(ZmA), X)
            Q .+= temp .* exp(θ[i]*im)./nodes
        end
#         # Q .= Matrix(qr(Q).Q)
        mul!(resvecs, A, Q) ## why does this one allocate?
        mul!(Aq, Q', resvecs) ### Aq = Q' * A * Q
        mul!(Bq, Q', Q) ### Bq = Q' * Q
        F = eigen!(Aq, Bq)
        Λ = F.values
        Xq = F.vectors
#         Λ, Xq = eigen!(Aq, Bq) ### eigen! allocates
#         for i=1:m₀
#             Xq[:,i] = Xq[:,i] / vecnorm(Xq)
#         end
        mul!(X, Q, Xq) ### X = Q*Xq
        for i=1:m₀
            vn = norm(@view X[:,i]) ## allocates
            for j=1:N
                X[j,i] /= vn
            end
#             res[i] = vecnorm(A*X[:,i]-X[:,i]*Λ[i])/vecnorm(X[:,i]*Λ[i])
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

end # module
