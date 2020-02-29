
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
            # res[i] = norm(T(Λ[i])*X[:,i])/norm(T(c)*X[:,i])
        res[i] = norm(T(Λ[i])*X[:,i])
    end
    p = sortperm(res)
    Λ[p], X[:, p], res[p]
end
