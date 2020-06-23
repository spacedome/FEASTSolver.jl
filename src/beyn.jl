using LinearAlgebra: svd
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

function block_SS!(T, X::AbstractMatrix{ComplexF64}, nodes=2^4, moments=2;
    c=complex(0.0,0.0), r=1.0, debug=false)

    N, m₀ = size(X)
    K = moments*m₀

    Y = rand(N, m₀)

    Q₀, Q₁ = zeros(ComplexF64, K, K), zeros(ComplexF64, K, K)
    S = zeros(ComplexF64, N, (2*moments+1)*m₀)

    θ = LinRange(π/nodes, 2*π-π/nodes, nodes)
    for i=1:nodes
        z = (r*exp(θ[i]*im)+c)
        S[:, 1:m₀] .= (T(z) \ X) .* (exp(θ[i]*im)/nodes)
        for j=2:2*moments+1
            S[:, (j-1)*m₀+1:j*m₀] .+= (z-c)^(j-1) .* S[:, 1:m₀]
        end
    end

    for i=1:moments, j=1:moments
        Q₀[(i-1)*m₀+1:i*m₀, (j-1)*m₀+1:j*m₀] .= X' * S[:, (i+j-1)*m₀+1:(i+j)*m₀]
        Q₁[(i-1)*m₀+1:i*m₀, (j-1)*m₀+1:j*m₀] .= X' * S[:, (i+j)*m₀+1:(i+j+1)*m₀]
    end

    V = svd(Q₀)
    n = min(count(V.S/V.S[1] .> 1e-13), K)

    H1 = V.U[:,1:n]' * Q₁ * V.V[:,1:n]
    H0 = V.U[:,1:n]' * Q₀ * V.V[:,1:n]

    Λ, Xq = eigen!(H1, H0)
    X = S[:,1:K] * V.V[:,1:n] *  Xq
    Λ = Λ .+ c

    # H1 = V.U[:,1:n]' * Q₁ * V.V[:,1:n] * Diagonal(1 ./ V.S[1:n])
    # Λ, Xq = eigen!(H1)
    # X = S[:,1:K] * V.V[:,1:n] * Diagonal(1 ./ V.S[1:n]) * Xq
    # Λ = Λ .+ c

    res = zeros(n)

    for i=1:size(X, 2)
        X[:,i] ./= norm(X[:,i])
        res[i] = norm(T(Λ[i])*X[:,i])/norm(T(Λ[i]))
    end
    # p = sortperm(res)
    p = 1:n
    Λ[p], X[:, p], res[p]
end
