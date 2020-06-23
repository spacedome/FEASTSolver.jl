function companion(A)
    N = size(A[1], 1)
    L = size(A, 1)-1
    C1 = zeros(ComplexF64, N*L, N*L)
    C2 = zeros(ComplexF64, N*L, N*L)
    res = zeros(N*L)
    C1[1:N, 1:N] .= A[1]
    for i=N+1:N*L
        C1[i,i] = 1
        C2[i,i-N] = 1
    end
    for i=1:L
        C2[1:N, N*(i-1)+1:N*i] .= -A[i+1]
    end
    F = eigen!(C1, C2)
    X = F.vectors[(L-1)*N+1:L*N, :]
    Λ = F.values

    for i=1:N*L
        X[:,i] ./= norm(X[:,i])
        R = zeros(ComplexF64, N)
        for j=1:L+1
            R += A[j] * X[:,i] .* Λ[i]^(j-1)
        end
        res[i] = norm(R) / norm(sum(A[j] .* Λ[i]^(j-1) for j=1:L+1))
    end
    return Λ, X, res
end
