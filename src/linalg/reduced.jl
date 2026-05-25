function beyn_svd_step!(Q₀::AbstractMatrix, Q₁::AbstractMatrix, A::AbstractMatrix, B::AbstractMatrix, X::AbstractMatrix{ComplexF64}, Λ::Array)
    S = svd!(Q₀)
    mul!(A, S.U', Q₁)
    mul!(B, A, S.V)
    mul!(A, B, Diagonal(1 ./ S.S))
    F = eigen!(A)
    mul!(X, S.U, F.vectors)
    Λ .= F.values
end

function beyn_svd_step!(
    Q₀::AbstractMatrix,
    Q₁::AbstractMatrix,
    A::AbstractMatrix,
    B::AbstractMatrix,
    X::AbstractMatrix{ComplexF64},
    Λ::Array,
    svd_ws::SVDsddWs,
    eigen_ws::EigenWs,
    Xq::AbstractMatrix,
)
    U, S, Vt = dense_lapack_svd!(Q₀, svd_ws, 'S')
    mul!(A, adjoint(U), Q₁)
    mul!(B, A, adjoint(Vt))
    copyto!(A, B)
    inv_scale_columns!(A, S)
    dense_lapack_eigen!(Λ, Xq, A, eigen_ws)
    mul!(X, U, Xq)
    Λ
end

function add_weighted_columns!(Y::AbstractMatrix, X::AbstractMatrix, α)
    @inbounds for j in axes(Y, 2), i in axes(Y, 1)
        Y[i, j] += α * X[i, j]
    end
    Y
end

function beyn_qr_step!(Q₀::AbstractMatrix, Q₁::AbstractMatrix, X::AbstractMatrix, Λ::Array)
    qt, rt = qr!(Q₀)
    qt = Matrix(qt)
    F = eigen!(qt' * Q₁ * inv(rt))
    mul!(X, qt, F.vectors)
    Λ .= F.values
end

function beyn_rr_step!(Q₀::AbstractMatrix, Q₁::AbstractMatrix, A::AbstractMatrix, B::AbstractMatrix, X::AbstractMatrix{ComplexF64}, Λ::Array)
    mul!(A, X', Q₁)
    mul!(B, X', Q₀)
    F = eigen!(A, B)
    mul!(X, Q₀, F.vectors)
    Λ .= F.values
end

function beyn_rr_step2!(Q₀::AbstractMatrix, Q₁::AbstractMatrix, A::AbstractMatrix, B::AbstractMatrix, X::AbstractMatrix{ComplexF64}, Λ::Array)
    mul!(A, Q₀', Q₁)
    mul!(B, Q₀', Q₀)
    F = eigen!(A, B)
    mul!(X, Q₀, F.vectors)
    Λ .= F.values
end
