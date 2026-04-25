const DenseLapackScalar = Union{Float32, Float64, ComplexF32, ComplexF64}

function dense_lapack_lu_workspace(A::AbstractMatrix{T}) where {T}
    T <: DenseLapackScalar || return nothing
    A isa StridedMatrix || return nothing
    LUWs(A)
end

function use_dense_lapack_lu(A, X, store, factorizer, left_divider, ::Type{T}) where {T}
    !store || return false
    factorizer === lu || return false
    left_divider === ldiv! || return false
    A isa StridedMatrix || return false
    X isa StridedMatrix || return false
    T <: DenseLapackScalar
end

function materialize_standard_shift!(C::AbstractMatrix, A::AbstractMatrix, z)
    copyto!(C, A)
    n = min(size(C)...)
    @inbounds for i in 1:n
        C[i, i] -= z
    end
    C
end

function materialize_generalized_shift!(C::AbstractMatrix, A::AbstractMatrix, B::AbstractMatrix, z)
    @inbounds for j in axes(C, 2), i in axes(C, 1)
        C[i, j] = A[i, j] - z * B[i, j]
    end
    C
end

function materialize_generalized_shift!(C::AbstractMatrix, A::AbstractMatrix, B::UniformScaling, z)
    copyto!(C, A)
    n = min(size(C)...)
    @inbounds for i in 1:n
        C[i, i] -= z * B.λ
    end
    C
end

function materialize_adjoint_generalized_shift!(C::AbstractMatrix, A::AbstractMatrix, B::AbstractMatrix, z)
    @inbounds for j in axes(C, 2), i in axes(C, 1)
        C[i, j] = conj(A[j, i]) - conj(z) * conj(B[j, i])
    end
    C
end

function materialize_adjoint_generalized_shift!(C::AbstractMatrix, A::AbstractMatrix, B::UniformScaling, z)
    copyto!(C, adjoint(A))
    n = min(size(C)...)
    @inbounds for i in 1:n
        C[i, i] -= conj(z * B.λ)
    end
    C
end

function dense_lapack_factor!(C::AbstractMatrix, ws::LUWs)
    LAPACK.getrf!(ws, C; resize=false)
    C
end

function dense_lapack_solve_factored!(
    Y::AbstractVecOrMat,
    C::AbstractMatrix,
    X::AbstractVecOrMat,
    ws::LUWs,
    trans::Char='N',
)
    copyto!(Y, X)
    LAPACK.getrs!(ws, trans, C, Y)
    Y
end

function dense_lapack_linsolve!(Y::AbstractVecOrMat, C::AbstractMatrix, X::AbstractVecOrMat, ws::LUWs)
    dense_lapack_factor!(C, ws)
    dense_lapack_solve_factored!(Y, C, X, ws)
end

function dense_lapack_qr!(Q::AbstractMatrix, ws::QRWs)
    LAPACK.geqrf!(ws, Q; resize=false)
    LAPACK.orgqr!(ws, Q)
    Q
end

function dense_lapack_eigen!(Λ::AbstractVector, X::AbstractMatrix, A::AbstractMatrix, ws::EigenWs)
    ret = LAPACK.geevx!(ws, 'N', 'N', 'V', 'N', A; resize=false)
    copyto!(Λ, ret[2])
    copyto!(X, ret[4])
    Λ, X
end

function dense_lapack_generalized_eigen!(
    Λ::AbstractVector,
    X::AbstractMatrix,
    A::AbstractMatrix,
    B::AbstractMatrix,
    ws::GeneralizedEigenWs,
)
    α, β, _, vr = LAPACK.ggev!(ws, 'V', 'V', A, B; resize=false)
    Λ .= α ./ β
    copyto!(X, vr)
    Λ, X
end

function dense_lapack_generalized_eigen!(
    Λ::AbstractVector,
    Xl::AbstractMatrix,
    Xr::AbstractMatrix,
    A::AbstractMatrix,
    B::AbstractMatrix,
    ws::GeneralizedEigenWs,
)
    α, β, vl, vr = LAPACK.ggev!(ws, 'V', 'V', A, B; resize=false)
    Λ .= α ./ β
    copyto!(Xl, vl)
    copyto!(Xr, vr)
    Λ, Xl, Xr
end

dense_lapack_svd!(A::AbstractMatrix, ws::SVDsddWs) = LAPACK.gesdd!(ws, 'A', A; resize=false)

function scale_columns!(A::AbstractMatrix, scales::AbstractVector, weight=one(eltype(A)))
    @inbounds for j in axes(A, 2)
        α = scales[j] * weight
        for i in axes(A, 1)
            A[i, j] *= α
        end
    end
    A
end

function inv_scale_columns!(A::AbstractMatrix, scales::AbstractVector)
    @inbounds for j in axes(A, 2)
        α = inv(scales[j])
        for i in axes(A, 1)
            A[i, j] *= α
        end
    end
    A
end
