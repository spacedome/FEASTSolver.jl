const DenseLapackScalar = Union{Float32, Float64, ComplexF32, ComplexF64}

"""
    dense_lapack_lu_workspace(A)

Create the reusable FastLapackInterface LU workspace for dense strided LAPACK
arrays. Returns `nothing` for arrays that should fall back to generic Julia
factorization paths.
"""
function dense_lapack_lu_workspace(A::AbstractMatrix{T}) where {T}
    T <: DenseLapackScalar || return nothing
    A isa StridedMatrix || return nothing
    LUWs(A)
end

"""
    use_dense_lapack_lu(A, X, store, factorizer, left_divider, T)

True when the FEAST solve path can use in-place LAPACK factor/solve calls
without changing user-visible semantics. Stored-factor mode intentionally uses
the user-provided factorizer because the factors must persist across contour
iterations.
"""
function use_dense_lapack_lu(A, X, store, factorizer, left_divider, ::Type{T}) where {T}
    !store || return false
    factorizer === lu || return false
    left_divider === ldiv! || return false
    A isa StridedMatrix || return false
    X isa StridedMatrix || return false
    T <: DenseLapackScalar
end

"""
    materialize_standard_shift!(C, A, z)

Overwrite `C` with `A - zI`. FEAST needs a fresh shifted matrix at each contour
node; this helper makes that mutation explicit and keeps the input matrix `A`
unchanged.
"""
function materialize_standard_shift!(C::AbstractMatrix, A::AbstractMatrix, z)
    copyto!(C, A)
    n = min(size(C)...)
    @inbounds for i in 1:n
        C[i, i] -= z
    end
    C
end

"""
    materialize_generalized_shift!(C, A, B, z)

Overwrite `C` with the generalized shifted pencil `A - zB`.
"""
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

"""
    materialize_adjoint_generalized_shift!(C, A, B, z)

Overwrite `C` with `(A - zB)'`, used for the left subspace in dual FEAST when
adjoint solves cannot reuse the right-side LU factors.
"""
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

"""Factor `C` in place with a preallocated LAPACK LU workspace."""
function dense_lapack_factor!(C::AbstractMatrix, ws::LUWs)
    LAPACK.getrf!(ws, C; resize=false)
    C
end

"""
    dense_lapack_solve_factored!(Y, C, X, ws, trans='N')

Solve a system using an already-factorized matrix `C`, writing the result to
`Y`. `X` is copied first because LAPACK overwrites the right-hand side.
"""
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

"""Factor `C` and solve `C \\ X` into `Y` using reusable LAPACK workspaces."""
function dense_lapack_linsolve!(Y::AbstractVecOrMat, C::AbstractMatrix, X::AbstractVecOrMat, ws::LUWs)
    dense_lapack_factor!(C, ws)
    dense_lapack_solve_factored!(Y, C, X, ws)
end

"""Overwrite `Q` with its reduced QR basis using reusable LAPACK workspaces."""
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

dense_lapack_svd!(A::AbstractMatrix, ws::SVDsddWs, job::Char='A') = LAPACK.gesdd!(ws, job, A; resize=false)

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
