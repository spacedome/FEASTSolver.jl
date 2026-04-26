struct MomentRIIIterationStats
    iteration::Int
    rank::Int
    inside_count::Int
    converged_inside_count::Int
    pair_residual::Float64
    max_inside_residual::Float64
    singular_values::Vector{Float64}
end

struct MomentRIIResult
    values::Vector{ComplexF64}
    vectors::Matrix{ComplexF64}
    residuals::Vector{Float64}
    pair_vectors::Matrix{ComplexF64}
    pair_matrix::Matrix{ComplexF64}
    history::Vector{MomentRIIIterationStats}
end

function polynomial_matrix(coeffs::AbstractVector, z)
    M = complex.(coeffs[end])
    for j in (length(coeffs)-1):-1:1
        @. M = z * M + coeffs[j]
    end
    M
end

function polynomial_pair_residual(coeffs::AbstractVector, X::AbstractMatrix, S::AbstractMatrix)
    R = zeros(ComplexF64, size(X))
    Xpower = copy(X)
    next_power = similar(X)
    work = similar(X)
    for (j, A) in pairs(coeffs)
        mul!(work, A, Xpower)
        R .+= work
        if j < length(coeffs)
            mul!(next_power, Xpower, S)
            copyto!(Xpower, next_power)
        end
    end
    R
end

function lifted_pair_matrix(X::AbstractMatrix, S::AbstractMatrix, lift::Integer)
    n, p = size(X)
    lifted = zeros(ComplexF64, lift * n, p)
    block = copy(X)
    next_block = similar(X)
    for j in 1:lift
        lifted[(j - 1) * n + 1:j * n, :] .= block
        if j < lift
            mul!(next_block, block, S)
            copyto!(block, next_block)
        end
    end
    lifted
end

function normalize_lifted_pair(X::AbstractMatrix, S::AbstractMatrix, lift::Integer)
    lifted = lifted_pair_matrix(X, S, lift)
    F = LinearAlgebra.qr(lifted)
    p = size(X, 2)
    Q = Matrix(F.Q)[:, 1:p]
    R = F.R[1:p, :]
    Q[1:size(X, 1), :], R * S / R
end

function pair_selection(values::AbstractVector, contour::Contour, keep_extra::Integer)
    inside = in_contour(values, contour)
    inside_count = count(inside)
    keep_count = min(length(values), inside_count + keep_extra)
    keep_count == 0 && return BitVector(falses(length(values)))
    scores = map(eachindex(values)) do i
        z = values[i]
        distance_score = contour isa CircularContour ? abs(abs(z - contour.c) / contour.r - 1) : zero(abs(z))
        (inside[i] ? 0 : 1, distance_score, abs(z))
    end
    selected_indices = sortperm(scores)[1:keep_count]
    selected = falses(length(values))
    selected[selected_indices] .= true
    BitVector(selected)
end

moment_coordinate(z, contour::Contour) = z
moment_coordinate(z, contour::CircularContour) = (z - contour.c) / contour.r

function lambda_matrix_from_moment_coordinate(S::AbstractMatrix, contour::Contour)
    Matrix(S)
end

function lambda_matrix_from_moment_coordinate(S::AbstractMatrix, contour::CircularContour)
    contour.c * Matrix{ComplexF64}(I, size(S, 1), size(S, 2)) + contour.r * S
end

function moment_coordinate_from_lambda_matrix(S::AbstractMatrix, contour::Contour)
    Matrix(S)
end

function moment_coordinate_from_lambda_matrix(S::AbstractMatrix, contour::CircularContour)
    (S - contour.c * Matrix{ComplexF64}(I, size(S, 1), size(S, 2))) / contour.r
end

function restrict_pair_to_contour(X::AbstractMatrix, S::AbstractMatrix, contour::Contour, lift::Integer; keep_extra::Integer=0)
    F = LinearAlgebra.schur(S)
    select = pair_selection(F.values, contour, keep_extra)
    count(select) == size(S, 1) && return normalize_lifted_pair(X, S, lift)
    any(select) || return normalize_lifted_pair(X, S, lift)
    ordered = LinearAlgebra.ordschur(F, select)
    p = count(select)
    Xrestricted = X * ordered.Z[:, 1:p]
    Srestricted = ordered.T[1:p, 1:p]
    normalize_lifted_pair(Xrestricted, Srestricted, lift)
end

function restrict_coordinate_pair_to_contour(X::AbstractMatrix, Scoord::AbstractMatrix, contour::Contour, lift::Integer; keep_extra::Integer=0)
    Sλ = lambda_matrix_from_moment_coordinate(Scoord, contour)
    Xrestricted, Sλrestricted = restrict_pair_to_contour(X, Sλ, contour, lift; keep_extra=keep_extra)
    Scoordrestricted = moment_coordinate_from_lambda_matrix(Sλrestricted, contour)
    normalize_lifted_pair(Xrestricted, Scoordrestricted, lift)
end

function hankel_pair_identity(moments::AbstractVector, moment_count::Integer; ranktol=1e-10, maxrank=typemax(Int))
    n, m = size(moments[1])
    H0 = zeros(ComplexF64, moment_count * n, moment_count * m)
    H1 = similar(H0)
    for i in 1:moment_count, j in 1:moment_count
        rows = (i - 1) * n + 1:i * n
        cols = (j - 1) * m + 1:j * m
        H0[rows, cols] .= moments[i + j - 1]
        H1[rows, cols] .= moments[i + j]
    end

    F = LinearAlgebra.svd(H0)
    isempty(F.S) && error("empty Hankel SVD")
    rank = count(F.S ./ F.S[1] .> ranktol)
    rank = min(rank, maxrank, length(F.S))
    rank > 0 || error("Hankel rank is zero; loosen ranktol or change the contour/probe")

    U = F.U[:, 1:rank]
    V = F.V[:, 1:rank]
    B = U' * H1 * V * Diagonal(1 ./ F.S[1:rank])
    X = U[1:n, :]
    X, B = normalize_lifted_pair(X, B, moment_count)
    X, B, rank, Float64.(F.S)
end

function orthonormal_probe(probe::AbstractMatrix)
    F = LinearAlgebra.qr(probe)
    cols = min(size(probe)...)
    Matrix(F.Q)[:, 1:cols]
end

function projected_hankel_pair_identity(
    moments::AbstractVector,
    left_probe::AbstractMatrix,
    moment_count::Integer;
    ranktol=1e-10,
    maxrank=typemax(Int),
)
    n, m = size(moments[1])
    size(left_probe, 1) == n || error("left_probe row count must match moment row count")
    ell = size(left_probe, 2)
    H0 = zeros(ComplexF64, moment_count * ell, moment_count * m)
    H1 = similar(H0)
    right_moments = zeros(ComplexF64, n, moment_count * m)
    for j in 1:moment_count
        right_moments[:, (j - 1) * m + 1:j * m] .= moments[j]
    end

    for i in 1:moment_count, j in 1:moment_count
        rows = (i - 1) * ell + 1:i * ell
        cols = (j - 1) * m + 1:j * m
        H0[rows, cols] .= left_probe' * moments[i + j - 1]
        H1[rows, cols] .= left_probe' * moments[i + j]
    end

    F = LinearAlgebra.svd(H0)
    isempty(F.S) && error("empty projected Hankel SVD")
    rank = count(F.S ./ F.S[1] .> ranktol)
    rank = min(rank, maxrank, length(F.S))
    rank > 0 || error("projected Hankel rank is zero; loosen ranktol or change probes/contour")

    U = F.U[:, 1:rank]
    V = F.V[:, 1:rank]
    invσ = Diagonal(1 ./ F.S[1:rank])
    B = U' * H1 * V * invσ
    X = right_moments * V * invσ
    X, B = normalize_lifted_pair(X, B, moment_count)
    X, B, rank, Float64.(F.S)
end

function extract_moment_pair(moments, moment_count; left_probe=nothing, ranktol=1e-10, maxrank=typemax(Int))
    if left_probe === nothing
        return hankel_pair_identity(moments, moment_count; ranktol=ranktol, maxrank=maxrank)
    end
    projected_hankel_pair_identity(
        moments,
        orthonormal_probe(left_probe),
        moment_count;
        ranktol=ranktol,
        maxrank=maxrank,
    )
end

function initial_polynomial_moments(coeffs::AbstractVector, Xprobe::AbstractMatrix, contour::Contour, moment_count::Integer)
    n, m = size(Xprobe)
    moments = [zeros(ComplexF64, n, m) for _ in 1:(2 * moment_count)]
    for (z, weight) in zip(contour_nodes(contour), contour_weights(contour))
        Tz = polynomial_matrix(coeffs, z)
        solved = Tz \ Xprobe
        coordinate = moment_coordinate(z, contour)
        zpower = one(ComplexF64)
        for p in eachindex(moments)
            moments[p] .+= (weight * zpower) .* solved
            zpower *= coordinate
        end
    end
    moments
end

function rii_polynomial_moments(coeffs::AbstractVector, X::AbstractMatrix, Sλ::AbstractMatrix, R::AbstractMatrix, contour::Contour, moment_count::Integer)
    n, p = size(X)
    moments = [zeros(ComplexF64, n, p) for _ in 1:(2 * moment_count)]
    I_p = Matrix{ComplexF64}(I, p, p)
    for (z, weight) in zip(contour_nodes(contour), contour_weights(contour))
        Tz = polynomial_matrix(coeffs, z)
        corrected = X - Tz \ R
        corrected = corrected / (z * I_p - Sλ)
        coordinate = moment_coordinate(z, contour)
        zpower = one(ComplexF64)
        for q in eachindex(moments)
            moments[q] .+= (weight * zpower) .* corrected
            zpower *= coordinate
        end
    end
    moments
end

function scalar_pair_diagnostics(coeffs::AbstractVector, X::AbstractMatrix, S::AbstractMatrix, contour::Contour; residual_tol=1e-8)
    F = LinearAlgebra.eigen(S)
    values = ComplexF64.(F.values)
    vectors = X * F.vectors
    residuals = zeros(Float64, length(values))
    for j in eachindex(values)
        v = vectors[:, j]
        v ./= norm(v)
        Tlambda = polynomial_matrix(coeffs, values[j])
        residuals[j] = norm(Tlambda * v) / max(norm(Tlambda), eps(Float64))
    end
    inside = in_contour(values, contour)
    converged_inside = count(inside .& (residuals .<= residual_tol))
    max_inside = any(inside) ? maximum(residuals[inside]) : Inf
    values, vectors, residuals, count(inside), converged_inside, max_inside
end

function record_moment_rii_iteration!(
    history::Vector{MomentRIIIterationStats},
    iteration::Integer,
    coeffs::AbstractVector,
    X::AbstractMatrix,
    S::AbstractMatrix,
    singular_values::Vector{Float64},
    contour::Contour;
    residual_tol=1e-8,
)
    R = polynomial_pair_residual(coeffs, X, S)
    pair_residual = norm(R) / max(norm(lifted_pair_matrix(X, S, 1)), eps(Float64))
    _, _, residuals, inside_count, converged_inside, max_inside =
        scalar_pair_diagnostics(coeffs, X, S, contour; residual_tol=residual_tol)
    push!(
        history,
        MomentRIIIterationStats(
            Int(iteration),
            size(X, 2),
            inside_count,
            converged_inside,
            pair_residual,
            max_inside,
            singular_values,
        ),
    )
    history
end

"""
    nlfeast_moment_rii!(coeffs, X, nodes, iter; kwargs...)

Experimental higher-moment nonlinear FEAST prototype for dense polynomial
eigenvalue problems. The active state is an invariant pair `(X, S)`, not a
diagonal eigenpair list. The RII update uses
`(X - T(z) \\ T(X, S)) * inv(zI - S)` and the pair is normalized through the
lifted block `[X; X*S; ...; X*S^(K-1)]`, which is essential for defective or
higher-order spectra.

This is intentionally limited to polynomial coefficients while the moment-RII
research direction is being validated. `coeffs[j]` is the coefficient of
`z^(j-1)`.
"""
function nlfeast_moment_rii!(
    coeffs::AbstractVector,
    Xprobe::AbstractMatrix{ComplexF64},
    nodes::Integer,
    iter::Integer;
    c=complex(0.0, 0.0),
    r=1.0,
    contour=nothing,
    moments::Integer=2,
    ranktol=1e-10,
    maxrank=typemax(Int),
    residual_tol=1e-8,
    keep_extra::Integer=0,
    restrict_inside::Bool=true,
    left_probe=nothing,
)
    moments >= 1 || error("moments must be positive")
    contour = contour === nothing ? circular_contour_trapezoidal(c, r, nodes) : contour
    moments_data = initial_polynomial_moments(coeffs, Xprobe, contour, moments)
    X, Scoord, _, singular_values = extract_moment_pair(moments_data, moments; left_probe=left_probe, ranktol=ranktol, maxrank=maxrank)
    restrict_inside && ((X, Scoord) = restrict_coordinate_pair_to_contour(X, Scoord, contour, moments; keep_extra=keep_extra))

    history = MomentRIIIterationStats[]
    Sλ = lambda_matrix_from_moment_coordinate(Scoord, contour)
    record_moment_rii_iteration!(history, 0, coeffs, X, Sλ, singular_values, contour; residual_tol=residual_tol)
    for iteration in 1:iter
        Sλ = lambda_matrix_from_moment_coordinate(Scoord, contour)
        R = polynomial_pair_residual(coeffs, X, Sλ)
        moments_data = rii_polynomial_moments(coeffs, X, Sλ, R, contour, moments)
        X, Scoord, _, singular_values = extract_moment_pair(moments_data, moments; left_probe=left_probe, ranktol=ranktol, maxrank=maxrank)
        restrict_inside && ((X, Scoord) = restrict_coordinate_pair_to_contour(X, Scoord, contour, moments; keep_extra=keep_extra))
        Sλ = lambda_matrix_from_moment_coordinate(Scoord, contour)
        record_moment_rii_iteration!(history, iteration, coeffs, X, Sλ, singular_values, contour; residual_tol=residual_tol)
    end

    Sλ = lambda_matrix_from_moment_coordinate(Scoord, contour)
    values, vectors, residuals, _, _, _ = scalar_pair_diagnostics(coeffs, X, Sλ, contour; residual_tol=residual_tol)
    order = sortperm(residuals)
    MomentRIIResult(values[order], vectors[:, order], residuals[order], X, Sλ, history)
end
