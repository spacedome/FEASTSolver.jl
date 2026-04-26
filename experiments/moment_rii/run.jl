using FEASTSolver
using LinearAlgebra
using MatrixMarket
using Random
using Printf

const REPO_ROOT = normpath(joinpath(@__DIR__, "..", ".."))

struct PairHistory
    iteration::Int
    rank::Int
    inside::Int
    converged_inside::Int
    pair_residual::Float64
    max_inside_residual::Float64
    lambdas::Vector{ComplexF64}
    residuals::Vector{Float64}
    singular_values::Vector{Float64}
end

function circular_rule(center, radius, nodes)
    theta = LinRange(pi / nodes, 2pi - pi / nodes, nodes)
    z = ComplexF64[radius * exp(im * t) + center for t in theta]
    w = ComplexF64[radius * exp(im * t) / nodes for t in theta]
    z, w
end

function polynomial_matrix(coeffs, z)
    M = complex.(coeffs[end])
    for j in (length(coeffs)-1):-1:1
        @. M = z * M + coeffs[j]
    end
    M
end

function polynomial_pair_residual(coeffs, X, S)
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

function scalar_diagnostics(coeffs, X, S, center, radius; residual_tol=1e-8)
    F = eigen(S)
    lambdas = ComplexF64.(F.values)
    vectors = X * F.vectors
    residuals = zeros(Float64, length(lambdas))
    for j in eachindex(lambdas)
        v = vectors[:, j]
        v ./= norm(v)
        Tlambda = polynomial_matrix(coeffs, lambdas[j])
        residuals[j] = norm(Tlambda * v) / max(norm(Tlambda), eps(Float64))
    end
    inside = FEASTSolver.in_contour(lambdas, center, radius)
    converged_inside = count(inside .& (residuals .<= residual_tol))
    max_inside = any(inside) ? maximum(residuals[inside]) : Inf
    lambdas, residuals, count(inside), converged_inside, max_inside
end

function lifted_pair_matrix(X, S, lift)
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

function normalize_lifted_pair(X, S, lift)
    lifted = lifted_pair_matrix(X, S, lift)
    F = qr(lifted)
    p = size(X, 2)
    Q = Matrix(F.Q)[:, 1:p]
    R = F.R[1:p, :]
    Q[1:size(X, 1), :], R * S / R
end

function pair_selection(values, center, radius, keep; target_count=false)
    inside = FEASTSolver.in_contour(values, center, radius)
    keep_count = if target_count && keep > 0
        min(length(values), keep)
    else
        min(length(values), max(count(inside), keep))
    end
    keep_count == 0 && return BitVector(falses(length(values)))
    scores = map(eachindex(values)) do i
        distance = abs(values[i] - center)
        (
            inside[i] ? 0 : 1,
            inside[i] ? distance / radius : abs(distance / radius - 1),
            distance,
        )
    end
    selected_indices = sortperm(scores)[1:keep_count]
    selected = falses(length(values))
    selected[selected_indices] .= true
    BitVector(selected)
end

function restrict_pair_to_contour(X, S, center, radius, lift; keep=0, target_count=false)
    F = schur(S)
    select = pair_selection(F.values, center, radius, keep; target_count=target_count)
    count(select) == size(S, 1) && return normalize_lifted_pair(X, S, lift)
    any(select) || return normalize_lifted_pair(X, S, lift)
    ordered = ordschur(F, select)
    p = count(select)
    Xrestricted = X * ordered.Z[:, 1:p]
    Srestricted = ordered.T[1:p, 1:p]
    normalize_lifted_pair(Xrestricted, Srestricted, lift)
end

function hankel_pair_identity(moments, moment_count; ranktol=1e-10, maxrank=typemax(Int))
    n, m = size(moments[1])
    H0 = zeros(ComplexF64, moment_count * n, moment_count * m)
    H1 = similar(H0)
    for i in 1:moment_count, j in 1:moment_count
        rows = (i - 1) * n + 1:i * n
        cols = (j - 1) * m + 1:j * m
        H0[rows, cols] .= moments[i + j - 1]
        H1[rows, cols] .= moments[i + j]
    end

    F = svd(H0)
    isempty(F.S) && error("empty Hankel SVD")
    rank = count(F.S ./ F.S[1] .> ranktol)
    rank = min(rank, maxrank, length(F.S))
    rank > 0 || error("Hankel rank is zero; loosen ranktol or change the contour/probe")

    U = F.U[:, 1:rank]
    V = F.V[:, 1:rank]
    B = U' * H1 * V * Diagonal(1 ./ F.S[1:rank])
    X = U[1:n, :]
    X, B = normalize_lifted_pair(X, B, moment_count)
    X, B, rank, copy(F.S)
end

function projected_hankel_pair_identity(moments, left_probe, moment_count; ranktol=1e-10, maxrank=typemax(Int))
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

    F = svd(H0)
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
    X, B, rank, copy(F.S)
end

function shifted_pair_identity(moments, lift; ranktol=1e-10, maxrank=typemax(Int))
    length(moments) >= lift + 1 || error("shifted realization needs at least lift+1 moments")
    n, p = size(moments[1])
    L0 = zeros(ComplexF64, lift * n, p)
    L1 = similar(L0)
    for j in 1:lift
        rows = (j - 1) * n + 1:j * n
        L0[rows, :] .= moments[j]
        L1[rows, :] .= moments[j + 1]
    end

    F = svd(L0)
    isempty(F.S) && error("empty shifted moment SVD")
    rank = count(F.S ./ F.S[1] .> ranktol)
    rank = min(rank, maxrank, p, length(F.S))
    rank > 0 || error("shifted moment rank is zero; loosen ranktol or change the contour/probe")

    if rank < p
        V = F.V[:, 1:rank]
        X = moments[1] * V
        S = (L0 * V) \ (L1 * V)
    else
        X = moments[1]
        S = L0 \ L1
    end

    X, S = normalize_lifted_pair(X, S, lift)
    X, S, rank, copy(F.S)
end

function chebyshev_shifted_pair_identity(moments, lift; ranktol=1e-10, maxrank=typemax(Int))
    length(moments) >= lift + 1 || error("Chebyshev realization needs at least lift+1 moments")
    n, p = size(moments[1])
    L0 = zeros(ComplexF64, lift * n, p)
    L1 = similar(L0)
    for j in 1:lift
        rows = (j - 1) * n + 1:j * n
        L0[rows, :] .= moments[j]
        if j == 1
            L1[rows, :] .= moments[2]
        else
            L1[rows, :] .= 0.5 .* (moments[j - 1] .+ moments[j + 1])
        end
    end

    F = svd(L0)
    isempty(F.S) && error("empty Chebyshev moment SVD")
    rank = count(F.S ./ F.S[1] .> ranktol)
    rank = min(rank, maxrank, p, length(F.S))
    rank > 0 || error("Chebyshev moment rank is zero; loosen ranktol or change the contour/probe")

    if rank < p
        V = F.V[:, 1:rank]
        X = moments[1] * V
        S = (L0 * V) \ (L1 * V)
    else
        X = moments[1]
        S = L0 \ L1
    end

    X, S = normalize_lifted_pair(X, S, lift)
    X, S, rank, copy(F.S)
end

function initial_moments(coeffs, Xprobe, z_nodes, z_weights, moment_count)
    n, m = size(Xprobe)
    moments = [zeros(ComplexF64, n, m) for _ in 1:(2 * moment_count)]
    for (z, weight) in zip(z_nodes, z_weights)
        Tz = polynomial_matrix(coeffs, z)
        solved = Tz \ Xprobe
        zpower = one(ComplexF64)
        for p in eachindex(moments)
            moments[p] .+= (weight * zpower) .* solved
            zpower *= z
        end
    end
    moments
end

function rii_moments(coeffs, X, S, R, z_nodes, z_weights, moment_count)
    n, p = size(X)
    moments = [zeros(ComplexF64, n, p) for _ in 1:(2 * moment_count)]
    I_p = Matrix{ComplexF64}(I, p, p)
    for (z, weight) in zip(z_nodes, z_weights)
        Tz = polynomial_matrix(coeffs, z)
        corrected = X - Tz \ R
        corrected = corrected / (z * I_p - S)
        zpower = one(ComplexF64)
        for q in eachindex(moments)
            moments[q] .+= (weight * zpower) .* corrected
            zpower *= z
        end
    end
    moments
end

function initial_polynomial_moments_scaled(coeffs, Xprobe, z_nodes, z_weights, center, radius, moment_count)
    n, m = size(Xprobe)
    moments = [zeros(ComplexF64, n, m) for _ in 1:(2 * moment_count)]
    for (z, weight) in zip(z_nodes, z_weights)
        solved = polynomial_matrix(coeffs, z) \ Xprobe
        μ = (z - center) / radius
        μpower = one(ComplexF64)
        for p in eachindex(moments)
            moments[p] .+= (weight * μpower) .* solved
            μpower *= μ
        end
    end
    moments
end

function rii_polynomial_moments_scaled(coeffs, X, Sλ, z_nodes, z_weights, center, radius, moment_count)
    n, p = size(X)
    moments = [zeros(ComplexF64, n, p) for _ in 1:(2 * moment_count)]
    I_p = Matrix{ComplexF64}(I, p, p)
    R = polynomial_pair_residual(coeffs, X, Sλ)
    for (z, weight) in zip(z_nodes, z_weights)
        corrected = X - polynomial_matrix(coeffs, z) \ R
        corrected = corrected / (z * I_p - Sλ)
        μ = (z - center) / radius
        μpower = one(ComplexF64)
        for q in eachindex(moments)
            moments[q] .+= (weight * μpower) .* corrected
            μpower *= μ
        end
    end
    moments
end

function polynomial_pair_residual_derivative(coeffs, X, S, ΔX, ΔS)
    ΔR = zeros(ComplexF64, size(X))
    Xpowers = Vector{Matrix{ComplexF64}}(undef, length(coeffs))
    Xpowers[1] = Matrix(X)
    for j in 2:length(coeffs)
        Xpowers[j] = Xpowers[j - 1] * S
    end

    for (j, A) in pairs(coeffs)
        ΔR .+= A * ΔX * (j == 1 ? Matrix{ComplexF64}(I, size(S, 1), size(S, 2)) : S^(j - 1))
        if j > 1
            for q in 0:(j - 2)
                ΔR .+= A * Xpowers[q + 1] * ΔS * S^(j - 2 - q)
            end
        end
    end
    ΔR
end

function invariant_pair_newton_step(coeffs, X, S; lift=1, damping=1.0)
    n, p = size(X)
    R = polynomial_pair_residual(coeffs, X, S)
    unknowns = n * p + p * p
    equations = n * p + p * p
    J = zeros(ComplexF64, equations, unknowns)

    column = 1
    for col in 1:p, row in 1:n
        ΔX = zeros(ComplexF64, n, p)
        ΔS = zeros(ComplexF64, p, p)
        ΔX[row, col] = 1
        J[1:n*p, column] .= vec(polynomial_pair_residual_derivative(coeffs, X, S, ΔX, ΔS))
        J[n*p+1:end, column] .= vec(X' * ΔX)
        column += 1
    end
    for col in 1:p, row in 1:p
        ΔX = zeros(ComplexF64, n, p)
        ΔS = zeros(ComplexF64, p, p)
        ΔS[row, col] = 1
        J[1:n*p, column] .= vec(polynomial_pair_residual_derivative(coeffs, X, S, ΔX, ΔS))
        J[n*p+1:end, column] .= vec(X' * ΔX)
        column += 1
    end

    rhs = vcat(-vec(R), zeros(ComplexF64, p * p))
    step = J \ rhs
    ΔX = reshape(step[1:n*p], n, p)
    ΔS = reshape(step[n*p+1:end], p, p)
    current = norm(R)
    best_X, best_S = X, S
    best_residual = current
    α = damping
    for _ in 1:8
        candidate_X, candidate_S = normalize_lifted_pair(X .+ α .* ΔX, S .+ α .* ΔS, lift)
        candidate_residual = norm(polynomial_pair_residual(coeffs, candidate_X, candidate_S))
        if candidate_residual < best_residual
            best_X, best_S = candidate_X, candidate_S
            best_residual = candidate_residual
            break
        end
        α /= 2
    end
    best_X, best_S, best_residual / max(current, eps(Float64))
end

function invariant_pair_newton_refine(coeffs, X, S; lift=1, steps=1)
    ratios = Float64[]
    for _ in 1:steps
        Xnew, Snew, ratio = invariant_pair_newton_step(coeffs, X, S; lift=lift)
        push!(ratios, ratio)
        X, S = Xnew, Snew
        ratio < 1 || break
    end
    X, S, ratios
end

function record_history!(history, iteration, coeffs, X, S, singular_values, center, radius; residual_tol=1e-8)
    R = polynomial_pair_residual(coeffs, X, S)
    pair_residual = norm(R) / max(norm(X), eps(Float64))
    lambdas, residuals, inside, converged_inside, max_inside =
        scalar_diagnostics(coeffs, X, S, center, radius; residual_tol=residual_tol)
    push!(
        history,
        PairHistory(
            iteration,
            size(X, 2),
            inside,
            converged_inside,
            pair_residual,
            max_inside,
            lambdas,
            residuals,
            singular_values,
        ),
    )
    history
end

function moment_rii_pair(coeffs, Xprobe; center, radius, nodes=32, iterations=3, moment_count=2, ranktol=1e-10, maxrank=typemax(Int), residual_tol=1e-8, restrict_inside=true, keep=0)
    z_nodes, z_weights = circular_rule(center, radius, nodes)
    moments = initial_moments(coeffs, Xprobe, z_nodes, z_weights, moment_count)
    X, S, _, singular_values = hankel_pair_identity(moments, moment_count; ranktol=ranktol, maxrank=maxrank)
    restrict_inside && ((X, S) = restrict_pair_to_contour(X, S, center, radius, moment_count; keep=keep))
    history = PairHistory[]
    record_history!(history, 0, coeffs, X, S, singular_values, center, radius; residual_tol=residual_tol)

    for iteration in 1:iterations
        R = polynomial_pair_residual(coeffs, X, S)
        moments = rii_moments(coeffs, X, S, R, z_nodes, z_weights, moment_count)
        X, S, _, singular_values = hankel_pair_identity(moments, moment_count; ranktol=ranktol, maxrank=maxrank)
        restrict_inside && ((X, S) = restrict_pair_to_contour(X, S, center, radius, moment_count; keep=keep))
        record_history!(history, iteration, coeffs, X, S, singular_values, center, radius; residual_tol=residual_tol)
    end
    X, S, history
end

function orthonormal_probe(probe)
    F = qr(probe)
    cols = min(size(probe)...)
    Matrix(F.Q)[:, 1:cols]
end

function moment_rii_pair_projected(coeffs, Xprobe, left_probe; center, radius, nodes=32, iterations=3, moment_count=2, ranktol=1e-10, maxrank=typemax(Int), residual_tol=1e-8, restrict_inside=true, keep=0, target_count=false)
    z_nodes, z_weights = circular_rule(center, radius, nodes)
    moments = initial_moments(coeffs, Xprobe, z_nodes, z_weights, moment_count)
    W = orthonormal_probe(left_probe)
    X, S, _, singular_values = projected_hankel_pair_identity(moments, W, moment_count; ranktol=ranktol, maxrank=maxrank)
    restrict_inside && ((X, S) = restrict_pair_to_contour(X, S, center, radius, moment_count; keep=keep, target_count=target_count))
    history = PairHistory[]
    record_history!(history, 0, coeffs, X, S, singular_values, center, radius; residual_tol=residual_tol)

    for iteration in 1:iterations
        R = polynomial_pair_residual(coeffs, X, S)
        moments = rii_moments(coeffs, X, S, R, z_nodes, z_weights, moment_count)
        X, S, _, singular_values = projected_hankel_pair_identity(moments, W, moment_count; ranktol=ranktol, maxrank=maxrank)
        restrict_inside && ((X, S) = restrict_pair_to_contour(X, S, center, radius, moment_count; keep=keep, target_count=target_count))
        record_history!(history, iteration, coeffs, X, S, singular_values, center, radius; residual_tol=residual_tol)
    end
    X, S, history
end

function moment_rii_pair_projected_scaled(
    coeffs,
    Xprobe,
    left_probe;
    center,
    radius,
    nodes=32,
    iterations=3,
    moment_count=2,
    ranktol=1e-10,
    maxrank=typemax(Int),
    residual_tol=1e-8,
    keep=0,
    target_count=false,
    update_mode=:projected,
    newton_steps=1,
)
    update_mode in (:projected, :shifted, :projected_newton) ||
        error("update_mode must be :projected, :shifted, or :projected_newton")

    z_nodes, z_weights = circular_rule(center, radius, nodes)
    W = orthonormal_probe(left_probe)
    moments = initial_polynomial_moments_scaled(coeffs, Xprobe, z_nodes, z_weights, center, radius, moment_count)
    X, Sμ, _, singular_values = projected_hankel_pair_identity(moments, W, moment_count; ranktol=ranktol, maxrank=maxrank)
    X, Sμ = restrict_scaled_pair_to_contour(X, Sμ, center, radius, moment_count; keep=keep, target_count=target_count)
    Sλ = lambda_matrix_from_scaled(Sμ, center, radius)
    update_mode === :projected_newton && ((X, Sλ, _) = invariant_pair_newton_refine(coeffs, X, Sλ; lift=moment_count, steps=newton_steps))
    Sμ = (Sλ .- center .* Matrix{ComplexF64}(I, size(Sλ, 1), size(Sλ, 2))) ./ radius

    history = PairHistory[]
    record_history!(history, 0, coeffs, X, Sλ, singular_values, center, radius; residual_tol=residual_tol)

    for iteration in 1:iterations
        Sλ = lambda_matrix_from_scaled(Sμ, center, radius)
        moments = rii_polynomial_moments_scaled(coeffs, X, Sλ, z_nodes, z_weights, center, radius, moment_count)
        if update_mode === :shifted
            X, Sμ, _, singular_values = shifted_pair_identity(moments, moment_count; ranktol=ranktol, maxrank=maxrank)
        else
            X, Sμ, _, singular_values = projected_hankel_pair_identity(moments, W, moment_count; ranktol=ranktol, maxrank=maxrank)
        end
        X, Sμ = restrict_scaled_pair_to_contour(X, Sμ, center, radius, moment_count; keep=keep, target_count=target_count)
        Sλ = lambda_matrix_from_scaled(Sμ, center, radius)
        if update_mode === :projected_newton
            X, Sλ, _ = invariant_pair_newton_refine(coeffs, X, Sλ; lift=moment_count, steps=newton_steps)
            Sμ = (Sλ .- center .* Matrix{ComplexF64}(I, size(Sλ, 1), size(Sλ, 2))) ./ radius
        end
        record_history!(history, iteration, coeffs, X, Sλ, singular_values, center, radius; residual_tol=residual_tol)
    end

    X, lambda_matrix_from_scaled(Sμ, center, radius), history
end

function initial_moments_generic(Tsolve, Xprobe, z_nodes, z_weights, moment_count)
    n, m = size(Xprobe)
    moments = [zeros(ComplexF64, n, m) for _ in 1:(2 * moment_count)]
    for (z, weight) in zip(z_nodes, z_weights)
        solved = Tsolve(z, Xprobe)
        zpower = one(ComplexF64)
        for p in eachindex(moments)
            moments[p] .+= (weight * zpower) .* solved
            zpower *= z
        end
    end
    moments
end

function rii_moments_generic(Tsolve, pair_residual, X, S, z_nodes, z_weights, moment_count)
    n, p = size(X)
    moments = [zeros(ComplexF64, n, p) for _ in 1:(2 * moment_count)]
    I_p = Matrix{ComplexF64}(I, p, p)
    R = pair_residual(X, S)
    for (z, weight) in zip(z_nodes, z_weights)
        corrected = X - Tsolve(z, R)
        corrected = corrected / (z * I_p - S)
        zpower = one(ComplexF64)
        for q in eachindex(moments)
            moments[q] .+= (weight * zpower) .* corrected
            zpower *= z
        end
    end
    moments
end

function generic_scalar_diagnostics(pair_residual, scalar_residual, X, S, center, radius; residual_tol=1e-8)
    F = eigen(S)
    lambdas = ComplexF64.(F.values)
    vectors = X * F.vectors
    residuals = zeros(Float64, length(lambdas))
    for j in eachindex(lambdas)
        residuals[j] = scalar_residual(lambdas[j], vectors[:, j])
    end
    inside = FEASTSolver.in_contour(lambdas, center, radius)
    converged_inside = count(inside .& (residuals .<= residual_tol))
    max_inside = any(inside) ? maximum(residuals[inside]) : Inf
    pair_res = norm(pair_residual(X, S)) / max(norm(lifted_pair_matrix(X, S, 1)), eps(Float64))
    lambdas, residuals, count(inside), converged_inside, max_inside, pair_res
end

function moment_rii_pair_generic(Tsolve, pair_residual, scalar_residual, Xprobe; center, radius, nodes=64, iterations=4, moment_count=2, ranktol=1e-10, maxrank=typemax(Int), residual_tol=1e-8, keep=0)
    z_nodes, z_weights = circular_rule(center, radius, nodes)
    moments = initial_moments_generic(Tsolve, Xprobe, z_nodes, z_weights, moment_count)
    X, S, _, singular_values = hankel_pair_identity(moments, moment_count; ranktol=ranktol, maxrank=maxrank)
    X, S = restrict_pair_to_contour(X, S, center, radius, moment_count; keep=keep)
    history = PairHistory[]
    lambdas, residuals, inside, converged_inside, max_inside, pair_res =
        generic_scalar_diagnostics(pair_residual, scalar_residual, X, S, center, radius; residual_tol=residual_tol)
    push!(history, PairHistory(0, size(X, 2), inside, converged_inside, pair_res, max_inside, lambdas, residuals, singular_values))

    for iteration in 1:iterations
        moments = rii_moments_generic(Tsolve, pair_residual, X, S, z_nodes, z_weights, moment_count)
        X, S, _, singular_values = hankel_pair_identity(moments, moment_count; ranktol=ranktol, maxrank=maxrank)
        X, S = restrict_pair_to_contour(X, S, center, radius, moment_count; keep=keep)
        lambdas, residuals, inside, converged_inside, max_inside, pair_res =
            generic_scalar_diagnostics(pair_residual, scalar_residual, X, S, center, radius; residual_tol=residual_tol)
        push!(history, PairHistory(iteration, size(X, 2), inside, converged_inside, pair_res, max_inside, lambdas, residuals, singular_values))
    end
    X, S, history
end

function initial_moments_generic_scaled(Tsolve, Xprobe, z_nodes, z_weights, center, radius, moment_count)
    n, m = size(Xprobe)
    moments = [zeros(ComplexF64, n, m) for _ in 1:(2 * moment_count)]
    for (z, weight) in zip(z_nodes, z_weights)
        solved = Tsolve(z, Xprobe)
        μ = (z - center) / radius
        μpower = one(ComplexF64)
        for p in eachindex(moments)
            moments[p] .+= (weight * μpower) .* solved
            μpower *= μ
        end
    end
    moments
end

function rii_moments_generic_scaled(Tsolve, pair_residual, X, Sλ, z_nodes, z_weights, center, radius, moment_count)
    n, p = size(X)
    moments = [zeros(ComplexF64, n, p) for _ in 1:(2 * moment_count)]
    I_p = Matrix{ComplexF64}(I, p, p)
    R = pair_residual(X, Sλ)
    for (z, weight) in zip(z_nodes, z_weights)
        corrected = X - Tsolve(z, R)
        corrected = corrected / (z * I_p - Sλ)
        μ = (z - center) / radius
        μpower = one(ComplexF64)
        for q in eachindex(moments)
            moments[q] .+= (weight * μpower) .* corrected
            μpower *= μ
        end
    end
    moments
end

function rii_chebyshev_moments_generic_scaled(Tsolve, pair_residual, X, Sλ, z_nodes, z_weights, center, radius, moment_count)
    n, p = size(X)
    moments = [zeros(ComplexF64, n, p) for _ in 1:(moment_count + 1)]
    I_p = Matrix{ComplexF64}(I, p, p)
    R = pair_residual(X, Sλ)
    for (z, weight) in zip(z_nodes, z_weights)
        corrected = X - Tsolve(z, R)
        corrected = corrected / (z * I_p - Sλ)
        μ = (z - center) / radius
        tprev = one(ComplexF64)
        moments[1] .+= weight .* corrected
        moment_count == 0 && continue

        tcurr = μ
        moments[2] .+= (weight * tcurr) .* corrected
        for q in 3:length(moments)
            tnext = 2 * μ * tcurr - tprev
            moments[q] .+= (weight * tnext) .* corrected
            tprev, tcurr = tcurr, tnext
        end
    end
    moments
end

lambda_matrix_from_scaled(Sμ, center, radius) = center * Matrix{ComplexF64}(I, size(Sμ, 1), size(Sμ, 2)) + radius * Sμ

function restrict_scaled_pair_to_contour(X, Sμ, center, radius, lift; keep=0, target_count=false)
    Sλ = lambda_matrix_from_scaled(Sμ, center, radius)
    Xrestricted, Sλrestricted = restrict_pair_to_contour(X, Sλ, center, radius, lift; keep=keep, target_count=target_count)
    Sμrestricted = (Sλrestricted - center * Matrix{ComplexF64}(I, size(Sλrestricted, 1), size(Sλrestricted, 2))) / radius
    Xrestricted, Sμrestricted
end

function linear_initial_moments_scaled(A, Xprobe, z_nodes, z_weights, center, radius, moment_count)
    n, m = size(Xprobe)
    I_n = Matrix{ComplexF64}(I, n, n)
    moments = [zeros(ComplexF64, n, m) for _ in 1:(2 * moment_count)]
    for (z, weight) in zip(z_nodes, z_weights)
        solved = (z .* I_n .- A) \ Xprobe
        μ = (z - center) / radius
        μpower = one(ComplexF64)
        for p in eachindex(moments)
            moments[p] .+= (weight * μpower) .* solved
            μpower *= μ
        end
    end
    moments
end

function linear_pair_residual(A, X, Sλ)
    R = X * Sλ
    R .-= A * X
    R
end

function linear_rii_moments_scaled(A, X, Sλ, z_nodes, z_weights, center, radius, moment_count)
    n, p = size(X)
    I_n = Matrix{ComplexF64}(I, n, n)
    I_p = Matrix{ComplexF64}(I, p, p)
    R = linear_pair_residual(A, X, Sλ)
    moments = [zeros(ComplexF64, n, p) for _ in 1:(2 * moment_count)]
    for (z, weight) in zip(z_nodes, z_weights)
        corrected = X - (z .* I_n .- A) \ R
        corrected = corrected / (z * I_p - Sλ)
        μ = (z - center) / radius
        μpower = one(ComplexF64)
        for q in eachindex(moments)
            moments[q] .+= (weight * μpower) .* corrected
            μpower *= μ
        end
    end
    moments
end

function linear_scalar_diagnostics(A, X, Sλ, center, radius; residual_tol=1e-10)
    F = eigen(Sλ)
    lambdas = ComplexF64.(F.values)
    vectors = X * F.vectors
    residuals = zeros(Float64, length(lambdas))
    Anorm = max(norm(A), eps(Float64))
    for j in eachindex(lambdas)
        v = vectors[:, j]
        vnorm = norm(v)
        if vnorm <= eps(Float64)
            residuals[j] = Inf
        else
            v ./= vnorm
            residuals[j] = norm(A * v - lambdas[j] * v) / Anorm
        end
    end
    inside = FEASTSolver.in_contour(lambdas, center, radius)
    converged_inside = count(inside .& (residuals .<= residual_tol))
    max_inside = any(inside) ? maximum(residuals[inside]) : Inf
    pair_res = norm(linear_pair_residual(A, X, Sλ)) / max(norm(lifted_pair_matrix(X, Sλ, 1)), eps(Float64))
    lambdas, residuals, count(inside), converged_inside, max_inside, pair_res
end

function linear_ss_feast_projected(A, Xprobe, left_probe; center, radius, nodes=32, iterations=4, moment_count=2, ranktol=1e-10, maxrank=typemax(Int), residual_tol=1e-10, keep=0, target_count=false)
    z_nodes, z_weights = circular_rule(center, radius, nodes)
    W = orthonormal_probe(left_probe)
    moments = linear_initial_moments_scaled(A, Xprobe, z_nodes, z_weights, center, radius, moment_count)
    X, Sμ, _, singular_values = projected_hankel_pair_identity(moments, W, moment_count; ranktol=ranktol, maxrank=maxrank)
    X, Sμ = restrict_scaled_pair_to_contour(X, Sμ, center, radius, moment_count; keep=keep, target_count=target_count)
    Sλ = lambda_matrix_from_scaled(Sμ, center, radius)
    history = PairHistory[]
    lambdas, residuals, inside, converged_inside, max_inside, pair_res =
        linear_scalar_diagnostics(A, X, Sλ, center, radius; residual_tol=residual_tol)
    push!(history, PairHistory(0, size(X, 2), inside, converged_inside, pair_res, max_inside, lambdas, residuals, singular_values))

    for iteration in 1:iterations
        Sλ = lambda_matrix_from_scaled(Sμ, center, radius)
        moments = linear_rii_moments_scaled(A, X, Sλ, z_nodes, z_weights, center, radius, moment_count)
        X, Sμ, _, singular_values = projected_hankel_pair_identity(moments, W, moment_count; ranktol=ranktol, maxrank=maxrank)
        X, Sμ = restrict_scaled_pair_to_contour(X, Sμ, center, radius, moment_count; keep=keep, target_count=target_count)
        Sλ = lambda_matrix_from_scaled(Sμ, center, radius)
        lambdas, residuals, inside, converged_inside, max_inside, pair_res =
            linear_scalar_diagnostics(A, X, Sλ, center, radius; residual_tol=residual_tol)
        push!(history, PairHistory(iteration, size(X, 2), inside, converged_inside, pair_res, max_inside, lambdas, residuals, singular_values))
    end
    X, lambda_matrix_from_scaled(Sμ, center, radius), history
end

function moment_rii_pair_generic_scaled(Tsolve, pair_residual, scalar_residual, Xprobe; center, radius, nodes=64, iterations=4, moment_count=2, ranktol=1e-10, maxrank=typemax(Int), residual_tol=1e-8, keep=0, extraction=:hankel)
    extraction in (:hankel, :shifted, :chebyshev_shifted) || error("extraction must be :hankel, :shifted, or :chebyshev_shifted")
    z_nodes, z_weights = circular_rule(center, radius, nodes)
    moments = initial_moments_generic_scaled(Tsolve, Xprobe, z_nodes, z_weights, center, radius, moment_count)
    X, Sμ, _, singular_values = hankel_pair_identity(moments, moment_count; ranktol=ranktol, maxrank=maxrank)
    X, Sμ = restrict_scaled_pair_to_contour(X, Sμ, center, radius, moment_count; keep=keep)
    Sλ = lambda_matrix_from_scaled(Sμ, center, radius)
    history = PairHistory[]
    lambdas, residuals, inside, converged_inside, max_inside, pair_res =
        generic_scalar_diagnostics(pair_residual, scalar_residual, X, Sλ, center, radius; residual_tol=residual_tol)
    push!(history, PairHistory(0, size(X, 2), inside, converged_inside, pair_res, max_inside, lambdas, residuals, singular_values))

    for iteration in 1:iterations
        Sλ = lambda_matrix_from_scaled(Sμ, center, radius)
        if extraction === :hankel
            moments = rii_moments_generic_scaled(Tsolve, pair_residual, X, Sλ, z_nodes, z_weights, center, radius, moment_count)
            X, Sμ, _, singular_values = hankel_pair_identity(moments, moment_count; ranktol=ranktol, maxrank=maxrank)
        elseif extraction === :shifted
            moments = rii_moments_generic_scaled(Tsolve, pair_residual, X, Sλ, z_nodes, z_weights, center, radius, moment_count)
            X, Sμ, _, singular_values = shifted_pair_identity(moments, moment_count; ranktol=ranktol, maxrank=maxrank)
        else
            moments = rii_chebyshev_moments_generic_scaled(Tsolve, pair_residual, X, Sλ, z_nodes, z_weights, center, radius, moment_count)
            X, Sμ, _, singular_values = chebyshev_shifted_pair_identity(moments, moment_count; ranktol=ranktol, maxrank=maxrank)
        end
        X, Sμ = restrict_scaled_pair_to_contour(X, Sμ, center, radius, moment_count; keep=keep)
        Sλ = lambda_matrix_from_scaled(Sμ, center, radius)
        lambdas, residuals, inside, converged_inside, max_inside, pair_res =
            generic_scalar_diagnostics(pair_residual, scalar_residual, X, Sλ, center, radius; residual_tol=residual_tol)
        push!(history, PairHistory(iteration, size(X, 2), inside, converged_inside, pair_res, max_inside, lambdas, residuals, singular_values))
    end
    X, lambda_matrix_from_scaled(Sμ, center, radius), history
end

function run_scalar_sine_realization_sweep(; radius=10.0, node_values=(32, 64), iterations=8, moment_counts=(7,), ranktol=1e-10, residual_tol=1e-8)
    center = 0.0 + 0.0im
    expected = ComplexF64[k * pi for k in -floor(Int, radius / pi):floor(Int, radius / pi)]
    expected = expected[FEASTSolver.in_contour(expected, center, radius)]
    println()
    println("Scalar sine realization update: n=1, expected_inside=$(length(expected)), radius=$radius")
    println("  compares repeated Hankel extraction with shifted L0*S≈L1 realization after each correction")

    Tsolve = (z, B) -> B ./ sin(z)
    pair_residual = (X, S) -> X * sin(S)
    scalar_residual = (λ, v) -> abs(sin(λ)) * norm(v) / max(norm(v), eps(Float64))

    for extraction in (:hankel, :shifted, :chebyshev_shifted)
        for moment_count in moment_counts
            for nodes in node_values
                Xprobe = ones(ComplexF64, 1, 1)
                maxrank = moment_count
                history = try
                    last(moment_rii_pair_generic_scaled(
                        Tsolve,
                        pair_residual,
                        scalar_residual,
                        Xprobe;
                        center=center,
                        radius=radius,
                        nodes=nodes,
                        iterations=iterations,
                        moment_count=moment_count,
                        ranktol=ranktol,
                        maxrank=maxrank,
                        residual_tol=residual_tol,
                        keep=length(expected),
                        extraction=extraction,
                    ))
                catch err
                    println("  extraction=$extraction K=$moment_count nodes=$nodes failed: $(typeof(err)) $err")
                    continue
                end
                h0 = first(history)
                hend = last(history)
                @printf(
                    "  extraction=%s K=%d nodes=%d iter0(conv=%d/%d max=%.3e pair=%.3e) final(rank=%d conv=%d/%d max=%.3e pair=%.3e)\n",
                    string(extraction),
                    moment_count,
                    nodes,
                    h0.converged_inside,
                    h0.inside,
                    h0.max_inside_residual,
                    h0.pair_residual,
                    hend.rank,
                    hend.converged_inside,
                    hend.inside,
                    hend.max_inside_residual,
                    hend.pair_residual,
                )
            end
        end
    end
end

function companion_reference(coeffs, center, radius)
    lambdas, _, residuals = companion(coeffs)
    inside = FEASTSolver.in_contour(lambdas, center, radius) .& (residuals .< 1e-7)
    ComplexF64.(lambdas[inside])
end

function print_history(problem, label, history, expected_count)
    println()
    println("Problem: $problem")
    println("  $label")
    println("  expected_inside: $expected_count")
    for h in history
        s1 = isempty(h.singular_values) ? NaN : h.singular_values[1]
        srank = length(h.singular_values) >= h.rank ? h.singular_values[h.rank] : NaN
        @printf(
            "    iter=%d rank=%d inside=%d conv_inside=%d pair_res=%.3e max_inside_res=%.3e sigma_rank/sigma1=%.3e\n",
            h.iteration,
            h.rank,
            h.inside,
            h.converged_inside,
            h.pair_residual,
            h.max_inside_residual,
            srank / s1,
        )
    end
end

function diagonal_linear_problem()
    n = 10
    A = Diagonal(ComplexF64.(1:n))
    coeffs = [-Matrix(A), Matrix{ComplexF64}(I, n, n)]
    coeffs, 2.5 + 0im, 1.6, 4
end

function deficient_quadratic_problem()
    A0 = Matrix{ComplexF64}(Matrix(mmread(joinpath(REPO_ROOT, "data", "quadraticM0.mtx"))))
    A1 = Matrix{ComplexF64}(Matrix(mmread(joinpath(REPO_ROOT, "data", "quadraticM1.mtx"))))
    coeffs = [A0 - 0.02 * A1, 0.1 * A1, A1]
    coeffs, 0.0 + 0im, 0.25, 3
end

function butterfly_problem()
    N = diagm(-1 => ones(7))
    Mh0 = (4I + N + N') / 6
    Mh1 = N - N'
    Mh2 = -(2I - N - N')
    Mh3 = Mh1
    Mh4 = -Mh2
    c = [0.6 1.3; 1.3 0.1; 0.1 1.2; 1.0 1.0; 1.2 1.0]
    I8 = Matrix(I, 8, 8)
    coeffs = [
        c[1, 1] * kron(I8, Mh0) + c[1, 2] * kron(Mh0, I8),
        c[2, 1] * kron(I8, Mh1) + c[2, 2] * kron(Mh1, I8),
        c[3, 1] * kron(I8, Mh2) + c[3, 2] * kron(Mh2, I8),
        c[4, 1] * kron(I8, Mh3) + c[4, 2] * kron(Mh3, I8),
        c[5, 1] * kron(I8, Mh4) + c[5, 2] * kron(Mh4, I8),
    ]
    complex.(coeffs), 1.0 + 1.0im, 0.5, 16
end

function many_eigenvalue_diagonal_matrix()
    n = 20
    A = Matrix(Diagonal(ComplexF64.(1:n)))
    A, 5.5 + 0im, 5.1, 4
end

function grcar_matrix(n, bands=3)
    A = zeros(ComplexF64, n, n)
    for i in 1:n
        A[i, i] = 1
        i > 1 && (A[i, i - 1] = -1)
        for j in 1:bands
            i + j <= n && (A[i, i + j] = 1)
        end
    end
    A
end

function grcar_linear_matrix()
    A = grcar_matrix(32)
    A, 0.2 + 1.6im, 1.05, 8
end

function linear_reference_inside(A, center, radius)
    lambdas = ComplexF64.(eigvals(A))
    inside = FEASTSolver.in_contour(lambdas, center, radius)
    lambdas[inside]
end

function many_eigenvalue_quadratic_problem()
    n = 12
    inside_roots = ComplexF64[1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 6.0, 6.5, 7.0, 7.5]
    outside_roots = ComplexF64[16 + i for i in 1:n]
    coeffs = [
        Matrix(Diagonal(inside_roots .* outside_roots)),
        Matrix(Diagonal(-(inside_roots .+ outside_roots))),
        Matrix{ComplexF64}(I, n, n),
    ]
    coeffs, 4.25 + 0im, 3.75, 4
end

function polynomial_coefficients_from_roots(roots)
    coeffs = ComplexF64[1]
    for root in roots
        next = zeros(ComplexF64, length(coeffs) + 1)
        for (j, coeff) in pairs(coeffs)
            next[j] -= root * coeff
            next[j + 1] += coeff
        end
        coeffs = next
    end
    coeffs
end

function many_eigenvalue_nonnormal_polynomial_problem()
    n = 4
    degree = 8
    center = 0.0 + 0.0im
    radius = 2.5
    roots_by_direction = [
        ComplexF64[-2.0, -1.2, -0.2, 0.8, 1.7, 6.0, 7.0, 8.0],
        ComplexF64[-1.8, -1.0, 0.55, 1.0, 1.9, 6.5, 7.5, 8.5],
        ComplexF64[-1.6, -0.8, 0.2, 1.2, 2.1, 7.0, 8.0, 9.0],
        ComplexF64[-1.4, -0.6, 0.4, 1.4, 2.3, 7.5, 8.5, 9.5],
    ]
    scalar_coeffs = [polynomial_coefficients_from_roots(roots) for roots in roots_by_direction]

    # A single non-unitary similarity keeps the roots exact but makes the
    # physical eigenvectors strongly reused across many nonlinear eigenvalues.
    V = ComplexF64[
        1.0 0.7 -0.4 0.2
        0.2 1.0 0.8 -0.3
        -0.5 0.1 1.0 0.6
        0.4 -0.2 0.3 1.0
    ]
    Vinv = inv(V)
    coeffs = Matrix{ComplexF64}[]
    for j in 1:(degree + 1)
        push!(coeffs, V * Diagonal([scalar_coeffs[i][j] for i in 1:n]) * Vinv)
    end
    coeffs, center, radius, n
end

function match_expected_count(values, expected; atol)
    isempty(expected) && return 0
    count(expected) do λ
        !isempty(values) && minimum(abs.(values .- λ)) <= atol
    end
end

function state_contamination_diagnostics(coeffs, X, S, expected, center, radius; residual_tol=1e-8, match_tol=1e-6)
    lambdas, residuals, inside_count, converged_inside, max_inside = scalar_diagnostics(coeffs, X, S, center, radius; residual_tol=residual_tol)
    inside = FEASTSolver.in_contour(lambdas, center, radius)
    matched = match_expected_count(lambdas[inside], expected; atol=match_tol)
    extra_inside = max(inside_count - matched, 0)
    pair_all = norm(polynomial_pair_residual(coeffs, X, S)) / max(norm(X), eps(Float64))

    wanted = inside .& (residuals .<= max(residual_tol, match_tol))
    pair_wanted = Inf
    if any(wanted)
        F = eigen(S)
        Xwanted = X * F.vectors[:, wanted]
        Swanted = Diagonal(ComplexF64.(F.values[wanted]))
        pair_wanted = norm(polynomial_pair_residual(coeffs, Xwanted, Swanted)) / max(norm(Xwanted), eps(Float64))
    end

    (
        inside=inside_count,
        converged_inside=converged_inside,
        matched=matched,
        extra_inside=extra_inside,
        max_inside=max_inside,
        pair_all=pair_all,
        pair_wanted=pair_wanted,
    )
end

function retained_singular_ratio(history)
    h = last(history)
    (isempty(h.singular_values) || h.rank == 0) && return NaN
    index = min(h.rank, length(h.singular_values))
    h.singular_values[index] / h.singular_values[1]
end

function run_case(name, make_problem; nodes, iterations, moment_counts, seed=4401, ranktol=1e-10, residual_tol=1e-8, keep_extra=0)
    coeffs, center, radius, probe_cols = make_problem()
    expected = companion_reference(coeffs, center, radius)
    println("Reference $name: $(length(expected)) eigenvalues inside contour")

    for moment_count in moment_counts
        Random.seed!(seed + moment_count)
        Xprobe = rand(ComplexF64, size(coeffs[1], 1), probe_cols)
        maxrank = min(moment_count * size(coeffs[1], 1), moment_count * probe_cols)
        keep = min(maxrank, length(expected) + keep_extra)
        _, _, history = moment_rii_pair(
            coeffs,
            Xprobe;
            center=center,
            radius=radius,
            nodes=nodes,
            iterations=iterations,
            moment_count=moment_count,
            ranktol=ranktol,
            maxrank=maxrank,
            residual_tol=residual_tol,
            restrict_inside=true,
            keep=keep,
        )
        print_history(name, "restricted invariant-pair moment RII K=$moment_count nodes=$nodes probe_cols=$probe_cols keep=$keep", history, length(expected))
    end
end

function run_node_sweep(name, make_problem; node_values, iterations, moment_count, seed=5501, ranktol=1e-10, residual_tol=1e-8, keep_extra=0)
    coeffs, center, radius, probe_cols = make_problem()
    expected = companion_reference(coeffs, center, radius)
    println()
    println("Node sweep: $name K=$moment_count expected_inside=$(length(expected))")
    for nodes in node_values
        Random.seed!(seed + nodes + 100 * moment_count)
        Xprobe = rand(ComplexF64, size(coeffs[1], 1), probe_cols)
        maxrank = min(moment_count * size(coeffs[1], 1), moment_count * probe_cols)
        keep = min(maxrank, length(expected) + keep_extra)
        _, _, history = moment_rii_pair(
            coeffs,
            Xprobe;
            center=center,
            radius=radius,
            nodes=nodes,
            iterations=iterations,
            moment_count=moment_count,
            ranktol=ranktol,
            maxrank=maxrank,
            residual_tol=residual_tol,
            restrict_inside=true,
            keep=keep,
        )
        h0 = first(history)
        hend = last(history)
        @printf(
            "  nodes=%d iter0(conv=%d/%d max=%.3e pair=%.3e) final(iter=%d rank=%d conv=%d/%d max=%.3e pair=%.3e)\n",
            nodes,
            h0.converged_inside,
            h0.inside,
            h0.max_inside_residual,
            h0.pair_residual,
            hend.iteration,
            hend.rank,
            hend.converged_inside,
            hend.inside,
            hend.max_inside_residual,
            hend.pair_residual,
        )
    end
end

function run_projected_hankel_case(name, make_problem; nodes, iterations, moment_count, seed=6601, ranktol=1e-10, residual_tol=1e-8, keep_extra=0, left_cols=nothing, target_count=false)
    coeffs, center, radius, probe_cols = make_problem()
    expected = companion_reference(coeffs, center, radius)
    n = size(coeffs[1], 1)
    ell = min(n, something(left_cols, probe_cols))
    Random.seed!(seed)
    Xprobe = rand(ComplexF64, n, probe_cols)
    W = rand(ComplexF64, n, ell)
    maxrank = min(moment_count * probe_cols, moment_count * ell)
    keep = min(maxrank, length(expected) + keep_extra)
    _, _, history = moment_rii_pair_projected(
        coeffs,
        Xprobe,
        W;
        center=center,
        radius=radius,
        nodes=nodes,
        iterations=iterations,
        moment_count=moment_count,
        ranktol=ranktol,
        maxrank=maxrank,
        residual_tol=residual_tol,
        restrict_inside=true,
        keep=keep,
        target_count=target_count,
    )
    println()
    println("Two-sided projected Hankel: $name K=$moment_count nodes=$nodes")
    println("  expected_inside: $(length(expected))")
    println("  full Hankel shape would be ($(moment_count * n), $(moment_count * probe_cols)); projected shape is ($(moment_count * ell), $(moment_count * probe_cols)); target_count=$target_count")
    h0 = first(history)
    hend = last(history)
    @printf(
        "  iter0(rank=%d conv=%d/%d max=%.3e pair=%.3e) final(iter=%d rank=%d conv=%d/%d max=%.3e pair=%.3e)\n",
        h0.rank,
        h0.converged_inside,
        h0.inside,
        h0.max_inside_residual,
        h0.pair_residual,
        hend.iteration,
        hend.rank,
        hend.converged_inside,
        hend.inside,
        hend.max_inside_residual,
        hend.pair_residual,
    )
end

function run_nonlinear_update_comparison(
    name,
    make_problem;
    nodes,
    iterations,
    moment_count,
    seed=8801,
    ranktol=1e-10,
    residual_tol=1e-8,
    keep_extra=0,
    left_cols=nothing,
    target_count=false,
    newton_steps=1,
    modes=(:projected, :shifted, :projected_newton),
)
    coeffs, center, radius, probe_cols = make_problem()
    expected = companion_reference(coeffs, center, radius)
    n = size(coeffs[1], 1)
    ell = min(n, something(left_cols, probe_cols))
    Random.seed!(seed)
    Xprobe = rand(ComplexF64, n, probe_cols)
    W = rand(ComplexF64, n, ell)
    maxrank = min(moment_count * probe_cols, moment_count * ell)
    keep = min(maxrank, length(expected) + keep_extra)

    println()
    println("Nonlinear projected update comparison: $name K=$moment_count nodes=$nodes")
    println("  expected_inside: $(length(expected)); probe_cols=$probe_cols; left_cols=$ell; keep=$keep; target_count=$target_count")
    println("  projected Hankel shape: ($(moment_count * ell), $(moment_count * probe_cols))")

    for mode in modes
        X, S, history = try
            moment_rii_pair_projected_scaled(
                coeffs,
                Xprobe,
                W;
                center=center,
                radius=radius,
                nodes=nodes,
                iterations=iterations,
                moment_count=moment_count,
                ranktol=ranktol,
                maxrank=maxrank,
                residual_tol=residual_tol,
                keep=keep,
                target_count=target_count,
                update_mode=mode,
                newton_steps=newton_steps,
            )
        catch err
            println("  mode=$mode failed: $(typeof(err)) $err")
            continue
        end
        h0 = first(history)
        hend = last(history)
        diag = state_contamination_diagnostics(
            coeffs,
            X,
            S,
            expected,
            center,
            radius;
            residual_tol=residual_tol,
            match_tol=1e-6,
        )
        sigma_ratio = retained_singular_ratio(history)
        @printf(
            "  mode=%s iter0(rank=%d conv=%d/%d max=%.3e pair=%.3e) final(rank=%d conv=%d/%d matched=%d extra=%d max=%.3e pair_all=%.3e pair_wanted=%.3e sigma_rank/sigma1=%.3e)\n",
            string(mode),
            h0.rank,
            h0.converged_inside,
            h0.inside,
            h0.max_inside_residual,
            h0.pair_residual,
            hend.rank,
            diag.converged_inside,
            diag.inside,
            diag.matched,
            diag.extra_inside,
            diag.max_inside,
            diag.pair_all,
            diag.pair_wanted,
            sigma_ratio,
        )
    end
end

function run_linear_ss_feast_control(name, make_problem; nodes, iterations, moment_count, seed=7701, ranktol=1e-10, residual_tol=1e-10, left_cols=nothing, target_count=false)
    A, center, radius, probe_cols = make_problem()
    expected = linear_reference_inside(A, center, radius)
    n = size(A, 1)
    ell = min(n, something(left_cols, probe_cols))
    Random.seed!(seed)
    Xprobe = rand(ComplexF64, n, probe_cols)
    W = rand(ComplexF64, n, ell)
    maxrank = min(n, moment_count * probe_cols, moment_count * ell)
    keep = min(maxrank, length(expected))

    feast_values, _, feast_residuals = feast!(
        copy(Xprobe),
        A;
        nodes=nodes,
        iter=iterations,
        c=center,
        r=radius,
        ϵ=residual_tol,
        store=false,
    )
    Xwide = rand(ComplexF64, n, maxrank)
    wide_feast_values, _, wide_feast_residuals = feast!(
        Xwide,
        A;
        nodes=nodes,
        iter=iterations,
        c=center,
        r=radius,
        ϵ=residual_tol,
        store=false,
    )

    _, _, history = linear_ss_feast_projected(
        A,
        Xprobe,
        W;
        center=center,
        radius=radius,
        nodes=nodes,
        iterations=iterations,
        moment_count=moment_count,
        ranktol=ranktol,
        maxrank=maxrank,
        residual_tol=residual_tol,
        keep=keep,
        target_count=target_count,
    )

    h0 = first(history)
    hend = last(history)
    println()
    println("Linear SS-FEAST control: $name K=$moment_count nodes=$nodes")
    println("  expected_inside: $(length(expected)); probe_cols=$probe_cols; left_cols=$ell")
    println("  standard FEAST with same probe cols returned $(length(feast_values)) values; max_res=$(isempty(feast_residuals) ? Inf : maximum(feast_residuals))")
    println("  standard FEAST with $(maxrank) probe cols returned $(length(wide_feast_values)) values; max_res=$(isempty(wide_feast_residuals) ? Inf : maximum(wide_feast_residuals))")
    println("  projected Hankel shape: ($(moment_count * ell), $(moment_count * probe_cols)); target_count=$target_count")
    @printf(
        "  SS iter0(rank=%d conv=%d/%d max=%.3e pair=%.3e) final(iter=%d rank=%d conv=%d/%d max=%.3e pair=%.3e)\n",
        h0.rank,
        h0.converged_inside,
        h0.inside,
        h0.max_inside_residual,
        h0.pair_residual,
        hend.iteration,
        hend.rank,
        hend.converged_inside,
        hend.inside,
        hend.max_inside_residual,
        hend.pair_residual,
    )
end

function run_scalar_sine_sweep(; radius=10.0, node_values=(16, 24, 32, 48, 64), iterations=8, moment_counts=(4, 6, 8), ranktol=1e-10, residual_tol=1e-8)
    center = 0.0 + 0.0im
    expected = ComplexF64[k * pi for k in -floor(Int, radius / pi):floor(Int, radius / pi)]
    expected = expected[FEASTSolver.in_contour(expected, center, radius)]
    println()
    println("Scalar sine NEP: n=1, expected_inside=$(length(expected)), radius=$radius")
    println("  T(z)=sin(z), pair residual T(X,S)=X*sin(S)")

    Tsolve = (z, B) -> B ./ sin(z)
    pair_residual = (X, S) -> X * sin(S)
    scalar_residual = (λ, v) -> abs(sin(λ)) * norm(v) / max(norm(v), eps(Float64))

    for moment_count in moment_counts
        for nodes in node_values
            Xprobe = ones(ComplexF64, 1, 1)
            maxrank = moment_count
            _, _, history = moment_rii_pair_generic(
                Tsolve,
                pair_residual,
                scalar_residual,
                Xprobe;
                center=center,
                radius=radius,
                nodes=nodes,
                iterations=iterations,
                moment_count=moment_count,
                ranktol=ranktol,
                maxrank=maxrank,
                residual_tol=residual_tol,
                keep=length(expected),
            )
            h0 = first(history)
            hend = last(history)
            @printf(
                "  K=%d nodes=%d iter0(conv=%d/%d max=%.3e pair=%.3e) final(rank=%d conv=%d/%d max=%.3e pair=%.3e)\n",
                moment_count,
                nodes,
                h0.converged_inside,
                h0.inside,
                h0.max_inside_residual,
                h0.pair_residual,
                hend.rank,
                hend.converged_inside,
                hend.inside,
                hend.max_inside_residual,
                hend.pair_residual,
            )
        end
    end
end

function run_scalar_sine_scaled_sweep(; radius=10.0, node_values=(16, 24, 32, 48, 64), iterations=8, moment_counts=(4, 6, 8), ranktol=1e-10, residual_tol=1e-8)
    center = 0.0 + 0.0im
    expected = ComplexF64[k * pi for k in -floor(Int, radius / pi):floor(Int, radius / pi)]
    expected = expected[FEASTSolver.in_contour(expected, center, radius)]
    println()
    println("Scaled scalar sine NEP: n=1, expected_inside=$(length(expected)), radius=$radius")
    println("  moments use mu=(z-c)/r; pair residual still uses lambda-matrix S")

    Tsolve = (z, B) -> B ./ sin(z)
    pair_residual = (X, S) -> X * sin(S)
    scalar_residual = (λ, v) -> abs(sin(λ)) * norm(v) / max(norm(v), eps(Float64))

    for moment_count in moment_counts
        for nodes in node_values
            Xprobe = ones(ComplexF64, 1, 1)
            maxrank = moment_count
            history = try
                last(moment_rii_pair_generic_scaled(
                    Tsolve,
                    pair_residual,
                    scalar_residual,
                    Xprobe;
                    center=center,
                    radius=radius,
                    nodes=nodes,
                    iterations=iterations,
                    moment_count=moment_count,
                    ranktol=ranktol,
                    maxrank=maxrank,
                    residual_tol=residual_tol,
                    keep=length(expected),
                ))
            catch err
                println("  K=$moment_count nodes=$nodes failed: $(typeof(err)) $err")
                continue
            end
            h0 = first(history)
            hend = last(history)
            @printf(
                "  K=%d nodes=%d iter0(conv=%d/%d max=%.3e pair=%.3e) final(rank=%d conv=%d/%d max=%.3e pair=%.3e)\n",
                moment_count,
                nodes,
                h0.converged_inside,
                h0.inside,
                h0.max_inside_residual,
                h0.pair_residual,
                hend.rank,
                hend.converged_inside,
                hend.inside,
                hend.max_inside_residual,
                hend.pair_residual,
            )
        end
    end
end

function main()
    run_case("diagonal_linear", diagonal_linear_problem; nodes=16, iterations=3, moment_counts=(1, 2), ranktol=1e-12, residual_tol=1e-10)
    run_case("deficient_quadratic", deficient_quadratic_problem; nodes=64, iterations=5, moment_counts=(1, 2, 3), ranktol=1e-9, residual_tol=1e-8)
    run_case("butterfly", butterfly_problem; nodes=32, iterations=3, moment_counts=(1, 2), ranktol=1e-9, residual_tol=1e-8, keep_extra=3)
    run_node_sweep("deficient_quadratic", deficient_quadratic_problem; node_values=(8, 12, 16, 24, 32), iterations=5, moment_count=2, ranktol=1e-9, residual_tol=1e-8)
    run_node_sweep("butterfly", butterfly_problem; node_values=(8, 12, 16, 24, 32), iterations=5, moment_count=2, ranktol=1e-9, residual_tol=1e-8, keep_extra=3)
    run_projected_hankel_case("deficient_quadratic", deficient_quadratic_problem; nodes=16, iterations=5, moment_count=2, ranktol=1e-9, residual_tol=1e-8)
    run_projected_hankel_case("butterfly", butterfly_problem; nodes=32, iterations=3, moment_count=2, ranktol=1e-9, residual_tol=1e-8, keep_extra=3)
    run_linear_ss_feast_control("many_eigenvalue_diagonal", many_eigenvalue_diagonal_matrix; nodes=32, iterations=4, moment_count=3, ranktol=1e-11, residual_tol=1e-10, target_count=true)
    run_linear_ss_feast_control("grcar_nonnormal", grcar_linear_matrix; nodes=48, iterations=5, moment_count=2, ranktol=1e-10, residual_tol=1e-8)
    run_nonlinear_update_comparison("deficient_quadratic", deficient_quadratic_problem; nodes=16, iterations=5, moment_count=2, ranktol=1e-9, residual_tol=1e-8)
    run_nonlinear_update_comparison("butterfly", butterfly_problem; nodes=32, iterations=3, moment_count=2, ranktol=1e-9, residual_tol=1e-8, keep_extra=3)
    run_nonlinear_update_comparison("butterfly_target", butterfly_problem; nodes=32, iterations=3, moment_count=2, ranktol=1e-9, residual_tol=1e-8, target_count=true)
    run_nonlinear_update_comparison(
        "many_eigenvalue_nonnormal_polynomial_K4_capacity_failure",
        many_eigenvalue_nonnormal_polynomial_problem;
        nodes=32,
        iterations=6,
        moment_count=4,
        ranktol=1e-10,
        residual_tol=1e-8,
        target_count=true,
        modes=(:projected, :shifted),
    )
    run_nonlinear_update_comparison(
        "many_eigenvalue_nonnormal_polynomial_low_nodes",
        many_eigenvalue_nonnormal_polynomial_problem;
        nodes=8,
        iterations=8,
        moment_count=5,
        ranktol=1e-10,
        residual_tol=1e-8,
        target_count=true,
        modes=(:projected, :shifted),
    )
    run_nonlinear_update_comparison(
        "many_eigenvalue_nonnormal_polynomial",
        many_eigenvalue_nonnormal_polynomial_problem;
        nodes=16,
        iterations=6,
        moment_count=5,
        ranktol=1e-10,
        residual_tol=1e-8,
        target_count=true,
        modes=(:projected, :shifted),
    )
    run_scalar_sine_sweep()
    run_scalar_sine_scaled_sweep()
    run_scalar_sine_realization_sweep()
    run_scalar_sine_realization_sweep(radius=20.0, node_values=(128,), iterations=8, moment_counts=(13,), ranktol=1e-10, residual_tol=1e-8)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
