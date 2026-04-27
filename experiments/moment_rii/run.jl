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

function diagonal_balance_pair(X, S; sweeps=20)
    Xbalanced = copy(X)
    Sbalanced = copy(S)
    p = size(Sbalanced, 1)
    for _ in 1:sweeps
        changed = false
        for i in 1:p
            row_norm = norm(Sbalanced[i, setdiff(1:p, i)])
            col_norm = norm(Sbalanced[setdiff(1:p, i), i])
            (row_norm == 0 || col_norm == 0) && continue
            factor = sqrt(row_norm / col_norm)
            isfinite(factor) || continue
            factor = clamp(factor, 1e-4, 1e4)
            (factor < 0.95 || factor > 1.05) || continue
            Sbalanced[:, i] .*= factor
            Sbalanced[i, :] ./= factor
            Xbalanced[:, i] .*= factor
            changed = true
        end
        changed || break
    end
    Xbalanced, Sbalanced
end

function schur_gauge_pair(X, S)
    F = schur(S)
    X * F.Z, Matrix(F.T)
end

function observable_eigen_gauge_pair(X, S; min_norm=1e-14, max_scale=1e14)
    F = eigen(S)
    G = Matrix(F.vectors)
    Xg = X * G
    for j in axes(Xg, 2)
        col_norm = norm(view(Xg, :, j))
        isfinite(col_norm) || continue
        col_norm > min_norm || continue
        scale = clamp(inv(col_norm), inv(max_scale), max_scale)
        Xg[:, j] .*= scale
    end
    Xg, Diagonal(ComplexF64.(F.values)) |> Matrix
end

function apply_pair_gauge(X, S, gauge)
    if gauge === :none
        return X, S
    elseif gauge === :diagonal_balance
        return diagonal_balance_pair(X, S)
    elseif gauge === :schur
        return schur_gauge_pair(X, S)
    elseif gauge === :observable_eigen
        return observable_eigen_gauge_pair(X, S)
    elseif gauge === :diagonal_balance_observable_eigen
        Xb, Sb = diagonal_balance_pair(X, S)
        return observable_eigen_gauge_pair(Xb, Sb)
    elseif gauge === :schur_diagonal_balance
        Xs, Ss = schur_gauge_pair(X, S)
        return diagonal_balance_pair(Xs, Ss)
    else
        error("unknown pair gauge: $gauge")
    end
end

function small_operator_conditioning(S)
    isempty(S) && return (eigcond=NaN, condS=NaN)
    eigcond = try
        size(S, 1) == 1 ? 1.0 : cond(eigen(S).vectors)
    catch
        Inf
    end
    condS = try
        cond(S)
    catch
        Inf
    end
    (eigcond=eigcond, condS=condS)
end

function nearest_anchor_score(value, anchors, radius)
    isempty(anchors) && return 0.0
    minimum(abs.(value .- anchors)) / radius
end

function pair_selection(
    values,
    center,
    radius,
    keep;
    target_count=false,
    policy=:contour,
    residual_scores=nothing,
    anchors=ComplexF64[],
)
    inside = FEASTSolver.in_contour(values, center, radius)
    keep_count = if target_count && keep > 0
        min(length(values), keep)
    else
        min(length(values), max(count(inside), keep))
    end
    keep_count == 0 && return BitVector(falses(length(values)))
    scores = map(eachindex(values)) do i
        distance = abs(values[i] - center)
        contour_score = inside[i] ? distance / radius : abs(distance / radius - 1)
        residual_score = residual_scores === nothing ? 0.0 : residual_scores[i]
        anchor_score = nearest_anchor_score(values[i], anchors, radius)
        if policy === :contour
            (inside[i] ? 0 : 1, contour_score, distance)
        elseif policy === :residual
            (inside[i] ? 0 : 1, residual_score, contour_score, distance)
        elseif policy === :persistent
            (inside[i] ? 0 : 1, anchor_score, residual_score, contour_score, distance)
        elseif policy === :residual_persistent
            (inside[i] ? 0 : 1, residual_score, anchor_score, contour_score, distance)
        else
            error("unknown pair-selection policy: $policy")
        end
    end
    selected_indices = sortperm(scores)[1:keep_count]
    selected = falses(length(values))
    selected[selected_indices] .= true
    BitVector(selected)
end

function residual_scores_for_schur_values(coeffs, X, S, schur_values, center, radius; residual_tol=1e-8)
    coeffs === nothing && return nothing
    lambdas, residuals, _, _, _ = scalar_diagnostics(coeffs, X, S, center, radius; residual_tol=residual_tol)
    isempty(lambdas) && return fill(Inf, length(schur_values))
    map(schur_values) do value
        residuals[argmin(abs.(lambdas .- value))]
    end
end

function restrict_pair_to_contour(
    X,
    S,
    center,
    radius,
    lift;
    keep=0,
    target_count=false,
    policy=:contour,
    coeffs=nothing,
    anchors=ComplexF64[],
    residual_tol=1e-8,
)
    F = schur(S)
    residual_scores = residual_scores_for_schur_values(coeffs, X, S, F.values, center, radius; residual_tol=residual_tol)
    select = pair_selection(
        F.values,
        center,
        radius,
        keep;
        target_count=target_count,
        policy=policy,
        residual_scores=residual_scores,
        anchors=anchors,
    )
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

function hankel_pair_identity_offset(moments, moment_count, moment_offset; ranktol=1e-10, maxrank=typemax(Int))
    moment_offset == 0 && return hankel_pair_identity(moments, moment_count; ranktol=ranktol, maxrank=maxrank)
    length(moments) >= 2 * moment_count + moment_offset ||
        error("not enough moments for requested Hankel offset")
    n, m = size(moments[1])
    H0 = zeros(ComplexF64, moment_count * n, moment_count * m)
    H1 = similar(H0)
    for i in 1:moment_count, j in 1:moment_count
        rows = (i - 1) * n + 1:i * n
        cols = (j - 1) * m + 1:j * m
        H0[rows, cols] .= moments[i + j - 1 + moment_offset]
        H1[rows, cols] .= moments[i + j + moment_offset]
    end

    F = svd(H0)
    isempty(F.S) && error("empty offset Hankel SVD")
    rank = count(F.S ./ F.S[1] .> ranktol)
    rank = min(rank, maxrank, length(F.S))
    rank > 0 || error("offset Hankel rank is zero; loosen ranktol or change the moment offset")

    U = F.U[:, 1:rank]
    V = F.V[:, 1:rank]
    B = U' * H1 * V * Diagonal(1 ./ F.S[1:rank])
    X = U[1:n, :]
    X, B = normalize_lifted_pair(X, B, moment_count)
    X, B, rank, copy(F.S)
end

function hankel_pair_identity_offsets(moments, moment_count, moment_offsets; ranktol=1e-10, maxrank=typemax(Int))
    offsets = Tuple(moment_offsets)
    length(offsets) == 1 && return hankel_pair_identity_offset(moments, moment_count, first(offsets); ranktol=ranktol, maxrank=maxrank)
    maximum(offsets) >= 0 || error("moment offsets must be nonnegative")
    length(moments) >= 2 * moment_count + maximum(offsets) ||
        error("not enough moments for requested multi-offset Hankel")
    n, m = size(moments[1])
    H0 = zeros(ComplexF64, length(offsets) * moment_count * n, moment_count * m)
    H1 = similar(H0)
    for (block, offset) in pairs(offsets), i in 1:moment_count, j in 1:moment_count
        rows = (block - 1) * moment_count * n + (i - 1) * n + 1:(block - 1) * moment_count * n + i * n
        cols = (j - 1) * m + 1:j * m
        H0[rows, cols] .= moments[i + j - 1 + offset]
        H1[rows, cols] .= moments[i + j + offset]
    end

    F = svd(H0)
    isempty(F.S) && error("empty multi-offset Hankel SVD")
    rank = count(F.S ./ F.S[1] .> ranktol)
    rank = min(rank, maxrank, length(F.S))
    rank > 0 || error("multi-offset Hankel rank is zero; loosen ranktol or change offsets")

    U = F.U[:, 1:rank]
    V = F.V[:, 1:rank]
    invσ = Diagonal(1 ./ F.S[1:rank])
    B = U' * H1 * V * invσ
    right_moments = zeros(ComplexF64, n, moment_count * m)
    for j in 1:moment_count
        right_moments[:, (j - 1) * m + 1:j * m] .= moments[j]
    end
    X = right_moments * V * invσ
    X, B = normalize_lifted_pair(X, B, moment_count)
    X, B, rank, copy(F.S)
end

function balanced_hankel_pair_identity_offset(moments, moment_count, moment_offset; ranktol=1e-10, maxrank=typemax(Int))
    length(moments) >= 2 * moment_count + moment_offset ||
        error("not enough moments for requested balanced Hankel offset")
    n, m = size(moments[1])
    H0 = zeros(ComplexF64, moment_count * n, moment_count * m)
    H1 = similar(H0)
    right_moments = zeros(ComplexF64, n, moment_count * m)
    for j in 1:moment_count
        right_moments[:, (j - 1) * m + 1:j * m] .= moments[j + moment_offset]
    end
    for i in 1:moment_count, j in 1:moment_count
        rows = (i - 1) * n + 1:i * n
        cols = (j - 1) * m + 1:j * m
        H0[rows, cols] .= moments[i + j - 1 + moment_offset]
        H1[rows, cols] .= moments[i + j + moment_offset]
    end

    F = svd(H0)
    isempty(F.S) && error("empty balanced Hankel SVD")
    rank = count(F.S ./ F.S[1] .> ranktol)
    rank = min(rank, maxrank, length(F.S))
    rank > 0 || error("balanced Hankel rank is zero; loosen ranktol or change offsets")

    U = F.U[:, 1:rank]
    V = F.V[:, 1:rank]
    sqrtσ = Diagonal(sqrt.(F.S[1:rank]))
    invsqrtσ = Diagonal(1 ./ sqrt.(F.S[1:rank]))
    B = invsqrtσ * (U' * H1 * V) * invsqrtσ
    X = right_moments * V * invsqrtσ
    X, B = normalize_lifted_pair(X, B, moment_count)
    X, B, rank, copy(F.S)
end

function balanced_hankel_pair_identity_offsets(moments, moment_count, moment_offsets; ranktol=1e-10, maxrank=typemax(Int))
    offsets = Tuple(moment_offsets)
    length(offsets) == 1 &&
        return balanced_hankel_pair_identity_offset(moments, moment_count, first(offsets); ranktol=ranktol, maxrank=maxrank)
    maximum(offsets) >= 0 || error("moment offsets must be nonnegative")
    length(moments) >= 2 * moment_count + maximum(offsets) ||
        error("not enough moments for requested multi-offset balanced Hankel")
    n, m = size(moments[1])
    H0 = zeros(ComplexF64, length(offsets) * moment_count * n, moment_count * m)
    H1 = similar(H0)
    for (block, offset) in pairs(offsets), i in 1:moment_count, j in 1:moment_count
        rows = (block - 1) * moment_count * n + (i - 1) * n + 1:(block - 1) * moment_count * n + i * n
        cols = (j - 1) * m + 1:j * m
        H0[rows, cols] .= moments[i + j - 1 + offset]
        H1[rows, cols] .= moments[i + j + offset]
    end

    F = svd(H0)
    isempty(F.S) && error("empty multi-offset balanced Hankel SVD")
    rank = count(F.S ./ F.S[1] .> ranktol)
    rank = min(rank, maxrank, length(F.S))
    rank > 0 || error("multi-offset balanced Hankel rank is zero; loosen ranktol or change offsets")

    U = F.U[:, 1:rank]
    V = F.V[:, 1:rank]
    sqrtσ = Diagonal(sqrt.(F.S[1:rank]))
    invsqrtσ = Diagonal(1 ./ sqrt.(F.S[1:rank]))
    B = invsqrtσ * (U' * H1 * V) * invsqrtσ
    right_moments = zeros(ComplexF64, n, moment_count * m)
    for j in 1:moment_count
        right_moments[:, (j - 1) * m + 1:j * m] .= moments[j]
    end
    X = right_moments * V * invsqrtσ
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

function projected_hankel_pair_identity_offset(moments, left_probe, moment_count, moment_offset; ranktol=1e-10, maxrank=typemax(Int))
    moment_offset == 0 &&
        return projected_hankel_pair_identity(moments, left_probe, moment_count; ranktol=ranktol, maxrank=maxrank)
    length(moments) >= 2 * moment_count + moment_offset ||
        error("not enough moments for requested projected Hankel offset")
    n, m = size(moments[1])
    size(left_probe, 1) == n || error("left_probe row count must match moment row count")
    ell = size(left_probe, 2)
    H0 = zeros(ComplexF64, moment_count * ell, moment_count * m)
    H1 = similar(H0)
    right_moments = zeros(ComplexF64, n, moment_count * m)
    for j in 1:moment_count
        right_moments[:, (j - 1) * m + 1:j * m] .= moments[j + moment_offset]
    end

    for i in 1:moment_count, j in 1:moment_count
        rows = (i - 1) * ell + 1:i * ell
        cols = (j - 1) * m + 1:j * m
        H0[rows, cols] .= left_probe' * moments[i + j - 1 + moment_offset]
        H1[rows, cols] .= left_probe' * moments[i + j + moment_offset]
    end

    F = svd(H0)
    isempty(F.S) && error("empty offset projected Hankel SVD")
    rank = count(F.S ./ F.S[1] .> ranktol)
    rank = min(rank, maxrank, length(F.S))
    rank > 0 || error("offset projected Hankel rank is zero; loosen ranktol or change probes/offset")

    U = F.U[:, 1:rank]
    V = F.V[:, 1:rank]
    invσ = Diagonal(1 ./ F.S[1:rank])
    B = U' * H1 * V * invσ
    X = right_moments * V * invσ
    X, B = normalize_lifted_pair(X, B, moment_count)
    X, B, rank, copy(F.S)
end

function projected_hankel_pair_identity_offsets(moments, left_probe, moment_count, moment_offsets; ranktol=1e-10, maxrank=typemax(Int))
    offsets = Tuple(moment_offsets)
    length(offsets) == 1 &&
        return projected_hankel_pair_identity_offset(moments, left_probe, moment_count, first(offsets); ranktol=ranktol, maxrank=maxrank)
    maximum(offsets) >= 0 || error("moment offsets must be nonnegative")
    length(moments) >= 2 * moment_count + maximum(offsets) ||
        error("not enough moments for requested multi-offset projected Hankel")
    n, m = size(moments[1])
    size(left_probe, 1) == n || error("left_probe row count must match moment row count")
    ell = size(left_probe, 2)
    H0 = zeros(ComplexF64, length(offsets) * moment_count * ell, moment_count * m)
    H1 = similar(H0)
    for (block, offset) in pairs(offsets), i in 1:moment_count, j in 1:moment_count
        rows = (block - 1) * moment_count * ell + (i - 1) * ell + 1:(block - 1) * moment_count * ell + i * ell
        cols = (j - 1) * m + 1:j * m
        H0[rows, cols] .= left_probe' * moments[i + j - 1 + offset]
        H1[rows, cols] .= left_probe' * moments[i + j + offset]
    end

    F = svd(H0)
    isempty(F.S) && error("empty multi-offset projected Hankel SVD")
    rank = count(F.S ./ F.S[1] .> ranktol)
    rank = min(rank, maxrank, length(F.S))
    rank > 0 || error("multi-offset projected Hankel rank is zero; loosen ranktol or change probes/offsets")

    U = F.U[:, 1:rank]
    V = F.V[:, 1:rank]
    invσ = Diagonal(1 ./ F.S[1:rank])
    B = U' * H1 * V * invσ
    right_moments = zeros(ComplexF64, n, moment_count * m)
    for j in 1:moment_count
        right_moments[:, (j - 1) * m + 1:j * m] .= moments[j]
    end
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

function shifted_pair_identity_offset(moments, lift, moment_offset; ranktol=1e-10, maxrank=typemax(Int))
    moment_offset == 0 && return shifted_pair_identity(moments, lift; ranktol=ranktol, maxrank=maxrank)
    length(moments) >= lift + 1 + moment_offset || error("shifted realization needs more moments for requested offset")
    n, p = size(moments[1])
    L0 = zeros(ComplexF64, lift * n, p)
    L1 = similar(L0)
    for j in 1:lift
        rows = (j - 1) * n + 1:j * n
        L0[rows, :] .= moments[j + moment_offset]
        L1[rows, :] .= moments[j + 1 + moment_offset]
    end

    F = svd(L0)
    isempty(F.S) && error("empty offset shifted moment SVD")
    rank = count(F.S ./ F.S[1] .> ranktol)
    rank = min(rank, maxrank, p, length(F.S))
    rank > 0 || error("offset shifted moment rank is zero; loosen ranktol or change the moment offset")

    if rank < p
        V = F.V[:, 1:rank]
        X = moments[1 + moment_offset] * V
        S = (L0 * V) \ (L1 * V)
    else
        X = moments[1 + moment_offset]
        S = L0 \ L1
    end

    X, S = normalize_lifted_pair(X, S, lift)
    X, S, rank, copy(F.S)
end

function shifted_pair_identity_offsets(moments, lift, moment_offsets; ranktol=1e-10, maxrank=typemax(Int))
    offsets = Tuple(moment_offsets)
    length(offsets) == 1 && return shifted_pair_identity_offset(moments, lift, first(offsets); ranktol=ranktol, maxrank=maxrank)
    length(moments) >= lift + 1 + maximum(offsets) || error("shifted realization needs more moments for requested offsets")
    n, p = size(moments[1])
    L0 = zeros(ComplexF64, length(offsets) * lift * n, p)
    L1 = similar(L0)
    for (block, offset) in pairs(offsets), j in 1:lift
        rows = (block - 1) * lift * n + (j - 1) * n + 1:(block - 1) * lift * n + j * n
        L0[rows, :] .= moments[j + offset]
        L1[rows, :] .= moments[j + 1 + offset]
    end

    F = svd(L0)
    isempty(F.S) && error("empty multi-offset shifted moment SVD")
    rank = count(F.S ./ F.S[1] .> ranktol)
    rank = min(rank, maxrank, p, length(F.S))
    rank > 0 || error("multi-offset shifted moment rank is zero; loosen ranktol or change offsets")

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

function balanced_shifted_pair_identity(moments, lift; ranktol=1e-10, maxrank=typemax(Int))
    length(moments) >= lift + 1 || error("balanced shifted realization needs at least lift+1 moments")
    n, p = size(moments[1])
    L0 = zeros(ComplexF64, lift * n, p)
    L1 = similar(L0)
    for j in 1:lift
        rows = (j - 1) * n + 1:j * n
        L0[rows, :] .= moments[j]
        L1[rows, :] .= moments[j + 1]
    end

    F = svd(L0)
    isempty(F.S) && error("empty balanced shifted moment SVD")
    rank = count(F.S ./ F.S[1] .> ranktol)
    rank = min(rank, maxrank, length(F.S))
    rank > 0 || error("balanced shifted moment rank is zero; loosen ranktol or change the contour/probe")

    U = F.U[:, 1:rank]
    V = F.V[:, 1:rank]
    σ = F.S[1:rank]
    sqrtσ = Diagonal(sqrt.(σ))
    invsqrtσ = Diagonal(1 ./ sqrt.(σ))

    # Ho-Kalman/balanced realization of L0 = observability * controllability.
    # This fixes the realization gauge using Hankel singular directions instead
    # of scalar Ritz-value pruning.
    S = invsqrtσ * (U' * L1 * V) * invsqrtσ
    X = U[1:n, :] * sqrtσ
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
    step = try
        J \ rhs
    catch err
        err isa SingularException || rethrow()
        pinv(J; rtol=1e-10) * rhs
    end
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

function generic_lifted_newton_step(pair_residual, X, S; lift=1, step_size=1e-7)
    n, p = size(X)
    R = pair_residual(X, S)
    L = lifted_pair_matrix(X, S, lift)
    unknowns = n * p + p * p
    equations = n * p + p * p
    J = zeros(ComplexF64, equations, unknowns)

    column = 1
    for col in 1:p, row in 1:n
        ΔX = zeros(ComplexF64, n, p)
        ΔS = zeros(ComplexF64, p, p)
        ΔX[row, col] = 1
        Rplus = pair_residual(X .+ step_size .* ΔX, S)
        Lplus = lifted_pair_matrix(X .+ step_size .* ΔX, S, lift)
        J[1:n*p, column] .= vec((Rplus .- R) ./ step_size)
        J[n*p+1:end, column] .= vec(L' * ((Lplus .- L) ./ step_size))
        column += 1
    end
    for col in 1:p, row in 1:p
        ΔX = zeros(ComplexF64, n, p)
        ΔS = zeros(ComplexF64, p, p)
        ΔS[row, col] = 1
        Rplus = pair_residual(X, S .+ step_size .* ΔS)
        Lplus = lifted_pair_matrix(X, S .+ step_size .* ΔS, lift)
        J[1:n*p, column] .= vec((Rplus .- R) ./ step_size)
        J[n*p+1:end, column] .= vec(L' * ((Lplus .- L) ./ step_size))
        column += 1
    end

    rhs = vcat(-vec(R), zeros(ComplexF64, p * p))
    step = try
        J \ rhs
    catch err
        err isa SingularException || rethrow()
        pinv(J; rtol=1e-10) * rhs
    end
    ΔX = reshape(step[1:n*p], n, p)
    ΔS = reshape(step[n*p+1:end], p, p)
    current = norm(R)
    best_X, best_S = X, S
    best_residual = current
    α = 1.0
    for _ in 1:10
        candidate_X, candidate_S = normalize_lifted_pair(X .+ α .* ΔX, S .+ α .* ΔS, lift)
        candidate_residual = norm(pair_residual(candidate_X, candidate_S))
        if candidate_residual < best_residual
            best_X, best_S = candidate_X, candidate_S
            best_residual = candidate_residual
            break
        end
        α /= 2
    end
    best_X, best_S, best_residual / max(current, eps(Float64))
end

function generic_lifted_newton_refine(pair_residual, X, S; lift=1, steps=1)
    ratios = Float64[]
    for _ in 1:steps
        Xnew, Snew, ratio = generic_lifted_newton_step(pair_residual, X, S; lift=lift)
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
    retention_policy=:contour,
    gauge=:none,
)
    update_mode in (:projected, :shifted, :balanced_shifted, :projected_newton) ||
        error("update_mode must be :projected, :shifted, :balanced_shifted, or :projected_newton")

    z_nodes, z_weights = circular_rule(center, radius, nodes)
    W = orthonormal_probe(left_probe)
    anchors = ComplexF64[]
    moments = initial_polynomial_moments_scaled(coeffs, Xprobe, z_nodes, z_weights, center, radius, moment_count)
    X, Sμ, _, singular_values = projected_hankel_pair_identity(moments, W, moment_count; ranktol=ranktol, maxrank=maxrank)
    X, Sμ = restrict_scaled_pair_to_contour(
        X,
        Sμ,
        center,
        radius,
        moment_count;
        keep=keep,
        target_count=target_count,
        policy=retention_policy,
        coeffs=coeffs,
        anchors=anchors,
        residual_tol=residual_tol,
    )
    Sλ = lambda_matrix_from_scaled(Sμ, center, radius)
    update_mode === :projected_newton && ((X, Sλ, _) = invariant_pair_newton_refine(coeffs, X, Sλ; lift=moment_count, steps=newton_steps))
    X, Sλ = apply_pair_gauge(X, Sλ, gauge)
    Sμ = scaled_matrix_from_lambda(Sλ, center, radius)
    anchors = retained_anchor_values(Sλ, center, radius)

    history = PairHistory[]
    record_history!(history, 0, coeffs, X, Sλ, singular_values, center, radius; residual_tol=residual_tol)

    for iteration in 1:iterations
        Sλ = lambda_matrix_from_scaled(Sμ, center, radius)
        moments = rii_polynomial_moments_scaled(coeffs, X, Sλ, z_nodes, z_weights, center, radius, moment_count)
        if update_mode === :shifted
            X, Sμ, _, singular_values = shifted_pair_identity(moments, moment_count; ranktol=ranktol, maxrank=maxrank)
        elseif update_mode === :balanced_shifted
            X, Sμ, _, singular_values = balanced_shifted_pair_identity(moments, moment_count; ranktol=ranktol, maxrank=maxrank)
        else
            X, Sμ, _, singular_values = projected_hankel_pair_identity(moments, W, moment_count; ranktol=ranktol, maxrank=maxrank)
        end
        X, Sμ = restrict_scaled_pair_to_contour(
            X,
            Sμ,
            center,
            radius,
            moment_count;
            keep=keep,
            target_count=target_count,
            policy=retention_policy,
            coeffs=coeffs,
            anchors=anchors,
            residual_tol=residual_tol,
        )
        Sλ = lambda_matrix_from_scaled(Sμ, center, radius)
        if update_mode === :projected_newton
            X, Sλ, _ = invariant_pair_newton_refine(coeffs, X, Sλ; lift=moment_count, steps=newton_steps)
        end
        X, Sλ = apply_pair_gauge(X, Sλ, gauge)
        Sμ = scaled_matrix_from_lambda(Sλ, center, radius)
        anchors = retained_anchor_values(Sλ, center, radius)
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

function contour_coordinate(z, center, radius, coordinate_shift, coordinate_kind)
    μ = (z - center) / radius + coordinate_shift
    coordinate_kind === :direct && return μ
    coordinate_kind === :inverse && return inv(μ)
    if coordinate_kind === :mobius
        a = coordinate_shift
        abs(a) < 1 || error("Möbius coordinate shift must be inside the unit disk")
        raw = (z - center) / radius
        return (raw - a) / (1 - conj(a) * raw)
    end
    error("unknown coordinate kind: $coordinate_kind")
end

function initial_moments_generic_scaled(Tsolve, Xprobe, z_nodes, z_weights, center, radius, moment_count; coordinate_shift=0.0, coordinate_kind=:direct, moment_offset=0)
    n, m = size(Xprobe)
    moments = [zeros(ComplexF64, n, m) for _ in 1:(2 * moment_count + moment_offset)]
    for (z, weight) in zip(z_nodes, z_weights)
        solved = Tsolve(z, Xprobe)
        μ = contour_coordinate(z, center, radius, coordinate_shift, coordinate_kind)
        μpower = one(ComplexF64)
        for p in eachindex(moments)
            moments[p] .+= (weight * μpower) .* solved
            μpower *= μ
        end
    end
    moments
end

function rii_moments_generic_scaled(Tsolve, pair_residual, X, Sλ, z_nodes, z_weights, center, radius, moment_count; coordinate_shift=0.0, coordinate_kind=:direct, moment_offset=0)
    n, p = size(X)
    moments = [zeros(ComplexF64, n, p) for _ in 1:(2 * moment_count + moment_offset)]
    I_p = Matrix{ComplexF64}(I, p, p)
    R = pair_residual(X, Sλ)
    for (z, weight) in zip(z_nodes, z_weights)
        corrected = X - Tsolve(z, R)
        corrected = corrected / (z * I_p - Sλ)
        μ = contour_coordinate(z, center, radius, coordinate_shift, coordinate_kind)
        μpower = one(ComplexF64)
        for q in eachindex(moments)
            moments[q] .+= (weight * μpower) .* corrected
            μpower *= μ
        end
    end
    moments
end

function rii_chebyshev_moments_generic_scaled(Tsolve, pair_residual, X, Sλ, z_nodes, z_weights, center, radius, moment_count; coordinate_shift=0.0, coordinate_kind=:direct)
    n, p = size(X)
    moments = [zeros(ComplexF64, n, p) for _ in 1:(moment_count + 1)]
    I_p = Matrix{ComplexF64}(I, p, p)
    R = pair_residual(X, Sλ)
    for (z, weight) in zip(z_nodes, z_weights)
        corrected = X - Tsolve(z, R)
        corrected = corrected / (z * I_p - Sλ)
        μ = contour_coordinate(z, center, radius, coordinate_shift, coordinate_kind)
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
scaled_matrix_from_lambda(Sλ, center, radius) = (Sλ .- center .* Matrix{ComplexF64}(I, size(Sλ, 1), size(Sλ, 2))) ./ radius
lambda_matrix_from_scaled(Sμ, center, radius, coordinate_shift) = lambda_matrix_from_scaled(Sμ, center, radius, coordinate_shift, :direct)
scaled_matrix_from_lambda(Sλ, center, radius, coordinate_shift) = scaled_matrix_from_lambda(Sλ, center, radius, coordinate_shift, :direct)

function lambda_matrix_from_scaled(Scoord, center, radius, coordinate_shift, coordinate_kind)
    I_s = Matrix{ComplexF64}(I, size(Scoord, 1), size(Scoord, 2))
    if coordinate_kind === :direct
        return center * I_s + radius * (Scoord .- coordinate_shift .* I_s)
    elseif coordinate_kind === :inverse
        return center * I_s + radius * (inv(Scoord) .- coordinate_shift .* I_s)
    elseif coordinate_kind === :mobius
        a = coordinate_shift
        abs(a) < 1 || error("Möbius coordinate shift must be inside the unit disk")
        μ = (Scoord .+ a .* I_s) / (I_s .+ conj(a) .* Scoord)
        return center * I_s + radius * μ
    end
    error("unknown coordinate kind: $coordinate_kind")
end

function scaled_matrix_from_lambda(Sλ, center, radius, coordinate_shift, coordinate_kind)
    I_s = Matrix{ComplexF64}(I, size(Sλ, 1), size(Sλ, 2))
    raw = (Sλ .- center .* I_s) ./ radius
    if coordinate_kind === :direct
        return raw .+ coordinate_shift .* I_s
    elseif coordinate_kind === :inverse
        return inv(raw .+ coordinate_shift .* I_s)
    elseif coordinate_kind === :mobius
        a = coordinate_shift
        abs(a) < 1 || error("Möbius coordinate shift must be inside the unit disk")
        return (raw .- a .* I_s) / (I_s .- conj(a) .* raw)
    end
    error("unknown coordinate kind: $coordinate_kind")
end

function restrict_scaled_pair_to_contour(
    X,
    Sμ,
    center,
    radius,
    lift;
    keep=0,
    target_count=false,
    policy=:contour,
    coeffs=nothing,
    anchors=ComplexF64[],
    residual_tol=1e-8,
    coordinate_shift=0.0,
    coordinate_kind=:direct,
)
    Sλ = lambda_matrix_from_scaled(Sμ, center, radius, coordinate_shift, coordinate_kind)
    Xrestricted, Sλrestricted = restrict_pair_to_contour(
        X,
        Sλ,
        center,
        radius,
        lift;
        keep=keep,
        target_count=target_count,
        policy=policy,
        coeffs=coeffs,
        anchors=anchors,
        residual_tol=residual_tol,
    )
    Sμrestricted = scaled_matrix_from_lambda(Sλrestricted, center, radius, coordinate_shift, coordinate_kind)
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

function moment_rii_pair_generic_scaled(Tsolve, pair_residual, scalar_residual, Xprobe; center, radius, nodes=64, iterations=4, moment_count=2, ranktol=1e-10, maxrank=typemax(Int), residual_tol=1e-8, keep=0, extraction=:hankel, coordinate_shift=0.0, coordinate_kind=:direct, moment_offset=0)
    extraction in (
        :hankel,
        :hankel_gauge_balanced,
        :hankel_balanced,
        :hankel_observable_eigen,
        :hankel_balanced_observable_eigen,
        :shifted,
        :shifted_gauge_balanced,
        :shifted_observable_eigen,
        :shifted_gauge_observable_eigen,
        :shifted_schur,
        :shifted_schur_balanced,
        :balanced_shifted,
        :balanced_shifted_gauge_balanced,
        :chebyshev_shifted,
        :chebyshev_shifted_gauge_balanced,
        :chebyshev_shifted_observable_eigen,
        :chebyshev_shifted_schur,
    ) ||
        error("unknown extraction: $extraction")
    gauge = if extraction in (:hankel_gauge_balanced, :shifted_gauge_balanced, :balanced_shifted_gauge_balanced, :chebyshev_shifted_gauge_balanced)
        :diagonal_balance
    elseif extraction in (:hankel_observable_eigen, :hankel_balanced_observable_eigen, :shifted_observable_eigen, :chebyshev_shifted_observable_eigen)
        :observable_eigen
    elseif extraction === :shifted_gauge_observable_eigen
        :diagonal_balance_observable_eigen
    elseif extraction in (:shifted_schur, :chebyshev_shifted_schur)
        :schur
    elseif extraction === :shifted_schur_balanced
        :schur_diagonal_balance
    else
        :none
    end
    moment_offsets = moment_offset isa Integer ? (moment_offset,) : Tuple(moment_offset)
    max_moment_offset = maximum(moment_offsets)
    z_nodes, z_weights = circular_rule(center, radius, nodes)
    moments = initial_moments_generic_scaled(Tsolve, Xprobe, z_nodes, z_weights, center, radius, moment_count; coordinate_shift=coordinate_shift, coordinate_kind=coordinate_kind, moment_offset=max_moment_offset)
    if extraction in (:hankel_balanced, :hankel_balanced_observable_eigen)
        X, Sμ, _, singular_values = balanced_hankel_pair_identity_offsets(moments, moment_count, moment_offsets; ranktol=ranktol, maxrank=maxrank)
    else
        X, Sμ, _, singular_values = hankel_pair_identity_offsets(moments, moment_count, moment_offsets; ranktol=ranktol, maxrank=maxrank)
    end
    X, Sμ = restrict_scaled_pair_to_contour(X, Sμ, center, radius, moment_count; keep=keep, coordinate_shift=coordinate_shift, coordinate_kind=coordinate_kind)
    Sλ = lambda_matrix_from_scaled(Sμ, center, radius, coordinate_shift, coordinate_kind)
    X, Sλ = apply_pair_gauge(X, Sλ, gauge)
    Sμ = scaled_matrix_from_lambda(Sλ, center, radius, coordinate_shift, coordinate_kind)
    history = PairHistory[]
    lambdas, residuals, inside, converged_inside, max_inside, pair_res =
        generic_scalar_diagnostics(pair_residual, scalar_residual, X, Sλ, center, radius; residual_tol=residual_tol)
    push!(history, PairHistory(0, size(X, 2), inside, converged_inside, pair_res, max_inside, lambdas, residuals, singular_values))

    for iteration in 1:iterations
        Sλ = lambda_matrix_from_scaled(Sμ, center, radius, coordinate_shift, coordinate_kind)
        if extraction in (:hankel, :hankel_gauge_balanced, :hankel_observable_eigen)
            moments = rii_moments_generic_scaled(Tsolve, pair_residual, X, Sλ, z_nodes, z_weights, center, radius, moment_count; coordinate_shift=coordinate_shift, coordinate_kind=coordinate_kind, moment_offset=max_moment_offset)
            X, Sμ, _, singular_values = hankel_pair_identity_offsets(moments, moment_count, moment_offsets; ranktol=ranktol, maxrank=maxrank)
        elseif extraction in (:hankel_balanced, :hankel_balanced_observable_eigen)
            moments = rii_moments_generic_scaled(Tsolve, pair_residual, X, Sλ, z_nodes, z_weights, center, radius, moment_count; coordinate_shift=coordinate_shift, coordinate_kind=coordinate_kind, moment_offset=max_moment_offset)
            X, Sμ, _, singular_values = balanced_hankel_pair_identity_offsets(moments, moment_count, moment_offsets; ranktol=ranktol, maxrank=maxrank)
        elseif extraction in (:shifted, :shifted_gauge_balanced, :shifted_observable_eigen, :shifted_gauge_observable_eigen, :shifted_schur, :shifted_schur_balanced)
            moments = rii_moments_generic_scaled(Tsolve, pair_residual, X, Sλ, z_nodes, z_weights, center, radius, moment_count; coordinate_shift=coordinate_shift, coordinate_kind=coordinate_kind, moment_offset=max_moment_offset)
            X, Sμ, _, singular_values = shifted_pair_identity_offsets(moments, moment_count, moment_offsets; ranktol=ranktol, maxrank=maxrank)
        elseif extraction in (:balanced_shifted, :balanced_shifted_gauge_balanced)
            moments = rii_moments_generic_scaled(Tsolve, pair_residual, X, Sλ, z_nodes, z_weights, center, radius, moment_count; coordinate_shift=coordinate_shift, coordinate_kind=coordinate_kind, moment_offset=max_moment_offset)
            X, Sμ, _, singular_values = balanced_shifted_pair_identity(moments, moment_count; ranktol=ranktol, maxrank=maxrank)
        else
            moments = rii_chebyshev_moments_generic_scaled(Tsolve, pair_residual, X, Sλ, z_nodes, z_weights, center, radius, moment_count; coordinate_shift=coordinate_shift, coordinate_kind=coordinate_kind)
            X, Sμ, _, singular_values = chebyshev_shifted_pair_identity(moments, moment_count; ranktol=ranktol, maxrank=maxrank)
        end
        X, Sμ = restrict_scaled_pair_to_contour(X, Sμ, center, radius, moment_count; keep=keep, coordinate_shift=coordinate_shift, coordinate_kind=coordinate_kind)
        Sλ = lambda_matrix_from_scaled(Sμ, center, radius, coordinate_shift, coordinate_kind)
        X, Sλ = apply_pair_gauge(X, Sλ, gauge)
        Sμ = scaled_matrix_from_lambda(Sλ, center, radius, coordinate_shift, coordinate_kind)
        lambdas, residuals, inside, converged_inside, max_inside, pair_res =
            generic_scalar_diagnostics(pair_residual, scalar_residual, X, Sλ, center, radius; residual_tol=residual_tol)
        push!(history, PairHistory(iteration, size(X, 2), inside, converged_inside, pair_res, max_inside, lambdas, residuals, singular_values))
    end
    X, lambda_matrix_from_scaled(Sμ, center, radius, coordinate_shift, coordinate_kind), history
end

function moment_rii_pair_generic_projected_scaled(
    Tsolve,
    pair_residual,
    scalar_residual,
    Xprobe,
    left_probe;
    center,
    radius,
    nodes=64,
    iterations=4,
    moment_count=2,
    ranktol=1e-10,
    maxrank=typemax(Int),
    residual_tol=1e-8,
    keep=0,
    extraction=:projected,
    coordinate_shift=0.0,
    coordinate_kind=:direct,
    moment_offset=0,
)
    extraction in (:projected, :projected_gauge_balanced, :projected_newton) ||
        error("unknown projected generic extraction: $extraction")
    gauge = extraction === :projected_gauge_balanced ? :diagonal_balance : :none
    moment_offsets = moment_offset isa Integer ? (moment_offset,) : Tuple(moment_offset)
    max_moment_offset = maximum(moment_offsets)
    z_nodes, z_weights = circular_rule(center, radius, nodes)
    W = orthonormal_probe(left_probe)
    moments = initial_moments_generic_scaled(
        Tsolve,
        Xprobe,
        z_nodes,
        z_weights,
        center,
        radius,
        moment_count;
        coordinate_shift=coordinate_shift,
        coordinate_kind=coordinate_kind,
        moment_offset=max_moment_offset,
    )
    X, Sμ, _, singular_values = projected_hankel_pair_identity_offsets(moments, W, moment_count, moment_offsets; ranktol=ranktol, maxrank=maxrank)
    X, Sμ = restrict_scaled_pair_to_contour(X, Sμ, center, radius, moment_count; keep=keep, coordinate_shift=coordinate_shift, coordinate_kind=coordinate_kind)
    Sλ = lambda_matrix_from_scaled(Sμ, center, radius, coordinate_shift, coordinate_kind)
    extraction === :projected_newton && ((X, Sλ, _) = generic_lifted_newton_refine(pair_residual, X, Sλ; lift=moment_count, steps=1))
    X, Sλ = apply_pair_gauge(X, Sλ, gauge)
    Sμ = scaled_matrix_from_lambda(Sλ, center, radius, coordinate_shift, coordinate_kind)

    history = PairHistory[]
    lambdas, residuals, inside, converged_inside, max_inside, pair_res =
        generic_scalar_diagnostics(pair_residual, scalar_residual, X, Sλ, center, radius; residual_tol=residual_tol)
    push!(history, PairHistory(0, size(X, 2), inside, converged_inside, pair_res, max_inside, lambdas, residuals, singular_values))

    for iteration in 1:iterations
        Sλ = lambda_matrix_from_scaled(Sμ, center, radius, coordinate_shift, coordinate_kind)
        moments = rii_moments_generic_scaled(
            Tsolve,
            pair_residual,
            X,
            Sλ,
            z_nodes,
            z_weights,
            center,
            radius,
            moment_count;
            coordinate_shift=coordinate_shift,
            coordinate_kind=coordinate_kind,
            moment_offset=max_moment_offset,
        )
        X, Sμ, _, singular_values = projected_hankel_pair_identity_offsets(moments, W, moment_count, moment_offsets; ranktol=ranktol, maxrank=maxrank)
        X, Sμ = restrict_scaled_pair_to_contour(X, Sμ, center, radius, moment_count; keep=keep, coordinate_shift=coordinate_shift, coordinate_kind=coordinate_kind)
        Sλ = lambda_matrix_from_scaled(Sμ, center, radius, coordinate_shift, coordinate_kind)
        extraction === :projected_newton && ((X, Sλ, _) = generic_lifted_newton_refine(pair_residual, X, Sλ; lift=moment_count, steps=1))
        X, Sλ = apply_pair_gauge(X, Sλ, gauge)
        Sμ = scaled_matrix_from_lambda(Sλ, center, radius, coordinate_shift, coordinate_kind)
        lambdas, residuals, inside, converged_inside, max_inside, pair_res =
            generic_scalar_diagnostics(pair_residual, scalar_residual, X, Sλ, center, radius; residual_tol=residual_tol)
        push!(history, PairHistory(iteration, size(X, 2), inside, converged_inside, pair_res, max_inside, lambdas, residuals, singular_values))
    end
    X, lambda_matrix_from_scaled(Sμ, center, radius, coordinate_shift, coordinate_kind), history
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

    for extraction in (:hankel, :shifted, :shifted_gauge_balanced, :shifted_schur, :balanced_shifted, :chebyshev_shifted, :chebyshev_shifted_gauge_balanced)
        for moment_count in moment_counts
            for nodes in node_values
                Xprobe = ones(ComplexF64, 1, 1)
                maxrank = moment_count
                X, S, history = try
                    moment_rii_pair_generic_scaled(
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
                    )
                catch err
                    println("  extraction=$extraction K=$moment_count nodes=$nodes failed: $(typeof(err)) $err")
                    continue
                end
                h0 = first(history)
                hend = last(history)
                conditioning = small_operator_conditioning(S)
                @printf(
                    "  extraction=%s K=%d nodes=%d iter0(conv=%d/%d max=%.3e pair=%.3e) final(rank=%d conv=%d/%d max=%.3e pair=%.3e eigcond=%.3e condS=%.3e)\n",
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
                    conditioning.eigcond,
                    conditioning.condS,
                )
            end
        end
    end
end

function scalar_roots_in_contour(candidates, center, radius)
    roots = ComplexF64.(candidates)
    roots[FEASTSolver.in_contour(roots, center, radius)]
end

function scalar_sine_case()
    (
        name="sin",
        f=z -> sin(z),
        fmat=S -> sin(S),
        roots=(center, radius) -> scalar_roots_in_contour(
            [k * pi for k in -ceil(Int, radius / pi)-2:ceil(Int, radius / pi)+2],
            center,
            radius,
        ),
    )
end

function scalar_cosine_case()
    (
        name="cos",
        f=z -> cos(z),
        fmat=S -> cos(S),
        roots=(center, radius) -> scalar_roots_in_contour(
            [(k + 0.5) * pi for k in -ceil(Int, radius / pi)-3:ceil(Int, radius / pi)+3],
            center,
            radius,
        ),
    )
end

function scalar_shifted_sine_case(alpha=0.3)
    a = asin(alpha)
    (
        name="sin_minus_$alpha",
        f=z -> sin(z) - alpha,
        fmat=S -> sin(S) .- alpha .* Matrix{ComplexF64}(I, size(S, 1), size(S, 2)),
        roots=(center, radius) -> begin
            kmax = ceil(Int, radius / (2pi)) + 3
            candidates = ComplexF64[]
            for k in -kmax:kmax
                push!(candidates, a + 2pi * k)
                push!(candidates, pi - a + 2pi * k)
            end
            scalar_roots_in_contour(candidates, center, radius)
        end,
    )
end

function scalar_expm1_case()
    (
        name="exp_minus_1",
        f=z -> exp(z) - 1,
        fmat=S -> exp(S) .- Matrix{ComplexF64}(I, size(S, 1), size(S, 2)),
        roots=(center, radius) -> scalar_roots_in_contour(
            [2pi * im * k for k in -ceil(Int, radius / (2pi))-2:ceil(Int, radius / (2pi))+2],
            center,
            radius,
        ),
    )
end

function run_scalar_analytic_gauge_stress(;
    cases=(scalar_sine_case(), scalar_cosine_case(), scalar_shifted_sine_case(), scalar_expm1_case()),
    radii=(20.0, 30.0),
    iterations=10,
    node_factor=10,
    ranktol=1e-10,
    residual_tol=1e-8,
    extractions=(:shifted, :shifted_gauge_balanced, :shifted_schur, :chebyshev_shifted, :chebyshev_shifted_gauge_balanced),
    coordinate_shifts=(0.0,),
    coordinate_kinds=(:direct,),
    moment_offsets=(0,),
)
    center = 0.0 + 0.0im
    println()
    println("Scalar analytic gauge stress")
    println("  known-root scalar NEPs; K is set to the exact number of roots inside the circular contour")
    for case in cases
        for radius in radii
            expected = sort(ComplexF64.(case.roots(center, radius)); by=z -> (real(z), imag(z)))
            isempty(expected) && continue
            moment_count = length(expected)
            nodes = max(64, node_factor * moment_count)
            println()
            println("  case=$(case.name) radius=$radius expected_inside=$(length(expected)) K=$moment_count nodes=$nodes")
            Tsolve = (z, B) -> B ./ case.f(z)
            pair_residual = (X, S) -> X * case.fmat(S)
            scalar_residual = (λ, v) -> abs(case.f(λ)) * norm(v) / max(norm(v), eps(Float64))
            for moment_offset in moment_offsets
                for coordinate_kind in coordinate_kinds, coordinate_shift in coordinate_shifts, extraction in extractions
                    X, S, history = try
                        moment_rii_pair_generic_scaled(
                            Tsolve,
                            pair_residual,
                            scalar_residual,
                            ones(ComplexF64, 1, 1);
                            center=center,
                            radius=radius,
                            nodes=nodes,
                            iterations=iterations,
                            moment_count=moment_count,
                            ranktol=ranktol,
                            maxrank=moment_count,
                            residual_tol=residual_tol,
                            keep=length(expected),
                            extraction=extraction,
                            coordinate_shift=coordinate_shift,
                            coordinate_kind=coordinate_kind,
                            moment_offset=moment_offset,
                        )
                    catch err
                        println("    coord=$coordinate_kind offset=$moment_offset shift=$coordinate_shift extraction=$extraction failed: $(typeof(err)) $err")
                        continue
                    end
                    h = last(history)
                    inside = FEASTSolver.in_contour(h.lambdas, center, radius)
                    matched = match_expected_count(h.lambdas[inside], expected; atol=1e-6)
                    conditioning = small_operator_conditioning(S)
                    @printf(
                        "    coord=%s offset=%d shift=%.3g extraction=%s rank=%d conv=%d/%d matched=%d max=%.3e pair=%.3e eigcond=%.3e condS=%.3e\n",
                        string(coordinate_kind),
                        moment_offset,
                        real(coordinate_shift),
                        string(extraction),
                        h.rank,
                        h.converged_inside,
                        h.inside,
                        matched,
                        h.max_inside_residual,
                        h.pair_residual,
                        conditioning.eigcond,
                        conditioning.condS,
                    )
                end
            end
        end
    end
end

function unique_values(values; atol=1e-6)
    unique = ComplexF64[]
    for value in values
        if isempty(unique) || minimum(abs.(value .- unique)) > atol
            push!(unique, value)
        end
    end
    unique
end

function block_hankel_singular_values(moments, moment_count)
    n, m = size(moments[1])
    H0 = zeros(ComplexF64, moment_count * n, moment_count * m)
    for i in 1:moment_count, j in 1:moment_count
        rows = (i - 1) * n + 1:i * n
        cols = (j - 1) * m + 1:j * m
        H0[rows, cols] .= moments[i + j - 1]
    end
    svdvals(H0)
end

function estimate_generic_chart_rank(
    Tsolve,
    Xprobe;
    center,
    radius,
    capacity=24,
    nodes=max(128, 16 * capacity),
    count_ranktol=1e-6,
    coordinate_shift=0.0,
    coordinate_kind=:direct,
)
    z_nodes, z_weights = circular_rule(center, radius, nodes)
    moments = initial_moments_generic_scaled(
        Tsolve,
        Xprobe,
        z_nodes,
        z_weights,
        center,
        radius,
        capacity;
        coordinate_shift=coordinate_shift,
        coordinate_kind=coordinate_kind,
    )
    singular_values = block_hankel_singular_values(moments, capacity)
    rank = isempty(singular_values) ? 0 : count(singular_values ./ singular_values[1] .> count_ranktol)
    (rank=min(rank, capacity), singular_values=singular_values)
end

function solve_scalar_chart(
    case,
    center,
    radius;
    moment_count,
    moment_offset=0,
    iterations=12,
    nodes=max(64, 12 * moment_count),
    ranktol=1e-10,
    residual_tol=1e-8,
    extraction=:shifted_gauge_balanced,
)
    Tsolve = (z, B) -> B ./ case.f(z)
    pair_residual = (X, S) -> X * case.fmat(S)
    scalar_residual = (λ, v) -> abs(case.f(λ)) * norm(v) / max(norm(v), eps(Float64))
    X, S, history = moment_rii_pair_generic_scaled(
        Tsolve,
        pair_residual,
        scalar_residual,
        ones(ComplexF64, 1, 1);
        center=center,
        radius=radius,
        nodes=nodes,
        iterations=iterations,
        moment_count=moment_count,
        ranktol=ranktol,
        maxrank=moment_count,
        residual_tol=residual_tol,
        keep=moment_count,
        extraction=extraction,
        moment_offset=moment_offset,
    )
    h = last(history)
    inside = FEASTSolver.in_contour(h.lambdas, center, radius)
    good = inside .& (h.residuals .<= residual_tol)
    conditioning = small_operator_conditioning(S)
    (
        values=h.lambdas[good],
        all_values=h.lambdas[inside],
        good_count=count(good),
        inside=h.inside,
        rank=h.rank,
        max_inside_residual=h.max_inside_residual,
        pair_residual=h.pair_residual,
        eigcond=conditioning.eigcond,
        condS=conditioning.condS,
        history=history,
    )
end

function estimate_scalar_chart_rank(
    case,
    center,
    radius;
    capacity=24,
    nodes=max(128, 16 * capacity),
    count_ranktol=1e-6,
)
    Tsolve = (z, B) -> B ./ case.f(z)
    estimate_generic_chart_rank(
        Tsolve,
        ones(ComplexF64, 1, 1);
        center=center,
        radius=radius,
        capacity=capacity,
        nodes=nodes,
        count_ranktol=count_ranktol,
    )
end

function rank_at_threshold(singular_values, threshold)
    isempty(singular_values) && return 0
    count(singular_values ./ singular_values[1] .> threshold)
end

function singular_ratio(singular_values, index)
    (isempty(singular_values) || index < 1 || index > length(singular_values)) && return NaN
    singular_values[index] / singular_values[1]
end

function format_rank_sweep(singular_values, thresholds)
    join(
        (
            @sprintf("%.0e=>%d", threshold, rank_at_threshold(singular_values, threshold))
            for threshold in thresholds
        ),
        ", ",
    )
end

function run_scalar_rank_estimation_stress(;
    cases=(scalar_sine_case(), scalar_cosine_case(), scalar_shifted_sine_case(), scalar_expm1_case()),
    radii=(6.0, 10.0, 14.0, 20.0, 30.0),
    thresholds=(1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-10),
    capacity_margin=8,
)
    center = 0.0 + 0.0im
    println()
    println("Scalar rank-estimation stress")
    println("  Hankel singular-value ranks are compared to known roots only for validation")
    for case in cases
        println("  case=$(case.name)")
        for radius in radii
            expected = case.roots(center, radius)
            expected_count = length(expected)
            expected_count == 0 && continue
            capacity = max(16, expected_count + capacity_margin)
            estimate = estimate_scalar_chart_rank(
                case,
                center,
                radius;
                capacity=capacity,
                nodes=max(128, 16 * capacity),
                count_ranktol=minimum(thresholds),
            )
            rank_text = format_rank_sweep(estimate.singular_values, thresholds)
            expected_ratio = singular_ratio(estimate.singular_values, expected_count)
            next_ratio = singular_ratio(estimate.singular_values, expected_count + 1)
            @printf(
                "    radius=%.3g expected=%d sigma_expected=%.3e sigma_next=%.3e ranks={%s}\n",
                radius,
                expected_count,
                expected_ratio,
                next_ratio,
                rank_text,
            )
        end
    end
end

function run_scalar_nested_contour_demo(;
    case=scalar_cosine_case(),
    outer_radius=20.0,
    radii=(6.0, 10.0, 14.0, 20.0),
    moment_offsets=(0, 2),
    extraction=:shifted_gauge_balanced,
    iterations=12,
    node_factor=12,
    ranktol=1e-10,
    residual_tol=1e-8,
)
    center = 0.0 + 0.0im
    expected = case.roots(center, outer_radius)
    found = ComplexF64[]

    println()
    println("Nested scalar contour demo: $(case.name) outer_radius=$outer_radius expected_inside=$(length(expected))")
    for radius in radii
        local_expected = case.roots(center, radius)
        moment_count = length(local_expected)
        nodes = max(64, node_factor * moment_count)
        for moment_offset in moment_offsets
            result = solve_scalar_chart(
                case,
                center,
                radius;
                moment_count=moment_count,
                moment_offset=moment_offset,
                iterations=iterations,
                nodes=nodes,
                ranktol=ranktol,
                residual_tol=residual_tol,
                extraction=extraction,
            )
            append!(found, result.values)
            @printf(
                "  radius=%.3g offset=%d good=%d/%d max=%.3e eigcond=%.3e\n",
                radius,
                moment_offset,
                result.good_count,
                moment_count,
                result.max_inside_residual,
                result.eigcond,
            )
        end
    end

    unique = unique_values(sort(found; by=z -> (real(z), imag(z))); atol=1e-6)
    matched = match_expected_count(unique, expected; atol=1e-6)
    @printf(
        "  union unique=%d matched=%d/%d\n",
        length(unique),
        matched,
        length(expected),
    )
    unique
end

function adaptive_scalar_radii(outer_radius; first_radius=6.0, growth=1.6)
    radii = Float64[]
    radius = min(first_radius, outer_radius)
    while radius < outer_radius * (1 - 1e-12)
        push!(radii, radius)
        radius = min(outer_radius, radius * growth)
    end
    isempty(radii) || radii[end] != outer_radius || return radii
    push!(radii, outer_radius)
    radii
end

function run_scalar_adaptive_chart_demo(;
    case=scalar_cosine_case(),
    outer_radius=20.0,
    first_radius=6.0,
    growth=1.6,
    moment_offsets=(0, 2),
    iterations=12,
    node_factor=12,
    ranktol=1e-10,
    residual_tol=1e-8,
)
    center = 0.0 + 0.0im
    expected = case.roots(center, outer_radius)
    found = ComplexF64[]
    radii = adaptive_scalar_radii(outer_radius; first_radius=first_radius, growth=growth)

    println()
    println("Adaptive scalar chart demo: $(case.name) outer_radius=$outer_radius expected_inside=$(length(expected))")
    println("  supervised prototype: exact local counts are used only to size each finite realization")
    for radius in radii
        local_expected = case.roots(center, radius)
        moment_count = length(local_expected)
        moment_count == 0 && continue
        nodes = max(64, node_factor * moment_count)
        best = nothing
        best_new = -1
        for moment_offset in moment_offsets
            result = solve_scalar_chart(
                case,
                center,
                radius;
                moment_count=moment_count,
                moment_offset=moment_offset,
                iterations=iterations,
                nodes=nodes,
                ranktol=ranktol,
                residual_tol=residual_tol,
            )
            candidate_unique = unique_values(vcat(found, result.values); atol=1e-6)
            new_count = length(candidate_unique) - length(unique_values(found; atol=1e-6))
            if new_count > best_new || (new_count == best_new && result.max_inside_residual < best.max_inside_residual)
                best = merge(result, (offset=moment_offset, new_count=new_count))
                best_new = new_count
            end
        end
        append!(found, best.values)
        unique = unique_values(sort(found; by=z -> (real(z), imag(z))); atol=1e-6)
        matched = match_expected_count(unique, expected; atol=1e-6)
        @printf(
            "  radius=%.3g chose_offset=%d new=%d unique=%d matched=%d/%d max=%.3e eigcond=%.3e\n",
            radius,
            best.offset,
            best.new_count,
            length(unique),
            matched,
            length(expected),
            best.max_inside_residual,
            best.eigcond,
        )
        matched == length(expected) && break
    end

    unique = unique_values(sort(found; by=z -> (real(z), imag(z))); atol=1e-6)
    matched = match_expected_count(unique, expected; atol=1e-6)
    @printf("  final unique=%d matched=%d/%d\n", length(unique), matched, length(expected))
    unique
end

function run_scalar_rank_adaptive_chart_demo(;
    case=scalar_cosine_case(),
    outer_radius=20.0,
    first_radius=6.0,
    growth=1.6,
    capacity=24,
    count_ranktol=1e-6,
    moment_offsets=(0, 2),
    iterations=12,
    node_factor=12,
    ranktol=1e-10,
    residual_tol=1e-8,
)
    center = 0.0 + 0.0im
    expected = case.roots(center, outer_radius)
    outer_estimate = estimate_scalar_chart_rank(
        case,
        center,
        outer_radius;
        capacity=capacity,
        count_ranktol=count_ranktol,
    )
    target_count = outer_estimate.rank
    found = ComplexF64[]
    radii = adaptive_scalar_radii(outer_radius; first_radius=first_radius, growth=growth)

    println()
    println("Rank-adaptive scalar chart demo: $(case.name) outer_radius=$outer_radius")
    println("  moment-rank target estimate=$target_count; known roots are used only for validation")
    for radius in radii
        local_estimate = estimate_scalar_chart_rank(
            case,
            center,
            radius;
            capacity=capacity,
            count_ranktol=count_ranktol,
        )
        moment_count = min(local_estimate.rank, target_count)
        moment_count == 0 && continue
        nodes = max(64, node_factor * moment_count)
        best = nothing
        best_new = -1
        for moment_offset in moment_offsets
            result = solve_scalar_chart(
                case,
                center,
                radius;
                moment_count=moment_count,
                moment_offset=moment_offset,
                iterations=iterations,
                nodes=nodes,
                ranktol=ranktol,
                residual_tol=residual_tol,
            )
            previous_unique = unique_values(found; atol=1e-6)
            candidate_unique = unique_values(vcat(found, result.values); atol=1e-6)
            new_count = length(candidate_unique) - length(previous_unique)
            if new_count > best_new || (new_count == best_new && result.max_inside_residual < best.max_inside_residual)
                best = merge(result, (offset=moment_offset, new_count=new_count))
                best_new = new_count
            end
        end
        append!(found, best.values)
        unique = unique_values(sort(found; by=z -> (real(z), imag(z))); atol=1e-6)
        matched = match_expected_count(unique, expected; atol=1e-6)
        @printf(
            "  radius=%.3g estimated_rank=%d chose_offset=%d new=%d unique=%d matched=%d/%d max=%.3e eigcond=%.3e\n",
            radius,
            moment_count,
            best.offset,
            best.new_count,
            length(unique),
            matched,
            length(expected),
            best.max_inside_residual,
            best.eigcond,
        )
        radius == outer_radius && length(unique) >= target_count && break
    end

    unique = unique_values(sort(found; by=z -> (real(z), imag(z))); atol=1e-6)
    matched = match_expected_count(unique, expected; atol=1e-6)
    @printf(
        "  final unique=%d matched=%d/%d target_estimate=%d\n",
        length(unique),
        matched,
        length(expected),
        target_count,
    )
    unique
end

function rank_adaptive_scalar_chart(
    case,
    outer_radius;
    first_radius=6.0,
    growth=1.6,
    capacity=24,
    count_ranktol=1e-6,
    moment_offsets=(0, 2),
    iterations=12,
    node_factor=12,
    ranktol=1e-10,
    residual_tol=1e-8,
)
    center = 0.0 + 0.0im
    expected = case.roots(center, outer_radius)
    outer_estimate = estimate_scalar_chart_rank(
        case,
        center,
        outer_radius;
        capacity=capacity,
        count_ranktol=count_ranktol,
    )
    target_count = outer_estimate.rank
    found = ComplexF64[]
    steps = Any[]
    radii = adaptive_scalar_radii(outer_radius; first_radius=first_radius, growth=growth)

    for radius in radii
        local_estimate = estimate_scalar_chart_rank(
            case,
            center,
            radius;
            capacity=capacity,
            count_ranktol=count_ranktol,
        )
        moment_count = min(local_estimate.rank, target_count)
        moment_count == 0 && continue
        nodes = max(64, node_factor * moment_count)
        best = nothing
        best_new = -1
        for moment_offset in moment_offsets
            result = solve_scalar_chart(
                case,
                center,
                radius;
                moment_count=moment_count,
                moment_offset=moment_offset,
                iterations=iterations,
                nodes=nodes,
                ranktol=ranktol,
                residual_tol=residual_tol,
            )
            previous_unique = unique_values(found; atol=1e-6)
            candidate_unique = unique_values(vcat(found, result.values); atol=1e-6)
            new_count = length(candidate_unique) - length(previous_unique)
            if new_count > best_new || (new_count == best_new && result.max_inside_residual < best.max_inside_residual)
                best = merge(result, (offset=moment_offset, new_count=new_count))
                best_new = new_count
            end
        end
        append!(found, best.values)
        unique = unique_values(sort(found; by=z -> (real(z), imag(z))); atol=1e-6)
        matched = match_expected_count(unique, expected; atol=1e-6)
        push!(
            steps,
            (
                radius=radius,
                estimated_rank=moment_count,
                offset=best.offset,
                new_count=best.new_count,
                unique=length(unique),
                matched=matched,
                max_residual=best.max_inside_residual,
                eigcond=best.eigcond,
            ),
        )
        radius == outer_radius && length(unique) >= target_count && break
    end

    unique = unique_values(sort(found; by=z -> (real(z), imag(z))); atol=1e-6)
    (
        values=unique,
        target_count=target_count,
        expected_count=length(expected),
        matched=match_expected_count(unique, expected; atol=1e-6),
        steps=steps,
    )
end

function run_scalar_rank_adaptive_stress(;
    cases=(scalar_sine_case(), scalar_cosine_case(), scalar_shifted_sine_case(), scalar_expm1_case()),
    outer_radii=(10.0, 20.0),
    count_ranktol=1e-6,
)
    println()
    println("Scalar rank-adaptive chart stress")
    println("  known roots validate recovery but do not size local charts")
    for case in cases
        for outer_radius in outer_radii
            result = try
                rank_adaptive_scalar_chart(
                    case,
                    outer_radius;
                    count_ranktol=count_ranktol,
                    first_radius=min(6.0, outer_radius),
                )
            catch err
                println("  case=$(case.name) outer_radius=$outer_radius failed: $(typeof(err)) $err")
                continue
            end
            recovered = result.matched == result.expected_count
            target_ok = result.target_count == result.expected_count
            status = recovered ? (target_ok ? "ok" : "recovered_target_mismatch") : "miss"
            @printf(
                "  case=%s outer_radius=%.3g status=%s target=%d expected=%d matched=%d steps=%d\n",
                case.name,
                outer_radius,
                status,
                result.target_count,
                result.expected_count,
                result.matched,
                length(result.steps),
            )
            if status != "ok"
                for step in result.steps
                    @printf(
                        "    step radius=%.3g rank=%d offset=%d new=%d unique=%d matched=%d max=%.3e eigcond=%.3e\n",
                        step.radius,
                        step.estimated_rank,
                        step.offset,
                        step.new_count,
                        step.unique,
                        step.matched,
                        step.max_residual,
                        step.eigcond,
                    )
                end
            end
        end
    end
end

function diagonal_analytic_tools(cases)
    n = length(cases)
    function Tsolve(z, B)
        Y = similar(B, ComplexF64)
        for i in 1:n
            Y[i, :] .= B[i, :] ./ cases[i].f(z)
        end
        Y
    end
    function pair_residual(X, S)
        R = zeros(ComplexF64, size(X))
        for i in 1:n
            R[i:i, :] .= X[i:i, :] * cases[i].fmat(S)
        end
        R
    end
    function scalar_residual(λ, v)
        diagonal_values = ComplexF64[cases[i].f(λ) for i in 1:n]
        vnorm = norm(v)
        vnorm <= eps(Float64) && return Inf
        norm(diagonal_values .* v) / (max(norm(diagonal_values), eps(Float64)) * vnorm)
    end
    function expected_roots(center, radius)
        roots = ComplexF64[]
        for case in cases
            append!(roots, case.roots(center, radius))
        end
        unique_values(sort(roots; by=z -> (real(z), imag(z))); atol=1e-10)
    end
    Tsolve, pair_residual, scalar_residual, expected_roots
end

function best_history_scalar_summary(history, expected, center, radius; residual_tol=1e-8, match_tol=1e-6)
    best = nothing
    for h in history
        inside = FEASTSolver.in_contour(h.lambdas, center, radius)
        good = inside .& (h.residuals .<= residual_tol)
        good_values = h.lambdas[good]
        matched = match_expected_count(good_values, expected; atol=match_tol)
        candidate = (
            history=h,
            iteration=h.iteration,
            good_values=good_values,
            good_count=length(good_values),
            matched=matched,
            max_residual=h.max_inside_residual,
            pair_residual=h.pair_residual,
        )
        if best === nothing ||
           candidate.matched > best.matched ||
           (candidate.matched == best.matched && candidate.good_count > best.good_count) ||
           (candidate.matched == best.matched && candidate.good_count == best.good_count && candidate.max_residual < best.max_residual)
            best = candidate
        end
    end
    best
end

function run_diagonal_analytic_moment_stress(;
    cases=(scalar_sine_case(), scalar_cosine_case()),
    radii=(10.0, 20.0),
    capacity=32,
    count_ranktol=1e-6,
    moment_offsets=(0, 2),
    extraction=:chebyshev_shifted_observable_eigen,
    iterations=6,
    node_factor=12,
    ranktol=1e-10,
    residual_tol=1e-8,
)
    center = 0.0 + 0.0im
    Tsolve, pair_residual, scalar_residual, expected_roots = diagonal_analytic_tools(cases)
    n = length(cases)
    labels = join((case.name for case in cases), ",")

    println()
    println("Diagonal analytic moment stress: [$labels]")
    println("  non-scalar diagonal NEP; known roots validate but do not size the rank estimate")
    println("  extraction=$extraction iterations=$iterations")
    for radius in radii
        expected = expected_roots(center, radius)
        Random.seed!(9101 + round(Int, radius * 10))
        Xprobe = rand(ComplexF64, n, n)
        estimate = estimate_generic_chart_rank(
            Tsolve,
            Xprobe;
            center=center,
            radius=radius,
            capacity=capacity,
            nodes=max(128, 16 * capacity),
            count_ranktol=count_ranktol,
        )
        moment_count = max(1, estimate.rank)
        nodes = max(64, node_factor * moment_count)
        best = nothing
        best_matched = -1
        for moment_offset in moment_offsets
            X, S, history = try
                moment_rii_pair_generic_scaled(
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
                    maxrank=moment_count,
                    residual_tol=residual_tol,
                    keep=moment_count,
                    extraction=extraction,
                    moment_offset=moment_offset,
                )
            catch err
                println("  radius=$radius offset=$moment_offset failed: $(typeof(err)) $err")
                continue
            end
            summary = best_history_scalar_summary(history, expected, center, radius; residual_tol=residual_tol)
            h = summary.history
            conditioning = small_operator_conditioning(S)
            candidate = (
                offset=moment_offset,
                iteration=summary.iteration,
                rank=h.rank,
                good=summary.good_count,
                inside=h.inside,
                matched=summary.matched,
                max_residual=summary.max_residual,
                pair_residual=summary.pair_residual,
                eigcond=conditioning.eigcond,
            )
            if candidate.matched > best_matched || (candidate.matched == best_matched && candidate.max_residual < best.max_residual)
                best = candidate
                best_matched = candidate.matched
            end
        end
        if best === nothing
            @printf(
                "  radius=%.3g expected=%d estimated_rank=%d failed_all_offsets\n",
                radius,
                length(expected),
                moment_count,
            )
        else
            status = best.matched == length(expected) ? "ok" : "miss"
            @printf(
                "  radius=%.3g status=%s expected=%d estimated_rank=%d offset=%d best_iter=%d good=%d/%d matched=%d max=%.3e pair=%.3e eigcond=%.3e\n",
                radius,
                status,
                length(expected),
                moment_count,
                best.offset,
                best.iteration,
                best.good,
                best.inside,
                best.matched,
                best.max_residual,
                best.pair_residual,
                best.eigcond,
            )
        end
    end
end

function rank_adaptive_diagonal_analytic_chart(
    cases,
    outer_radius;
    first_radius=6.0,
    growth=1.6,
    capacity=32,
    count_ranktol=1e-6,
    moment_offsets=(0, 2),
    iterations=6,
    node_factor=12,
    ranktol=1e-10,
    residual_tol=1e-8,
    extraction=:chebyshev_shifted_observable_eigen,
)
    center = 0.0 + 0.0im
    Tsolve, pair_residual, scalar_residual, expected_roots = diagonal_analytic_tools(cases)
    expected = expected_roots(center, outer_radius)
    n = length(cases)
    Random.seed!(9201 + round(Int, outer_radius * 10))
    Xprobe = rand(ComplexF64, n, n)
    outer_estimate = estimate_generic_chart_rank(
        Tsolve,
        Xprobe;
        center=center,
        radius=outer_radius,
        capacity=capacity,
        nodes=max(128, 16 * capacity),
        count_ranktol=count_ranktol,
    )
    target_count = outer_estimate.rank
    found = ComplexF64[]
    steps = Any[]
    radii = adaptive_scalar_radii(outer_radius; first_radius=first_radius, growth=growth)

    for radius in radii
        local_estimate = estimate_generic_chart_rank(
            Tsolve,
            Xprobe;
            center=center,
            radius=radius,
            capacity=capacity,
            nodes=max(128, 16 * capacity),
            count_ranktol=count_ranktol,
        )
        moment_count = min(local_estimate.rank, capacity)
        moment_count == 0 && continue
        nodes = max(64, node_factor * moment_count)
        best = nothing
        best_new = -1
        for moment_offset in moment_offsets
            X, S, history = try
                moment_rii_pair_generic_scaled(
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
                    maxrank=moment_count,
                    residual_tol=residual_tol,
                    keep=moment_count,
                    extraction=extraction,
                    moment_offset=moment_offset,
                )
            catch
                continue
            end
            summary = best_history_scalar_summary(history, expected, center, radius; residual_tol=residual_tol)
            h = summary.history
            previous_unique = unique_values(found; atol=1e-6)
            candidate_unique = unique_values(vcat(found, summary.good_values); atol=1e-6)
            new_count = length(candidate_unique) - length(previous_unique)
            conditioning = small_operator_conditioning(S)
            candidate = (
                values=summary.good_values,
                offset=moment_offset,
                iteration=summary.iteration,
                rank=h.rank,
                new_count=new_count,
                inside=h.inside,
                good=summary.good_count,
                max_residual=summary.max_residual,
                pair_residual=summary.pair_residual,
                eigcond=conditioning.eigcond,
            )
            if new_count > best_new || (new_count == best_new && candidate.max_residual < best.max_residual)
                best = candidate
                best_new = new_count
            end
        end
        best === nothing && continue
        append!(found, best.values)
        unique = unique_values(sort(found; by=z -> (real(z), imag(z))); atol=1e-6)
        matched = match_expected_count(unique, expected; atol=1e-6)
        push!(
            steps,
            (
                radius=radius,
                estimated_rank=moment_count,
                offset=best.offset,
                iteration=best.iteration,
                new_count=best.new_count,
                unique=length(unique),
                matched=matched,
                max_residual=best.max_residual,
                eigcond=best.eigcond,
            ),
        )
        radius == outer_radius && length(unique) >= target_count && break
    end

    unique = unique_values(sort(found; by=z -> (real(z), imag(z))); atol=1e-6)
    (
        values=unique,
        target_count=target_count,
        expected_count=length(expected),
        matched=match_expected_count(unique, expected; atol=1e-6),
        steps=steps,
    )
end

function run_diagonal_analytic_rank_adaptive_stress(;
    cases=(scalar_sine_case(), scalar_cosine_case()),
    outer_radii=(10.0, 20.0),
    count_ranktol=1e-6,
    extraction=:chebyshev_shifted_observable_eigen,
    iterations=6,
)
    labels = join((case.name for case in cases), ",")
    println()
    println("Diagonal analytic rank-adaptive chart stress: [$labels]")
    println("  extraction=$extraction iterations=$iterations")
    for outer_radius in outer_radii
        result = rank_adaptive_diagonal_analytic_chart(
            cases,
            outer_radius;
            count_ranktol=count_ranktol,
            extraction=extraction,
            iterations=iterations,
            first_radius=min(6.0, outer_radius),
        )
        recovered = result.matched == result.expected_count
        target_ok = result.target_count == result.expected_count
        status = recovered ? (target_ok ? "ok" : "recovered_target_mismatch") : "miss"
        @printf(
            "  outer_radius=%.3g status=%s target=%d expected=%d matched=%d steps=%d\n",
            outer_radius,
            status,
            result.target_count,
            result.expected_count,
            result.matched,
            length(result.steps),
        )
        if status != "ok"
            for step in result.steps
                @printf(
                    "    step radius=%.3g rank=%d offset=%d best_iter=%d new=%d unique=%d matched=%d max=%.3e eigcond=%.3e\n",
                    step.radius,
                    step.estimated_rank,
                    step.offset,
                    step.iteration,
                    step.new_count,
                    step.unique,
                    step.matched,
                    step.max_residual,
                    step.eigcond,
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

function companion_pencil(coeffs)
    n = size(coeffs[1], 1)
    degree = length(coeffs) - 1
    C1 = zeros(ComplexF64, n * degree, n * degree)
    C2 = zeros(ComplexF64, n * degree, n * degree)
    C1[1:n, 1:n] .= coeffs[1]
    for i in (n + 1):(n * degree)
        C1[i, i] = 1
        C2[i, i - n] = 1
    end
    for i in 1:degree
        C2[1:n, n * (i - 1) + 1:n * i] .= -coeffs[i + 1]
    end
    C1, C2
end

function polynomial_residuals_from_companion_vectors(coeffs, values, companion_vectors)
    n = size(coeffs[1], 1)
    degree = length(coeffs) - 1
    physical = companion_vectors[(degree - 1) * n + 1:degree * n, :]
    residuals = zeros(Float64, length(values))
    for j in eachindex(values)
        v = physical[:, j]
        vnorm = norm(v)
        if vnorm <= eps(Float64)
            residuals[j] = Inf
        else
            v ./= vnorm
            Tλ = polynomial_matrix(coeffs, values[j])
            residuals[j] = norm(Tλ * v) / max(norm(Tλ), eps(Float64))
        end
    end
    residuals
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

function retained_anchor_values(S, center, radius)
    values = ComplexF64.(eigvals(S))
    inside = FEASTSolver.in_contour(values, center, radius)
    values[inside]
end

function final_persistence_score(history, center, radius)
    length(history) < 2 && return NaN
    previous = history[end - 1]
    current = history[end]
    previous_inside = FEASTSolver.in_contour(previous.lambdas, center, radius)
    current_inside = FEASTSolver.in_contour(current.lambdas, center, radius)
    anchors = previous.lambdas[previous_inside]
    values = current.lambdas[current_inside]
    (isempty(anchors) || isempty(values)) && return NaN
    maximum(value -> nearest_anchor_score(value, anchors, radius), values)
end

function update_and_retention_policy(mode)
    if mode === :shifted_residual
        return :shifted, :residual, :none
    elseif mode === :shifted_persistent
        return :shifted, :persistent, :none
    elseif mode === :shifted_residual_persistent
        return :shifted, :residual_persistent, :none
    elseif mode === :shifted_gauge_balanced
        return :shifted, :contour, :diagonal_balance
    elseif mode === :shifted_gauge_balanced_residual
        return :shifted, :residual, :diagonal_balance
    elseif mode === :shifted_schur
        return :shifted, :contour, :schur
    elseif mode === :shifted_schur_balanced
        return :shifted, :contour, :schur_diagonal_balance
    elseif mode === :balanced_shifted_residual
        return :balanced_shifted, :residual, :none
    elseif mode === :balanced_shifted_persistent
        return :balanced_shifted, :persistent, :none
    elseif mode === :balanced_shifted_residual_persistent
        return :balanced_shifted, :residual_persistent, :none
    elseif mode === :projected_residual
        return :projected, :residual, :none
    elseif mode === :projected_persistent
        return :projected, :persistent, :none
    elseif mode === :projected_residual_persistent
        return :projected, :residual_persistent, :none
    else
        return mode, :contour, :none
    end
end

function finite_eigenvalue_mask(values)
    map(values) do z
        isfinite(real(z)) && isfinite(imag(z))
    end
end

function run_polynomial_companion_control(
    name,
    make_problem;
    nodes,
    iterations,
    moment_count,
    seed=8801,
    ranktol=1e-10,
    residual_tol=1e-8,
    modes=(:projected_newton, :shifted_gauge_balanced),
)
    coeffs, center, radius, probe_cols = make_problem()
    expected = companion_reference(coeffs, center, radius)
    n = size(coeffs[1], 1)
    C1, C2 = companion_pencil(coeffs)
    companion_m = min(size(C1, 1), max(length(expected) + 4, moment_count * probe_cols))

    Random.seed!(seed)
    Xcomp = rand(ComplexF64, size(C1, 1), companion_m)
    comp_values, comp_vectors, comp_feast_residuals = gen_feast!(
        Xcomp,
        C1,
        C2;
        nodes=nodes,
        iter=iterations,
        c=center,
        r=radius,
        ϵ=residual_tol,
        store=false,
    )
    comp_poly_residuals = polynomial_residuals_from_companion_vectors(coeffs, comp_values, comp_vectors)
    companion_matched = match_expected_count(comp_values, expected; atol=1e-6)

    println()
    println("Polynomial companion control: $name")
    println("  finite polynomial target count: $(length(expected)); companion size=$(size(C1, 1)); FEAST subspace=$companion_m")
    @printf(
        "  companion FEAST returned=%d matched=%d/%d max_gen_res=%.3e max_poly_res=%.3e\n",
        length(comp_values),
        companion_matched,
        length(expected),
        isempty(comp_feast_residuals) ? Inf : maximum(comp_feast_residuals),
        isempty(comp_poly_residuals) ? Inf : maximum(comp_poly_residuals),
    )

    Random.seed!(seed + 1)
    Xprobe = rand(ComplexF64, n, probe_cols)
    W = rand(ComplexF64, n, probe_cols)
    maxrank = min(moment_count * probe_cols, moment_count * size(W, 2))
    keep = min(maxrank, length(expected))
    for mode in modes
        update_mode, retention_policy, gauge = update_and_retention_policy(mode)
        X, S, history = moment_rii_pair_projected_scaled(
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
            target_count=true,
            update_mode=update_mode,
            retention_policy=retention_policy,
            gauge=gauge,
        )
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
        hend = last(history)
        @printf(
            "  moment mode=%s rank=%d conv=%d/%d matched=%d/%d max=%.3e pair=%.3e\n",
            string(mode),
            hend.rank,
            diag.converged_inside,
            diag.inside,
            diag.matched,
            length(expected),
            diag.max_inside,
            diag.pair_wanted,
        )
    end
end

function degree_deficient_projective_polynomial()
    coeffs = [
        Matrix(Diagonal(ComplexF64[1, -3])),
        Matrix(Diagonal(ComplexF64[-2, 1])),
        Matrix(Diagonal(ComplexF64[1, 0])),
    ]
    coeffs
end

function run_polynomial_projective_chart_control(; nodes=16, iterations=4, ranktol=1e-10, residual_tol=1e-10)
    coeffs = degree_deficient_projective_polynomial()
    C1, C2 = companion_pencil(coeffs)
    values = ComplexF64.(eigen(C1, C2).values)
    finite = finite_eigenvalue_mask(values)
    reversed_coeffs = reverse(coeffs)
    center = 0.0 + 0.0im
    radius = 0.15
    # The reversed chart maps the infinite lambda-root to nu=0. The generic
    # companion residual filter divides by norm(T(0)), which is zero for this
    # deliberately singular projective example, so keep the known root explicit.
    expected_nu = ComplexF64[0.0 + 0.0im]

    Random.seed!(8817)
    Xprobe = rand(ComplexF64, size(coeffs[1], 1), 1)
    W = rand(ComplexF64, size(coeffs[1], 1), 1)
    X, S, history = moment_rii_pair_projected_scaled(
        reversed_coeffs,
        Xprobe,
        W;
        center=center,
        radius=radius,
        nodes=nodes,
        iterations=iterations,
        moment_count=1,
        ranktol=ranktol,
        maxrank=1,
        residual_tol=residual_tol,
        keep=1,
        target_count=true,
        update_mode=:projected_newton,
    )
    diag = state_contamination_diagnostics(
        reversed_coeffs,
        X,
        S,
        expected_nu,
        center,
        radius;
        residual_tol=residual_tol,
        match_tol=1e-8,
    )
    hend = last(history)

    println()
    println("Polynomial projective chart control")
    println("  P(lambda) has one eigenvalue at infinity because the leading coefficient is singular")
    @printf(
        "  original companion finite=%d infinite_or_singular=%d\n",
        count(finite),
        count(.!finite),
    )
    @printf(
        "  reversed chart nu=1/lambda around zero expected=%d rank=%d matched=%d/%d max=%.3e pair=%.3e\n",
        length(expected_nu),
        hend.rank,
        diag.matched,
        length(expected_nu),
        diag.max_inside,
        diag.pair_wanted,
    )
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
        update_mode, retention_policy, gauge = update_and_retention_policy(mode)
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
                update_mode=update_mode,
                newton_steps=newton_steps,
                retention_policy=retention_policy,
                gauge=gauge,
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
        persistence = final_persistence_score(history, center, radius)
        conditioning = small_operator_conditioning(S)
        @printf(
            "  mode=%s iter0(rank=%d conv=%d/%d max=%.3e pair=%.3e) final(rank=%d conv=%d/%d matched=%d extra=%d max=%.3e pair_all=%.3e pair_wanted=%.3e sigma_rank/sigma1=%.3e persist=%.3e eigcond=%.3e condS=%.3e)\n",
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
            persistence,
            conditioning.eigcond,
            conditioning.condS,
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
    run_polynomial_companion_control(
        "many_eigenvalue_nonnormal_polynomial",
        many_eigenvalue_nonnormal_polynomial_problem;
        nodes=16,
        iterations=6,
        moment_count=5,
        ranktol=1e-10,
        residual_tol=1e-8,
    )
    run_polynomial_projective_chart_control()
    run_nonlinear_update_comparison("deficient_quadratic", deficient_quadratic_problem; nodes=16, iterations=5, moment_count=2, ranktol=1e-9, residual_tol=1e-8)
    run_nonlinear_update_comparison("butterfly", butterfly_problem; nodes=32, iterations=3, moment_count=2, ranktol=1e-9, residual_tol=1e-8, keep_extra=3)
    run_nonlinear_update_comparison("butterfly_target", butterfly_problem; nodes=32, iterations=3, moment_count=2, ranktol=1e-9, residual_tol=1e-8, target_count=true, modes=(:projected, :shifted, :shifted_gauge_balanced, :projected_newton))
    run_nonlinear_update_comparison(
        "many_eigenvalue_nonnormal_polynomial_K4_capacity_failure",
        many_eigenvalue_nonnormal_polynomial_problem;
        nodes=32,
        iterations=6,
        moment_count=4,
        ranktol=1e-10,
        residual_tol=1e-8,
        target_count=true,
        modes=(:projected, :shifted, :shifted_gauge_balanced),
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
        modes=(:projected, :shifted, :shifted_gauge_balanced),
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
        modes=(:projected, :shifted, :shifted_gauge_balanced),
    )
    run_scalar_sine_sweep()
    run_scalar_sine_scaled_sweep()
    run_scalar_sine_realization_sweep()
    run_scalar_sine_realization_sweep(radius=20.0, node_values=(128,), iterations=8, moment_counts=(13,), ranktol=1e-10, residual_tol=1e-8)
    run_scalar_nested_contour_demo()
    run_scalar_adaptive_chart_demo()
    run_scalar_rank_adaptive_chart_demo()
    run_scalar_rank_estimation_stress()
    run_scalar_rank_adaptive_stress()
    run_diagonal_analytic_moment_stress()
    run_diagonal_analytic_rank_adaptive_stress()
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
