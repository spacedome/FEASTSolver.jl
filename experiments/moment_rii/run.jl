using FEASTSolver
using LinearAlgebra
using MatrixMarket
using Random
using Printf
using SparseArrays
using Distributed

const REPO_ROOT = normpath(joinpath(@__DIR__, "..", ".."))

if !isdefined(@__MODULE__, :_MOMENT_RII_REMOTE_WORKSPACES)
    const _MOMENT_RII_REMOTE_WORKSPACES = Dict{Symbol, Any}()
end

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

include("pipeline.jl")

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

function lifted_pair_derivative(X, S, ΔX, ΔS, lift)
    n, p = size(X)
    lifted = zeros(ComplexF64, lift * n, p)
    powers = Vector{Matrix{ComplexF64}}(undef, lift)
    powers[1] = Matrix{ComplexF64}(I, p, p)
    for j in 2:lift
        powers[j] = powers[j - 1] * S
    end

    for j in 1:lift
        block = ΔX * powers[j]
        for q in 0:(j - 2)
            block .+= X * powers[q + 1] * ΔS * powers[j - 1 - q]
        end
        lifted[(j - 1) * n + 1:j * n, :] .= block
    end
    lifted
end

function analytic_pair_residual_from_contour(Tred, X, S, center, radius; nodes=256)
    z_nodes, z_weights = circular_rule(center, radius, nodes)
    n, p = size(X)
    residual = zeros(ComplexF64, n, p)
    I_p = Matrix{ComplexF64}(I, p, p)
    for (z, weight) in zip(z_nodes, z_weights)
        residual .+= weight .* (Tred(z) * X / (z .* I_p .- S))
    end
    residual
end

function analytic_pair_residual_derivative_from_contour(Tred, X, S, ΔX, ΔS, center, radius; nodes=256)
    z_nodes, z_weights = circular_rule(center, radius, nodes)
    n, p = size(X)
    derivative = zeros(ComplexF64, n, p)
    I_p = Matrix{ComplexF64}(I, p, p)
    for (z, weight) in zip(z_nodes, z_weights)
        resolvent = inv(z .* I_p .- S)
        derivative .+= weight .* (Tred(z) * ΔX * resolvent)
        derivative .+= weight .* (Tred(z) * X * resolvent * ΔS * resolvent)
    end
    derivative
end

function analytic_invariant_pair_newton_step(Tred, X, S, center, radius; lift=1, nodes=256, damping=1.0)
    n, p = size(X)
    R = analytic_pair_residual_from_contour(Tred, X, S, center, radius; nodes=nodes)
    L = lifted_pair_matrix(X, S, lift)
    unknowns = n * p + p * p
    equations = n * p + p * p
    J = zeros(ComplexF64, equations, unknowns)

    column = 1
    for col in 1:p, row in 1:n
        ΔX = zeros(ComplexF64, n, p)
        ΔS = zeros(ComplexF64, p, p)
        ΔX[row, col] = 1
        ΔR = analytic_pair_residual_derivative_from_contour(
            Tred,
            X,
            S,
            ΔX,
            ΔS,
            center,
            radius;
            nodes=nodes,
        )
        ΔL = lifted_pair_derivative(X, S, ΔX, ΔS, lift)
        J[1:n*p, column] .= vec(ΔR)
        J[n*p+1:end, column] .= vec(L' * ΔL)
        column += 1
    end
    for col in 1:p, row in 1:p
        ΔX = zeros(ComplexF64, n, p)
        ΔS = zeros(ComplexF64, p, p)
        ΔS[row, col] = 1
        ΔR = analytic_pair_residual_derivative_from_contour(
            Tred,
            X,
            S,
            ΔX,
            ΔS,
            center,
            radius;
            nodes=nodes,
        )
        ΔL = lifted_pair_derivative(X, S, ΔX, ΔS, lift)
        J[1:n*p, column] .= vec(ΔR)
        J[n*p+1:end, column] .= vec(L' * ΔL)
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
    for _ in 1:10
        candidate_X, candidate_S = normalize_lifted_pair(X .+ α .* ΔX, S .+ α .* ΔS, lift)
        candidate_residual = norm(analytic_pair_residual_from_contour(
            Tred,
            candidate_X,
            candidate_S,
            center,
            radius;
            nodes=nodes,
        ))
        if candidate_residual < best_residual
            best_X, best_S = candidate_X, candidate_S
            best_residual = candidate_residual
            break
        end
        α /= 2
    end
    best_X, best_S, best_residual / max(current, eps(Float64))
end

function analytic_invariant_pair_newton_refine(Tred, X, S, center, radius; lift=1, nodes=256, steps=1)
    ratios = Float64[]
    for _ in 1:steps
        Xnew, Snew, ratio = analytic_invariant_pair_newton_step(
            Tred,
            X,
            S,
            center,
            radius;
            lift=lift,
            nodes=nodes,
        )
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

function real_periodic_roots_in_contour(base, period, center, radius)
    lower = (real(center) - radius - base) / period
    upper = (real(center) + radius - base) / period
    candidates = [base + period * k for k in floor(Int, lower)-2:ceil(Int, upper)+2]
    scalar_roots_in_contour(candidates, center, radius)
end

function imaginary_periodic_roots_in_contour(base, period, center, radius)
    lower = (imag(center) - radius - base) / period
    upper = (imag(center) + radius - base) / period
    candidates = [im * (base + period * k) for k in floor(Int, lower)-2:ceil(Int, upper)+2]
    scalar_roots_in_contour(candidates, center, radius)
end

function scalar_sine_case()
    (
        name="sin",
        f=z -> sin(z),
        df=z -> cos(z),
        fmat=S -> sin(S),
        roots=(center, radius) -> real_periodic_roots_in_contour(0.0, pi, center, radius),
    )
end

function scalar_cosine_case()
    (
        name="cos",
        f=z -> cos(z),
        df=z -> -sin(z),
        fmat=S -> cos(S),
        roots=(center, radius) -> real_periodic_roots_in_contour(0.5pi, pi, center, radius),
    )
end

function scalar_shifted_sine_case(alpha=0.3)
    a = asin(alpha)
    (
        name="sin_minus_$alpha",
        f=z -> sin(z) - alpha,
        df=z -> cos(z),
        fmat=S -> sin(S) .- alpha .* Matrix{ComplexF64}(I, size(S, 1), size(S, 2)),
        roots=(center, radius) -> vcat(
            real_periodic_roots_in_contour(a, 2pi, center, radius),
            real_periodic_roots_in_contour(pi - a, 2pi, center, radius),
        ),
    )
end

function scalar_squared_sine_case()
    (
        name="sin_squared",
        f=z -> sin(z)^2,
        df=z -> 2sin(z) * cos(z),
        fmat=S -> sin(S)^2,
        roots=(center, radius) -> real_periodic_roots_in_contour(0.0, pi, center, radius),
        multiplicity=2,
    )
end

function scalar_expm1_case()
    (
        name="exp_minus_1",
        f=z -> exp(z) - 1,
        df=z -> exp(z),
        fmat=S -> exp(S) .- Matrix{ComplexF64}(I, size(S, 1), size(S, 2)),
        roots=(center, radius) -> imaginary_periodic_roots_in_contour(0.0, 2pi, center, radius),
    )
end

function scalar_delay_case(; a=0.4, b=2.0, tau=1.0)
    (
        name="delay_a$(a)_b$(b)_tau$(tau)",
        f=z -> z + a - b * exp(-tau * z),
        df=z -> 1 + b * tau * exp(-tau * z),
        fmat=S -> S .+ a .* Matrix{ComplexF64}(I, size(S, 1), size(S, 2)) .-
                  b .* exp((-tau) * S),
        # Deliberately no closed-form oracle: this case validates the
        # count-driven policy against the full-operator argument-principle
        # count instead of exact root matching.
        roots=(center, radius) -> ComplexF64[],
    )
end

function scalar_two_delay_case(; a=0.15, b=1.6, tau=0.7, c=0.85, sigma=1.4)
    (
        name="two_delay_a$(a)_b$(b)_tau$(tau)_c$(c)_sigma$(sigma)",
        f=z -> z + a - b * exp(-tau * z) - c * exp(-sigma * z),
        df=z -> 1 + b * tau * exp(-tau * z) + c * sigma * exp(-sigma * z),
        fmat=S -> begin
            Ired = Matrix{ComplexF64}(I, size(S, 1), size(S, 2))
            S .+ a .* Ired .- b .* exp((-tau) * S) .- c .* exp((-sigma) * S)
        end,
        # No closed-form oracle: this tests count-driven stopping on a
        # quasipolynomial component with more than one delay scale.
        roots=(center, radius) -> ComplexF64[],
    )
end

function case_root_multiplicity(case)
    hasproperty(case, :multiplicity) ? case.multiplicity : 1
end

function expected_roots_counting_multiplicity(cases, center, radius)
    roots = ComplexF64[]
    for case in cases
        for root in case.roots(center, radius)
            for _ in 1:case_root_multiplicity(case)
                push!(roots, root)
            end
        end
    end
    sort(roots; by=z -> (real(z), imag(z)))
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

function moment_block_basis(moments, moment_count; ranktol=1e-10)
    n, m = size(moments[1])
    blocks = zeros(ComplexF64, n, moment_count * m)
    for j in 1:moment_count
        blocks[:, (j - 1) * m + 1:j * m] .= moments[j]
    end
    F = svd(blocks)
    isempty(F.S) && return zeros(ComplexF64, n, 0), Float64[]
    rank = count(F.S ./ F.S[1] .> ranktol)
    rank = min(rank, n, length(F.S))
    F.U[:, 1:rank], copy(F.S)
end

function biorthogonalize_bases(Xbasis, Ybasis; ranktol=1e-12)
    M = Ybasis' * Xbasis
    F = svd(M)
    isempty(F.S) && return Xbasis[:, 1:0], Ybasis[:, 1:0], Float64[]
    rank = count(F.S ./ F.S[1] .> ranktol)
    rank = min(rank, size(Xbasis, 2), size(Ybasis, 2), length(F.S))
    rank == 0 && return Xbasis[:, 1:0], Ybasis[:, 1:0], copy(F.S)
    invsqrtσ = Diagonal(1 ./ sqrt.(F.S[1:rank]))
    Xb = Xbasis * F.V[:, 1:rank] * invsqrtσ
    Yb = Ybasis * F.U[:, 1:rank] * invsqrtσ
    Xb, Yb, copy(F.S)
end

function initial_adjoint_moments_diagonal_scaled(cases, Wprobe, z_nodes, z_weights, center, radius, moment_count)
    n, m = size(Wprobe)
    moments = [zeros(ComplexF64, n, m) for _ in 1:moment_count]
    solved = similar(Wprobe, ComplexF64)
    for (z, weight) in zip(z_nodes, z_weights)
        for i in 1:n
            solved[i, :] .= Wprobe[i, :] ./ conj(cases[i].f(z))
        end
        μ = conj((z - center) / radius)
        μpower = one(ComplexF64)
        for p in eachindex(moments)
            moments[p] .+= (conj(weight) * μpower) .* solved
            μpower *= μ
        end
    end
    moments
end

function initial_adjoint_moments_generic_scaled(Tadjoint_solve, Wprobe, z_nodes, z_weights, center, radius, moment_count)
    n, m = size(Wprobe)
    moments = [zeros(ComplexF64, n, m) for _ in 1:moment_count]
    for (z, weight) in zip(z_nodes, z_weights)
        solved = Tadjoint_solve(z, Wprobe)
        μ = conj((z - center) / radius)
        μpower = one(ComplexF64)
        for p in eachindex(moments)
            moments[p] .+= (conj(weight) * μpower) .* solved
            μpower *= μ
        end
    end
    moments
end

function monic_roots_from_power_sums(power_sums, root_count)
    root_count == 0 && return ComplexF64[]
    coeffs = zeros(ComplexF64, root_count)
    for k in 1:root_count
        term = power_sums[k]
        for j in 1:(k - 1)
            term += coeffs[j] * power_sums[k - j]
        end
        coeffs[k] = -term / k
    end
    companion = zeros(ComplexF64, root_count, root_count)
    companion[1, :] .= .-coeffs
    for i in 2:root_count
        companion[i, i - 1] = 1
    end
    eigvals(companion)
end

function determinant_power_sums(Tred, Tred_derivative; center, radius, nodes=1024, capacity=64)
    z_nodes, z_weights = circular_rule(center, radius, nodes)
    sums = zeros(ComplexF64, capacity + 1)
    for (z, weight) in zip(z_nodes, z_weights)
        M = Tred(z)
        dM = Tred_derivative(z)
        rhs = dM isa AbstractSparseMatrix ? Matrix(dM) : dM
        logarithmic_derivative = tr(M \ rhs)
        μ = (z - center) / radius
        μpower = one(ComplexF64)
        for k in 0:capacity
            sums[k + 1] += weight * μpower * logarithmic_derivative
            μpower *= μ
        end
    end
    root_count = max(0, round(Int, real(sums[1])))
    root_count <= capacity || error("root count $root_count exceeds determinant power-sum capacity $capacity")
    roots = monic_roots_from_power_sums(sums[2:root_count + 1], root_count)
    roots, root_count, sums
end

function determinant_value_and_derivative(Tred, Tred_derivative, λ)
    M = Tred(λ)
    value = det(M)
    derivative = try
        value * tr(M \ Tred_derivative(λ))
    catch
        NaN + NaN * im
    end
    if !isfinite(real(derivative)) || !isfinite(imag(derivative)) || abs(derivative) <= eps(Float64)
        h = sqrt(eps(Float64)) * max(1.0, abs(λ))
        derivative = (det(Tred(λ + h)) - det(Tred(λ - h))) / (2h)
    end
    value, derivative
end

function refine_determinant_roots(Tred, Tred_derivative, roots; steps=12, step_limit=1.0)
    refined = ComplexF64.(roots)
    for j in eachindex(refined)
        λ = refined[j]
        for _ in 1:steps
            value, derivative = determinant_value_and_derivative(Tred, Tred_derivative, λ)
            (!isfinite(real(value)) || !isfinite(imag(value)) || abs(derivative) <= eps(Float64)) && break
            step = value / derivative
            if abs(step) > step_limit
                step *= step_limit / abs(step)
            end
            λ -= step
            abs(step) <= 1e-12 * max(1.0, abs(λ)) && break
        end
        refined[j] = λ
    end
    refined
end

function reduced_nep_residual(Tred, λ)
    values = svdvals(Tred(λ))
    isempty(values) && return Inf
    minimum(values) / max(maximum(values), eps(Float64))
end

function reduced_right_singular_vectors(Tred, values)
    nred = size(Tred(values[1]), 2)
    vectors = zeros(ComplexF64, nred, length(values))
    for j in eachindex(values)
        F = svd(Tred(values[j]))
        vectors[:, j] .= F.V[:, end]
    end
    vectors
end

function reduced_left_right_singular_vectors(Tred, values)
    isempty(values) && return zeros(ComplexF64, 0, 0), zeros(ComplexF64, 0, 0)
    first_svd = svd(Tred(values[1]))
    left = zeros(ComplexF64, size(first_svd.U, 1), length(values))
    right = zeros(ComplexF64, size(first_svd.V, 1), length(values))
    left[:, 1] .= first_svd.U[:, end]
    right[:, 1] .= first_svd.V[:, end]
    for j in 2:length(values)
        F = svd(Tred(values[j]))
        left[:, j] .= F.U[:, end]
        right[:, j] .= F.V[:, end]
    end
    left, right
end

function refine_reduced_analytic_triplets(
    Tred,
    Tred_derivative,
    values;
    steps=4,
    step_limit=Inf,
    derivative_floor=eps(Float64),
)
    refined = ComplexF64.(values)
    isempty(refined) && return refined, Float64[]
    corrections = Float64[]
    for j in eachindex(refined)
        λ = refined[j]
        for _ in 1:steps
            F = svd(Tred(λ))
            y = F.U[:, end]
            x = F.V[:, end]
            value = dot(y, Tred(λ) * x)
            derivative = dot(y, Tred_derivative(λ) * x)
            if !isfinite(real(value)) || !isfinite(imag(value)) ||
                    !isfinite(real(derivative)) || !isfinite(imag(derivative)) ||
                    abs(derivative) <= derivative_floor
                break
            end
            correction = value / derivative
            if isfinite(step_limit) && abs(correction) > step_limit
                correction *= step_limit / abs(correction)
            end
            λ -= correction
            push!(corrections, abs(correction))
            abs(correction) <= 1e-12 * max(1.0, abs(λ)) && break
        end
        refined[j] = λ
    end
    refined, corrections
end

function refine_reduced_analytic_invariant_pair(
    Tred,
    values,
    right_reduced,
    center,
    radius;
    steps=1,
    nodes=256,
    lift=nothing,
    selection=:inside,
)
    selected = if selection === :inside
        FEASTSolver.in_contour(values, center, radius)
    elseif selection === :all
        trues(length(values))
    else
        error("unknown reduced analytic invariant-pair refinement selection: $selection")
    end
    any(selected) || return values, right_reduced, Float64[]

    nred = size(right_reduced, 1)
    p = count(selected)
    effective_lift = lift === nothing ? max(1, ceil(Int, p / max(nred, 1))) : Int(lift)
    effective_lift = max(effective_lift, ceil(Int, p / max(nred, 1)))

    X = Matrix(right_reduced[:, selected])
    S = Matrix(Diagonal(ComplexF64.(values[selected])))
    X, S = normalize_lifted_pair(X, S, effective_lift)
    X, S, ratios = analytic_invariant_pair_newton_refine(
        Tred,
        X,
        S,
        center,
        radius;
        lift=effective_lift,
        nodes=nodes,
        steps=steps,
    )
    F = eigen(S)

    refined_values = copy(values)
    refined_right = copy(right_reduced)
    refined_values[selected] .= ComplexF64.(F.values)
    refined_right[:, selected] .= X * F.vectors
    normalize_columns_local!(refined_right)
    refined_values, refined_right, ratios
end

function reduced_analytic_refined_left_right(
    Tred,
    Tred_derivative,
    values,
    center,
    radius;
    refinement=:none,
    refinement_steps=4,
    refinement_nodes=256,
)
    refinement_corrections = Float64[]
    refinement_ratios = Float64[]
    left_reduced, right_reduced = reduced_left_right_singular_vectors(Tred, values)
    if refinement === :scalar_newton
        values, refinement_corrections = refine_reduced_analytic_triplets(
            Tred,
            Tred_derivative,
            values;
            steps=refinement_steps,
            step_limit=0.25 * radius,
        )
        left_reduced, right_reduced = reduced_left_right_singular_vectors(Tred, values)
    elseif refinement === :block_newton
        values, right_reduced, refinement_ratios = refine_reduced_analytic_invariant_pair(
            Tred,
            values,
            right_reduced,
            center,
            radius;
            steps=refinement_steps,
            nodes=refinement_nodes,
        )
        left_reduced, _ = reduced_left_right_singular_vectors(Tred, values)
    elseif refinement !== :none
        error("unknown reduced analytic refinement: $refinement")
    end
    values, left_reduced, right_reduced, refinement_corrections, refinement_ratios
end

function diagonal_full_residuals(cases, values, vectors)
    n = length(cases)
    residuals = zeros(Float64, length(values))
    for j in eachindex(values)
        λ = values[j]
        x = vectors[:, j]
        xnorm = norm(x)
        if xnorm <= eps(Float64)
            residuals[j] = Inf
            continue
        end
        diagonal_values = ComplexF64[cases[i].f(λ) for i in 1:n]
        residuals[j] = norm(diagonal_values .* x) / (max(norm(diagonal_values), eps(Float64)) * xnorm)
    end
    residuals
end

function matrix_vector_residuals(Tmatrix, values, vectors; adjoint=false, normalization=:operator)
    residuals = zeros(Float64, length(values))
    for j in eachindex(values)
        x = vectors[:, j]
        xnorm = norm(x)
        if xnorm <= eps(Float64)
            residuals[j] = Inf
            continue
        end
        Tλ = Tmatrix(values[j])
        residual = adjoint ? Tλ' * x : Tλ * x
        denominator = if normalization === :operator
            max(norm(Tλ), eps(Float64)) * xnorm
        elseif normalization === :vector
            xnorm
        else
            error("unknown residual normalization: $normalization")
        end
        residuals[j] = norm(residual) / denominator
    end
    residuals
end

function diagonal_reduced_operators(cases, Xbasis, Ybasis)
    n = length(cases)
    function Tred(z)
        D = Diagonal(ComplexF64[cases[i].f(z) for i in 1:n])
        Ybasis' * D * Xbasis
    end
    function Tred_derivative(z)
        D = Diagonal(ComplexF64[cases[i].df(z) for i in 1:n])
        Ybasis' * D * Xbasis
    end
    Tred, Tred_derivative
end

function analytic_component_scales(cases, center, radius; mode=:none, nodes=64)
    if mode === :none
        return ones(Float64, length(cases))
    elseif mode === :center
        return Float64[max(1.0, abs(case.f(center))) for case in cases]
    elseif mode === :contour_max
        z_nodes, _ = circular_rule(center, radius, nodes)
        return Float64[
            max(1.0, maximum(abs(case.f(z)) for z in z_nodes))
            for case in cases
        ]
    end
    error("unknown analytic component scaling mode: $mode")
end

function similarity_analytic_tools(cases; component_scales=nothing)
    n = length(cases)
    nodes = 1.0 .+ 0.15 .* (0:n - 1)
    V = ComplexF64[nodes[i]^(j - 1) for i in 1:n, j in 1:n]
    n >= 2 && (V[:, 2] .+= 0.05im .* V[:, 1])
    Vinv = inv(V)
    scales = component_scales === nothing ? ones(Float64, n) : Float64.(component_scales)

    function diagonal_values(z)
        ComplexF64[cases[i].f(z) / scales[i] for i in 1:n]
    end
    function diagonal_derivatives(z)
        ComplexF64[cases[i].df(z) / scales[i] for i in 1:n]
    end
    function Tmatrix(z)
        V * Diagonal(diagonal_values(z)) * Vinv
    end
    function Tderivative(z)
        V * Diagonal(diagonal_derivatives(z)) * Vinv
    end
    function Tsolve(z, B)
        V * (Diagonal(1 ./ diagonal_values(z)) * (Vinv * B))
    end
    function Tadjoint_solve(z, B)
        Vinv' * (Diagonal(1 ./ conj.(diagonal_values(z))) * (V' * B))
    end
    function expected_roots(center, radius)
        roots = ComplexF64[]
        for case in cases
            append!(roots, case.roots(center, radius))
        end
        unique_values(sort(roots; by=z -> (real(z), imag(z))); atol=1e-10)
    end
    Tmatrix, Tderivative, Tsolve, Tadjoint_solve, expected_roots
end

function triangular_analytic_tools(cases; component_scales=nothing, coupling=1.0)
    n = length(cases)
    scales = component_scales === nothing ? ones(Float64, n) : Float64.(component_scales)
    row_scale = Diagonal(ComplexF64.(1 ./ scales))
    N = zeros(ComplexF64, n, n)
    for i in 1:n-1
        N[i, i + 1] = coupling
    end

    function diagonal_values(z)
        ComplexF64[cases[i].f(z) for i in 1:n]
    end
    function diagonal_derivatives(z)
        ComplexF64[cases[i].df(z) for i in 1:n]
    end
    function Tmatrix(z)
        row_scale * (Diagonal(diagonal_values(z)) + N)
    end
    function Tderivative(z)
        row_scale * Diagonal(diagonal_derivatives(z))
    end
    function Tsolve(z, B)
        Tmatrix(z) \ B
    end
    function Tadjoint_solve(z, B)
        Tmatrix(z)' \ B
    end
    function expected_roots(center, radius)
        roots = ComplexF64[]
        for case in cases
            append!(roots, case.roots(center, radius))
        end
        unique_values(sort(roots; by=z -> (real(z), imag(z))); atol=1e-10)
    end
    Tmatrix, Tderivative, Tsolve, Tadjoint_solve, expected_roots
end

function triangular_operator_builder(; coupling=1.0)
    (cases; component_scales=nothing) -> triangular_analytic_tools(cases; component_scales=component_scales, coupling=coupling)
end

function reduced_analytic_determinant_extraction(
    Tmatrix,
    Tderivative,
    Xbasis,
    Ybasis,
    center,
    radius;
    determinant_nodes=2048,
    determinant_capacity=64,
    residual_normalization=:operator,
    refinement=:none,
    refinement_steps=4,
    refinement_nodes=256,
)
    size(Xbasis, 2) == size(Ybasis, 2) || error("reduced determinant extraction needs square left/right bases")
    function Tred(z)
        Ybasis' * Tmatrix(z) * Xbasis
    end
    function Tred_derivative(z)
        Ybasis' * Tderivative(z) * Xbasis
    end
    μ_roots, count_estimate, sums = determinant_power_sums(
        Tred,
        Tred_derivative;
        center=center,
        radius=radius,
        nodes=determinant_nodes,
        capacity=determinant_capacity,
    )
    values = center .+ radius .* μ_roots
    values = refine_determinant_roots(Tred, Tred_derivative, values; step_limit=0.25 * radius)
    finite = finite_eigenvalue_mask(values)
    values = ComplexF64.(values[finite])
    values, left_reduced, right_reduced, refinement_corrections, refinement_ratios =
        reduced_analytic_refined_left_right(
            Tred,
            Tred_derivative,
            values,
            center,
            radius;
            refinement=refinement,
            refinement_steps=refinement_steps,
            refinement_nodes=refinement_nodes,
        )
    right_vectors = Xbasis * right_reduced
    left_vectors = Ybasis * left_reduced
    normalize_columns_local!(right_vectors)
    normalize_columns_local!(left_vectors)
    right_residuals = matrix_vector_residuals(Tmatrix, values, right_vectors; normalization=residual_normalization)
    left_residuals = matrix_vector_residuals(Tmatrix, values, left_vectors; adjoint=true, normalization=residual_normalization)
    reduced_residuals = [reduced_nep_residual(Tred, value) for value in values]
    inside = FEASTSolver.in_contour(values, center, radius)
    (
        values=values,
        right_vectors=right_vectors,
        left_vectors=left_vectors,
        inside=inside,
        residuals=max.(right_residuals, left_residuals),
        right_residuals=right_residuals,
        left_residuals=left_residuals,
        reduced_residuals=reduced_residuals,
        count_estimate=count_estimate,
        count_error=abs(sums[1] - count_estimate),
        refinement=refinement,
        refinement_corrections=refinement_corrections,
        refinement_ratios=refinement_ratios,
    )
end

function reduced_analytic_ss_extraction(
    Tmatrix,
    Tderivative,
    Xbasis,
    Ybasis,
    center,
    radius;
    reduced_moments=8,
    reduced_nodes=256,
    ranktol=1e-10,
    maxrank=typemax(Int),
    count_estimate=nothing,
    count_error=NaN,
    ss_mode=:similarity,
    residual_normalization=:operator,
    refinement=:none,
    refinement_steps=4,
    refinement_nodes=256,
)
    size(Xbasis, 2) == size(Ybasis, 2) || error("reduced SS extraction needs square left/right bases")
    d = size(Xbasis, 2)
    function Tred(z)
        Ybasis' * Tmatrix(z) * Xbasis
    end
    function Tred_derivative(z)
        Ybasis' * Tderivative(z) * Xbasis
    end
    z_nodes, z_weights = circular_rule(center, radius, reduced_nodes)
    probe = Matrix{ComplexF64}(I, d, d)
    Tsolve_red = (z, B) -> Tred(z) \ B
    moments = initial_moments_generic_scaled(Tsolve_red, probe, z_nodes, z_weights, center, radius, reduced_moments)
    _, m = size(moments[1])
    H0 = zeros(ComplexF64, reduced_moments * d, reduced_moments * m)
    H1 = similar(H0)
    for i in 1:reduced_moments, j in 1:reduced_moments
        rows = (i - 1) * d + 1:i * d
        cols = (j - 1) * m + 1:j * m
        H0[rows, cols] .= moments[i + j - 1]
        H1[rows, cols] .= moments[i + j]
    end
    F = svd(H0)
    isempty(F.S) && error("empty reduced SS Hankel SVD")
    rank = count(F.S ./ F.S[1] .> ranktol)
    rank = min(rank, maxrank, length(F.S))
    rank > 0 || error("reduced SS Hankel rank is zero")
    U = F.U[:, 1:rank]
    V = F.V[:, 1:rank]
    if ss_mode === :similarity
        Sμ = U' * H1 * V * Diagonal(1 ./ F.S[1:rank])
        values = center .+ radius .* ComplexF64.(eigvals(Sμ))
    elseif ss_mode === :generalized
        values = center .+ radius .* ComplexF64.(eigvals(U' * H1 * V, U' * H0 * V))
    else
        error("unknown reduced SS mode: $ss_mode")
    end
    values = refine_determinant_roots(Tred, Tred_derivative, values; step_limit=0.25 * radius)
    finite = finite_eigenvalue_mask(values)
    values = ComplexF64.(values[finite])
    values, left_reduced, right_reduced, refinement_corrections, refinement_ratios =
        reduced_analytic_refined_left_right(
            Tred,
            Tred_derivative,
            values,
            center,
            radius;
            refinement=refinement,
            refinement_steps=refinement_steps,
            refinement_nodes=refinement_nodes,
        )
    right_vectors = Xbasis * right_reduced
    left_vectors = Ybasis * left_reduced
    normalize_columns_local!(right_vectors)
    normalize_columns_local!(left_vectors)
    right_residuals = matrix_vector_residuals(Tmatrix, values, right_vectors; normalization=residual_normalization)
    left_residuals = matrix_vector_residuals(Tmatrix, values, left_vectors; adjoint=true, normalization=residual_normalization)
    reduced_residuals = [reduced_nep_residual(Tred, value) for value in values]
    inside = FEASTSolver.in_contour(values, center, radius)
    reported_count = count_estimate === nothing ? rank : Int(count_estimate)
    (
        values=values,
        right_vectors=right_vectors,
        left_vectors=left_vectors,
        inside=inside,
        residuals=max.(right_residuals, left_residuals),
        right_residuals=right_residuals,
        left_residuals=left_residuals,
        reduced_residuals=reduced_residuals,
        count_estimate=reported_count,
        count_error=count_error,
        singular_values=Float64.(F.S),
        refinement=refinement,
        refinement_corrections=refinement_corrections,
        refinement_ratios=refinement_ratios,
    )
end

function run_fused_contour_sample_realization_diagnostic(;
    center=0.0 + 0.0im,
    radius=4.0,
    basis_nodes=256,
    moment_count=5,
    seed=20260503,
    ranktol=1e-10,
    basis_ranktol=0.1,
    residual_tol=1e-8,
    match_atol=1e-6,
    print_rows=true,
)
    cases = (scalar_sine_case(), scalar_cosine_case(), scalar_shifted_sine_case())
    expected = sorted_unique_values(
        reduce(vcat, (case.roots(center, radius) for case in cases));
        atol=match_atol,
    )
    n = length(cases)

    function Tmatrix(z)
        Diagonal(ComplexF64[case.f(z) for case in cases])
    end
    function Tderivative(z)
        Diagonal(ComplexF64[case.df(z) for case in cases])
    end
    function Tsolve(z, B)
        denom = ComplexF64[case.f(z) for case in cases]
        B ./ denom
    end
    function Tadjoint_solve(z, B)
        denom = conj.(ComplexF64[case.f(z) for case in cases])
        B ./ denom
    end

    Random.seed!(seed)
    Xprobe = rand(ComplexF64, n, n)
    Wprobe = rand(ComplexF64, n, n)
    chart = ContourChart(center, radius)
    cache = build_contour_sample_cache(
        Tsolve,
        Tadjoint_solve,
        Xprobe,
        Wprobe,
        chart,
        basis_nodes;
        source=:fused_analytic_diagnostic,
    )

    # One original contour-sample set supplies both the physical moments and
    # the small projected transfer moments W' T(z)^(-1) Xprobe.
    right_moment_blocks = right_moments(cache, moment_count)
    left_moment_blocks = left_moments(cache, moment_count)

    Xfused, Sfused, fused_rank, fused_singulars = projected_hankel_pair_identity(
        right_moment_blocks,
        Wprobe,
        moment_count;
        ranktol=ranktol,
        maxrank=length(expected),
    )
    Ffused = eigen(Sfused)
    fused_values = center .+ radius .* ComplexF64.(Ffused.values)
    fused_vectors = Xfused * Ffused.vectors
    normalize_columns_local!(fused_vectors)
    fused_residuals = matrix_vector_residuals(Tmatrix, fused_values, fused_vectors)
    fused_inside = FEASTSolver.in_contour(fused_values, center, radius)
    fused_good = fused_inside .& (fused_residuals .<= residual_tol)
    fused_matched = match_expected_count(fused_values[fused_good], expected; atol=match_atol)

    # Redundant path: build physical trial/test spaces, then resample and solve
    # the projected nonlinear problem with an inner SS contour extraction.
    Xbasis, right_singulars = moment_block_basis(right_moment_blocks, moment_count; ranktol=basis_ranktol)
    Ybasis, left_singulars = moment_block_basis(left_moment_blocks, moment_count; ranktol=basis_ranktol)
    d = min(size(Xbasis, 2), size(Ybasis, 2))
    old = reduced_analytic_ss_extraction(
        Tmatrix,
        Tderivative,
        Xbasis[:, 1:d],
        Ybasis[:, 1:d],
        center,
        radius;
        reduced_moments=moment_count,
        reduced_nodes=basis_nodes,
        ranktol=ranktol,
        maxrank=length(expected),
        count_estimate=length(expected),
        residual_normalization=:operator,
        refinement=:none,
    )
    old_good = old.inside .& (old.residuals .<= residual_tol)
    old_matched = match_expected_count(old.values[old_good], expected; atol=match_atol)
    cross_match = match_expected_count(fused_values[fused_good], old.values[old_good]; atol=match_atol)

    if print_rows
        println("Fused contour-sample realization diagnostic")
        println("  problem=sin/cos/shifted-sin diagonal; center=$center radius=$radius")
        println("  compares one original contour sample cache against redundant inner reduced SS")
        @printf(
            "  expected=%d fused(rank=%d good=%d matched=%d maxres=%.3e) old(good=%d matched=%d maxres=%.3e) cross=%d\n",
            length(expected),
            fused_rank,
            count(fused_good),
            fused_matched,
            any(fused_good) ? maximum(fused_residuals[fused_good]) : Inf,
            count(old_good),
            old_matched,
            any(old_good) ? maximum(old.residuals[old_good]) : Inf,
            cross_match,
        )
        @printf(
            "  basis_dims=(%d,%d) basis_sigma=(%.3e, %.3e) fused_sigma=%.3e\n",
            size(Xbasis, 2),
            size(Ybasis, 2),
            isempty(right_singulars) ? NaN : right_singulars[end] / right_singulars[1],
            isempty(left_singulars) ? NaN : left_singulars[end] / left_singulars[1],
            isempty(fused_singulars) ? NaN : fused_singulars[min(end, fused_rank)] / fused_singulars[1],
        )
    end

    (
        expected=length(expected),
        fused_values=fused_values,
        fused_residuals=fused_residuals,
        fused_good=count(fused_good),
        fused_matched=fused_matched,
        fused_rank=fused_rank,
        old_values=old.values,
        old_residuals=old.residuals,
        old_good=count(old_good),
        old_matched=old_matched,
        cross_match=cross_match,
        basis_dims=(right=size(Xbasis, 2), left=size(Ybasis, 2)),
        fused_singulars=Float64.(fused_singulars),
    )
end

function run_fused_polynomial_contour_sample_realization_diagnostic(;
    basis_nodes=96,
    moment_count=3,
    seed=20260504,
    ranktol=1e-10,
    basis_ranktol=1e-10,
    residual_tol=1e-8,
    match_atol=1e-6,
    print_rows=true,
)
    coeffs, center, radius, probe_cols = deficient_quadratic_problem()
    expected = companion_reference(coeffs, center, radius)
    n = size(coeffs[1], 1)

    Tmatrix = z -> polynomial_matrix(coeffs, z)
    Tderivative = z -> begin
        M = zeros(ComplexF64, n, n)
        power = one(ComplexF64)
        for j in 2:length(coeffs)
            M .+= (j - 1) * power .* coeffs[j]
            power *= z
        end
        M
    end

    Random.seed!(seed)
    Xprobe = rand(ComplexF64, n, max(probe_cols, length(expected)))
    Wprobe = rand(ComplexF64, n, max(probe_cols, length(expected)))
    Tsolve = (z, B) -> polynomial_matrix(coeffs, z) \ B
    Tadjoint_solve = (z, B) -> polynomial_matrix(coeffs, z)' \ B
    chart = ContourChart(center, radius)
    cache = build_contour_sample_cache(
        Tsolve,
        Tadjoint_solve,
        Xprobe,
        Wprobe,
        chart,
        basis_nodes;
        source=:fused_polynomial_diagnostic,
    )
    right_moment_blocks = right_moments(cache, moment_count)
    left_moment_blocks = left_moments(cache, moment_count)

    Xfused, Sfused, fused_rank, fused_singulars = projected_hankel_pair_identity(
        right_moment_blocks,
        Wprobe,
        moment_count;
        ranktol=ranktol,
        maxrank=length(expected),
    )
    Ffused = eigen(Sfused)
    fused_values = center .+ radius .* ComplexF64.(Ffused.values)
    fused_vectors = Xfused * Ffused.vectors
    normalize_columns_local!(fused_vectors)
    fused_residuals = polynomial_vector_residuals(coeffs, fused_values, fused_vectors)
    fused_inside = FEASTSolver.in_contour(fused_values, center, radius)
    fused_good = fused_inside .& (fused_residuals .<= residual_tol)
    fused_matched = match_expected_count(fused_values[fused_good], expected; atol=match_atol)

    Xbasis, right_singulars = moment_block_basis(right_moment_blocks, moment_count; ranktol=basis_ranktol)
    Ybasis, left_singulars = moment_block_basis(left_moment_blocks, moment_count; ranktol=basis_ranktol)
    d = min(size(Xbasis, 2), size(Ybasis, 2))
    old = reduced_analytic_ss_extraction(
        Tmatrix,
        Tderivative,
        Xbasis[:, 1:d],
        Ybasis[:, 1:d],
        center,
        radius;
        reduced_moments=moment_count,
        reduced_nodes=basis_nodes,
        ranktol=ranktol,
        maxrank=length(expected),
        count_estimate=length(expected),
        residual_normalization=:operator,
        refinement=:none,
    )
    old_good = old.inside .& (old.residuals .<= residual_tol)
    old_matched = match_expected_count(old.values[old_good], expected; atol=match_atol)
    cross_match = match_expected_count(fused_values[fused_good], old.values[old_good]; atol=match_atol)

    if print_rows
        println("Fused polynomial contour-sample realization diagnostic")
        println("  problem=deficient quadratic; center=$center radius=$radius")
        println("  compares original projected polynomial moments against redundant inner reduced SS")
        @printf(
            "  expected=%d fused(rank=%d good=%d matched=%d maxres=%.3e) old(good=%d matched=%d maxres=%.3e) cross=%d\n",
            length(expected),
            fused_rank,
            count(fused_good),
            fused_matched,
            any(fused_good) ? maximum(fused_residuals[fused_good]) : Inf,
            count(old_good),
            old_matched,
            any(old_good) ? maximum(old.residuals[old_good]) : Inf,
            cross_match,
        )
        @printf(
            "  basis_dims=(%d,%d) basis_sigma=(%.3e, %.3e) fused_sigma=%.3e\n",
            size(Xbasis, 2),
            size(Ybasis, 2),
            isempty(right_singulars) ? NaN : right_singulars[min(end, size(Xbasis, 2))] / right_singulars[1],
            isempty(left_singulars) ? NaN : left_singulars[min(end, size(Ybasis, 2))] / left_singulars[1],
            isempty(fused_singulars) ? NaN : fused_singulars[min(end, fused_rank)] / fused_singulars[1],
        )
    end

    (
        expected=length(expected),
        fused_values=fused_values,
        fused_residuals=fused_residuals,
        fused_good=count(fused_good),
        fused_matched=fused_matched,
        fused_rank=fused_rank,
        old_values=old.values,
        old_residuals=old.residuals,
        old_good=count(old_good),
        old_matched=old_matched,
        cross_match=cross_match,
        basis_dims=(right=size(Xbasis, 2), left=size(Ybasis, 2)),
        fused_singulars=Float64.(fused_singulars),
    )
end

function run_fused_cache_residual_augmentation_diagnostic(;
    basis_nodes=96,
    moment_count=3,
    seed=20260505,
    ranktol=1e-10,
    basis_ranktol=1e-10,
    residual_ranktol=1e-10,
    compression_ranktol=1e-10,
    residual_tol=1e-8,
    match_atol=1e-6,
    print_rows=true,
)
    coeffs, center, radius, probe_cols = deficient_quadratic_problem()
    expected = companion_reference(coeffs, center, radius)
    n = size(coeffs[1], 1)
    chart = ContourChart(center, radius)
    Tsolve = (z, B) -> polynomial_matrix(coeffs, z) \ B
    Tadjoint_solve = (z, B) -> polynomial_matrix(coeffs, z)' \ B

    Random.seed!(seed)
    Xprobe = rand(ComplexF64, n, max(probe_cols, length(expected)))
    Wprobe = rand(ComplexF64, n, max(probe_cols, length(expected)))
    cache = build_contour_sample_cache(
        Tsolve,
        Tadjoint_solve,
        Xprobe,
        Wprobe,
        chart,
        basis_nodes;
        source=:fused_residual_augmentation_base,
    )
    right_base_moments = right_moments(cache, moment_count)
    left_base_moments = left_moments(cache, moment_count)
    Xbasis, _ = moment_block_basis(right_base_moments, moment_count; ranktol=basis_ranktol)
    Ybasis, _ = moment_block_basis(left_base_moments, moment_count; ranktol=basis_ranktol)
    d = min(size(Xbasis, 2), size(Ybasis, 2))
    Xbasis = Xbasis[:, 1:d]
    Ybasis = Ybasis[:, 1:d]

    extraction = reduced_analytic_ss_extraction(
        z -> polynomial_matrix(coeffs, z),
        z -> begin
            M = zeros(ComplexF64, n, n)
            power = one(ComplexF64)
            for j in 2:length(coeffs)
                M .+= (j - 1) * power .* coeffs[j]
                power *= z
            end
            M
        end,
        Xbasis,
        Ybasis,
        center,
        radius;
        reduced_moments=moment_count,
        reduced_nodes=basis_nodes,
        ranktol=ranktol,
        maxrank=length(expected),
        count_estimate=length(expected),
        residual_normalization=:operator,
        refinement=:none,
    )
    Rright_basis, Rleft_basis, right_residual_singulars, left_residual_singulars =
        residual_blocks_from_matrix_extraction(z -> polynomial_matrix(coeffs, z), extraction; residual_ranktol=residual_ranktol)

    z_nodes, z_weights = circular_rule(center, radius, basis_nodes)
    right_update_moments, left_update_moments = residual_laurent_moment_blocks_generic(
        Tsolve,
        Tadjoint_solve,
        Rright_basis,
        Rleft_basis,
        z_nodes,
        z_weights,
        center,
        radius;
        moment_count=moment_count,
    )
    Xupdate, Yupdate, _, _, _, _ = compress_residual_laurent_candidates(
        Xbasis,
        Ybasis,
        right_update_moments,
        left_update_moments;
        compression_ranktol=compression_ranktol,
    )

    augmented = augment_contour_sample_cache(
        cache,
        Tsolve,
        Tadjoint_solve,
        Rright_basis,
        Rleft_basis;
        source=:fused_residual_augmentation,
    )
    right_range = (size(cache.right_probe, 2) + 1):size(augmented.right_probe, 2)
    left_range = (size(cache.left_probe, 2) + 1):size(augmented.left_probe, 2)
    right_cache_update_moments, left_cache_update_moments =
        residual_laurent_moments(augmented, right_range, left_range, moment_count)
    Xcache, Ycache, _, _, _, _ = compress_residual_laurent_candidates(
        Xbasis,
        Ybasis,
        right_cache_update_moments,
        left_cache_update_moments;
        compression_ranktol=compression_ranktol,
    )

    x_gap = subspace_projection_gap(Xupdate, Xcache)
    y_gap = subspace_projection_gap(Yupdate, Ycache)
    base_good = extraction.inside .& (extraction.residuals .<= residual_tol)
    base_matched = match_expected_count(extraction.values[base_good], expected; atol=match_atol)

    if print_rows
        println("Fused cache residual augmentation diagnostic")
        println("  problem=deficient quadratic; compares explicit residual-Laurent moments against cache augmentation")
        @printf(
            "  expected=%d base_matched=%d residual_ranks=(%d,%d) dims update=(%d,%d) cache=(%d,%d) gaps=(%.3e, %.3e)\n",
            length(expected),
            base_matched,
            size(Rright_basis, 2),
            size(Rleft_basis, 2),
            size(Xupdate, 2),
            size(Yupdate, 2),
            size(Xcache, 2),
            size(Ycache, 2),
            x_gap,
            y_gap,
        )
    end

    (
        expected=length(expected),
        base_matched=base_matched,
        right_residual_rank=size(Rright_basis, 2),
        left_residual_rank=size(Rleft_basis, 2),
        right_residual_singulars=Float64.(right_residual_singulars),
        left_residual_singulars=Float64.(left_residual_singulars),
        update_dims=(right=size(Xupdate, 2), left=size(Yupdate, 2)),
        cache_dims=(right=size(Xcache, 2), left=size(Ycache, 2)),
        x_projection_gap=x_gap,
        y_projection_gap=y_gap,
    )
end

function loewner_interpolation_points(count; radius=1.6, phase=0.0)
    ComplexF64[radius * exp(im * (phase + 2pi * (j - 1) / count)) for j in 1:count]
end

function contour_transfer_samples(Tred, alpha_points, center, radius; nodes=256)
    z_nodes, z_weights = circular_rule(center, radius, nodes)
    d = size(Tred(z_nodes[1]), 1)
    samples = [zeros(ComplexF64, d, d) for _ in alpha_points]
    identity = Matrix{ComplexF64}(I, d, d)
    for (z, weight) in zip(z_nodes, z_weights)
        ζ = (z - center) / radius
        solved = Tred(z) \ identity
        for (j, α) in pairs(alpha_points)
            samples[j] .+= (weight / (α - ζ)) .* solved
        end
    end
    samples
end

function block_loewner_pencil(left_points, left_samples, right_points, right_samples)
    length(left_points) == length(left_samples) || error("left point/sample count mismatch")
    length(right_points) == length(right_samples) || error("right point/sample count mismatch")
    d = size(first(left_samples), 1)
    L = zeros(ComplexF64, length(left_points) * d, length(right_points) * d)
    Ls = similar(L)
    for (i, μ) in pairs(left_points), (j, λ) in pairs(right_points)
        abs(μ - λ) > eps(Float64) || error("Loewner left/right interpolation points must be distinct")
        rows = (i - 1) * d + 1:i * d
        cols = (j - 1) * d + 1:j * d
        denominator = μ - λ
        L[rows, cols] .= (left_samples[i] .- right_samples[j]) ./ denominator
        Ls[rows, cols] .= (μ .* left_samples[i] .- λ .* right_samples[j]) ./ denominator
    end
    L, Ls
end

function reduced_analytic_loewner_extraction(
    Tmatrix,
    Tderivative,
    Xbasis,
    Ybasis,
    center,
    radius;
    reduced_nodes=256,
    ranktol=1e-10,
    maxrank=typemax(Int),
    loewner_points=4,
    loewner_radius=1.6,
    loewner_phase=0.0,
    count_estimate=nothing,
    count_error=NaN,
    residual_normalization=:operator,
    refinement=:none,
    refinement_steps=4,
    refinement_nodes=256,
)
    size(Xbasis, 2) == size(Ybasis, 2) || error("reduced Loewner extraction needs square left/right bases")
    function Tred(z)
        Ybasis' * Tmatrix(z) * Xbasis
    end
    function Tred_derivative(z)
        Ybasis' * Tderivative(z) * Xbasis
    end

    left_points = loewner_interpolation_points(loewner_points; radius=loewner_radius, phase=loewner_phase)
    right_points = loewner_interpolation_points(loewner_points; radius=loewner_radius, phase=loewner_phase + pi / loewner_points)
    left_samples = contour_transfer_samples(Tred, left_points, center, radius; nodes=reduced_nodes)
    right_samples = contour_transfer_samples(Tred, right_points, center, radius; nodes=reduced_nodes)
    L, Ls = block_loewner_pencil(left_points, left_samples, right_points, right_samples)

    F = svd(L)
    isempty(F.S) && error("empty reduced Loewner SVD")
    rank = count(F.S ./ F.S[1] .> ranktol)
    rank = min(rank, maxrank, length(F.S))
    rank > 0 || error("reduced Loewner rank is zero")
    U = F.U[:, 1:rank]
    V = F.V[:, 1:rank]
    Sμ = U' * Ls * V * Diagonal(1 ./ F.S[1:rank])
    values = center .+ radius .* ComplexF64.(eigvals(Sμ))
    values = refine_determinant_roots(Tred, Tred_derivative, values; step_limit=0.25 * radius)
    finite = finite_eigenvalue_mask(values)
    values = ComplexF64.(values[finite])
    values, left_reduced, right_reduced, refinement_corrections, refinement_ratios =
        reduced_analytic_refined_left_right(
            Tred,
            Tred_derivative,
            values,
            center,
            radius;
            refinement=refinement,
            refinement_steps=refinement_steps,
            refinement_nodes=refinement_nodes,
        )
    right_vectors = Xbasis * right_reduced
    left_vectors = Ybasis * left_reduced
    normalize_columns_local!(right_vectors)
    normalize_columns_local!(left_vectors)
    right_residuals = matrix_vector_residuals(Tmatrix, values, right_vectors; normalization=residual_normalization)
    left_residuals = matrix_vector_residuals(Tmatrix, values, left_vectors; adjoint=true, normalization=residual_normalization)
    reduced_residuals = [reduced_nep_residual(Tred, value) for value in values]
    inside = FEASTSolver.in_contour(values, center, radius)
    reported_count = count_estimate === nothing ? rank : Int(count_estimate)
    (
        values=values,
        right_vectors=right_vectors,
        left_vectors=left_vectors,
        inside=inside,
        residuals=max.(right_residuals, left_residuals),
        right_residuals=right_residuals,
        left_residuals=left_residuals,
        reduced_residuals=reduced_residuals,
        count_estimate=reported_count,
        count_error=count_error,
        singular_values=Float64.(F.S),
        loewner_left_points=left_points,
        loewner_right_points=right_points,
        refinement=refinement,
        refinement_corrections=refinement_corrections,
        refinement_ratios=refinement_ratios,
    )
end

function reduced_analytic_extraction(
    Tmatrix,
    Tderivative,
    Xbasis,
    Ybasis,
    center,
    radius;
    extractor=:determinant,
    determinant_nodes=2048,
    determinant_capacity=64,
    reduced_moments=8,
    reduced_nodes=256,
    reduced_ranktol=1e-10,
    reduced_maxrank=typemax(Int),
    reduced_ss_mode=:similarity,
    loewner_points=4,
    loewner_radius=1.6,
    loewner_phase=0.0,
    residual_normalization=:operator,
    refinement=:none,
    refinement_steps=4,
    refinement_nodes=256,
)
    if extractor === :determinant
        return reduced_analytic_determinant_extraction(
            Tmatrix,
            Tderivative,
            Xbasis,
            Ybasis,
            center,
            radius;
            determinant_nodes=determinant_nodes,
            determinant_capacity=determinant_capacity,
            residual_normalization=residual_normalization,
            refinement=refinement,
            refinement_steps=refinement_steps,
            refinement_nodes=refinement_nodes,
        )
    elseif extractor === :ss_hankel
        return reduced_analytic_ss_extraction(
            Tmatrix,
            Tderivative,
            Xbasis,
            Ybasis,
            center,
            radius;
            reduced_moments=reduced_moments,
            reduced_nodes=reduced_nodes,
            ranktol=reduced_ranktol,
            maxrank=reduced_maxrank,
            ss_mode=reduced_ss_mode,
            residual_normalization=residual_normalization,
            refinement=refinement,
            refinement_steps=refinement_steps,
            refinement_nodes=refinement_nodes,
        )
    elseif extractor === :loewner
        return reduced_analytic_loewner_extraction(
            Tmatrix,
            Tderivative,
            Xbasis,
            Ybasis,
            center,
            radius;
            reduced_nodes=reduced_nodes,
            ranktol=reduced_ranktol,
            maxrank=reduced_maxrank,
            loewner_points=loewner_points,
            loewner_radius=loewner_radius,
            loewner_phase=loewner_phase,
            residual_normalization=residual_normalization,
            refinement=refinement,
            refinement_steps=refinement_steps,
            refinement_nodes=refinement_nodes,
        )
    elseif extractor === :ss_counted
        Tred = z -> Ybasis' * Tmatrix(z) * Xbasis
        Tred_derivative = z -> Ybasis' * Tderivative(z) * Xbasis
        _, count_estimate, sums = determinant_power_sums(
            Tred,
            Tred_derivative;
            center=center,
            radius=radius,
            nodes=determinant_nodes,
            capacity=determinant_capacity,
        )
        return reduced_analytic_ss_extraction(
            Tmatrix,
            Tderivative,
            Xbasis,
            Ybasis,
            center,
            radius;
            reduced_moments=reduced_moments,
            reduced_nodes=reduced_nodes,
            ranktol=0.0,
            maxrank=count_estimate,
            count_estimate=count_estimate,
            count_error=abs(sums[1] - count_estimate),
            ss_mode=reduced_ss_mode,
            residual_normalization=residual_normalization,
            refinement=refinement,
            refinement_steps=refinement_steps,
            refinement_nodes=refinement_nodes,
        )
    elseif extractor === :loewner_counted
        Tred = z -> Ybasis' * Tmatrix(z) * Xbasis
        Tred_derivative = z -> Ybasis' * Tderivative(z) * Xbasis
        _, count_estimate, sums = determinant_power_sums(
            Tred,
            Tred_derivative;
            center=center,
            radius=radius,
            nodes=determinant_nodes,
            capacity=determinant_capacity,
        )
        effective_loewner_points = max(loewner_points, ceil(Int, count_estimate / max(size(Xbasis, 2), 1)))
        return reduced_analytic_loewner_extraction(
            Tmatrix,
            Tderivative,
            Xbasis,
            Ybasis,
            center,
            radius;
            reduced_nodes=reduced_nodes,
            ranktol=0.0,
            maxrank=count_estimate,
            count_estimate=count_estimate,
            count_error=abs(sums[1] - count_estimate),
            loewner_points=effective_loewner_points,
            loewner_radius=loewner_radius,
            loewner_phase=loewner_phase,
            residual_normalization=residual_normalization,
            refinement=refinement,
            refinement_steps=refinement_steps,
            refinement_nodes=refinement_nodes,
        )
    end
    error("unknown reduced analytic extractor: $extractor")
end

function run_dual_reduced_determinant_diagonal_stress(;
    cases=(scalar_sine_case(), scalar_cosine_case()),
    radii=(10.0, 20.0),
    basis_moments=4,
    basis_ranktol=1e-10,
    determinant_nodes=2048,
    determinant_capacity=64,
    residual_tol=1e-8,
)
    center = 0.0 + 0.0im
    Tsolve, _, _, expected_roots = diagonal_analytic_tools(cases)
    n = length(cases)
    labels = join((case.name for case in cases), ",")

    println()
    println("Dual reduced determinant extraction stress: [$labels]")
    println("  left/right moment bases feed an argument-principle solve of det(Y' T(lambda) X)")
    for radius in radii
        expected = expected_roots(center, radius)
        z_nodes, z_weights = circular_rule(center, radius, max(128, 16 * basis_moments))
        Random.seed!(9401 + round(Int, radius * 10))
        Xprobe = rand(ComplexF64, n, n)
        Wprobe = rand(ComplexF64, n, n)
        right_moments = initial_moments_generic_scaled(Tsolve, Xprobe, z_nodes, z_weights, center, radius, basis_moments)
        left_moments = initial_adjoint_moments_diagonal_scaled(cases, Wprobe, z_nodes, z_weights, center, radius, basis_moments)
        Xbasis, right_singulars = moment_block_basis(right_moments, basis_moments; ranktol=basis_ranktol)
        Ybasis, left_singulars = moment_block_basis(left_moments, basis_moments; ranktol=basis_ranktol)
        Xbi, Ybi, cross_singulars = biorthogonalize_bases(Xbasis, Ybasis)
        basis_cases = (
            ("dual", Xbasis, Ybasis),
            ("dual_biorth", Xbi, Ybi),
            ("galerkin", Xbasis, Xbasis),
        )
        for (label, Xtest, Ytest) in basis_cases
            if size(Xtest, 2) == 0 || size(Ytest, 2) == 0 || size(Xtest, 2) != size(Ytest, 2)
                println("  radius=$radius $label skipped: incompatible reduced basis sizes")
                continue
            end
            Tred, Tred_derivative = diagonal_reduced_operators(cases, Xtest, Ytest)
            μ_roots, count_estimate, sums = determinant_power_sums(
                Tred,
                Tred_derivative;
                center=center,
                radius=radius,
                nodes=determinant_nodes,
                capacity=determinant_capacity,
            )
            λ_roots = center .+ radius .* μ_roots
            λ_roots = refine_determinant_roots(Tred, Tred_derivative, λ_roots; step_limit=0.25 * radius)
            inside = FEASTSolver.in_contour(λ_roots, center, radius)
            residuals = [reduced_nep_residual(Tred, λ) for λ in λ_roots]
            reduced_vectors = reduced_right_singular_vectors(Tred, λ_roots)
            full_vectors = Xtest * reduced_vectors
            full_residuals = diagonal_full_residuals(cases, λ_roots, full_vectors)
            good = inside .& (full_residuals .<= residual_tol)
            matched = match_expected_count(λ_roots[good], expected; atol=1e-6)
            count_error = abs(sums[1] - count_estimate)
            @printf(
                "  radius=%.3g mode=%s expected=%d count=%d count_err=%.3e basis=(%d,%d) good=%d/%d matched=%d red_max=%.3e full_max=%.3e\n",
                radius,
                label,
                length(expected),
                count_estimate,
                count_error,
                size(Xtest, 2),
                size(Ytest, 2),
                count(good),
                count(inside),
                matched,
                any(inside) ? maximum(residuals[inside]) : Inf,
                any(inside) ? maximum(full_residuals[inside]) : Inf,
            )
        end
        @printf(
            "    basis singular ratios right=%.3e left=%.3e cross=%.3e\n",
            isempty(right_singulars) ? NaN : right_singulars[end] / right_singulars[1],
            isempty(left_singulars) ? NaN : left_singulars[end] / left_singulars[1],
            isempty(cross_singulars) ? NaN : cross_singulars[end] / cross_singulars[1],
        )
    end
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

function dual_sensitive_polynomial_problem()
    n = 6
    degree = 6
    center = 0.0 + 0.0im
    radius = 2.0
    inside_sets = [
        ComplexF64[-1.5, -0.6, 0.3, 1.2],
        ComplexF64[-1.2, -0.3, 0.6, 1.5],
        ComplexF64[-0.9, -0.1, 0.9, 1.7],
    ]
    outside_sets = [
        ComplexF64[4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        ComplexF64[4.3, 5.3, 6.3, 7.3, 8.3, 9.3],
        ComplexF64[4.6, 5.6, 6.6, 7.6, 8.6, 9.6],
    ]
    roots_by_direction = [
        vcat(inside_sets[1], ComplexF64[6.0, 8.0]),
        vcat(inside_sets[2], ComplexF64[6.5, 8.5]),
        vcat(inside_sets[3], ComplexF64[7.0, 9.0]),
        outside_sets[1],
        outside_sets[2],
        outside_sets[3],
    ]
    scalar_coeffs = [polynomial_coefficients_from_roots(roots) for roots in roots_by_direction]
    # A moderately ill-conditioned Vandermonde similarity keeps the exact roots
    # known while making one-sided Galerkin extraction admit spurious roots.
    nodes = 1.0 .+ 0.2 .* (0:n - 1)
    V = ComplexF64[nodes[i]^(j - 1) for i in 1:n, j in 1:n]
    V[:, 2] .+= 0.05im .* V[:, 1]
    Vinv = inv(V)
    coeffs = Matrix{ComplexF64}[]
    for j in 1:(degree + 1)
        push!(coeffs, V * Diagonal([scalar_coeffs[i][j] for i in 1:n]) * Vinv)
    end
    expected = sort(vcat(inside_sets...); by=z -> (real(z), imag(z)))
    coeffs, center, radius, n, expected
end

function initial_adjoint_moments_polynomial_scaled(coeffs, Wprobe, z_nodes, z_weights, center, radius, moment_count)
    n, m = size(Wprobe)
    moments = [zeros(ComplexF64, n, m) for _ in 1:moment_count]
    for (z, weight) in zip(z_nodes, z_weights)
        solved = polynomial_matrix(coeffs, z)' \ Wprobe
        μ = conj((z - center) / radius)
        μpower = one(ComplexF64)
        for p in eachindex(moments)
            moments[p] .+= (conj(weight) * μpower) .* solved
            μpower *= μ
        end
    end
    moments
end

function polynomial_vector_residuals(coeffs, values, vectors)
    residuals = zeros(Float64, length(values))
    for j in eachindex(values)
        v = vectors[:, j]
        vnorm = norm(v)
        if vnorm <= eps(Float64)
            residuals[j] = Inf
        else
            Tλ = polynomial_matrix(coeffs, values[j])
            residuals[j] = norm(Tλ * v) / (max(norm(Tλ), eps(Float64)) * vnorm)
        end
    end
    residuals
end

function polynomial_left_vector_residuals(coeffs, values, vectors)
    residuals = zeros(Float64, length(values))
    for j in eachindex(values)
        v = vectors[:, j]
        vnorm = norm(v)
        if vnorm <= eps(Float64)
            residuals[j] = Inf
        else
            Tλ = polynomial_matrix(coeffs, values[j])
            residuals[j] = norm(Tλ' * v) / (max(norm(Tλ), eps(Float64)) * vnorm)
        end
    end
    residuals
end

function normalize_columns_local!(X)
    for j in axes(X, 2)
        xnorm = norm(view(X, :, j))
        xnorm > eps(Float64) && (X[:, j] ./= xnorm)
    end
    X
end

function reduced_polynomial_newton_refine(
    reduced_coeffs,
    values,
    right_reduced,
    center,
    radius;
    steps=1,
    lift=nothing,
    selection=:inside,
)
    selected = if selection === :inside
        FEASTSolver.in_contour(values, center, radius)
    elseif selection === :all
        trues(length(values))
    else
        error("unknown reduced polynomial refinement selection: $selection")
    end
    any(selected) || return values, right_reduced, Float64[]

    nred = size(right_reduced, 1)
    p = count(selected)
    # The lifted gauge must have enough rows to represent p nonlinear states,
    # even when the reduced physical dimension is smaller than the root count.
    effective_lift = lift === nothing ? max(1, ceil(Int, p / max(nred, 1))) : Int(lift)
    effective_lift = max(effective_lift, ceil(Int, p / max(nred, 1)))

    X = Matrix(right_reduced[:, selected])
    S = Matrix(Diagonal(ComplexF64.(values[selected])))
    X, S = normalize_lifted_pair(X, S, effective_lift)
    X, S, ratios = invariant_pair_newton_refine(reduced_coeffs, X, S; lift=effective_lift, steps=steps)
    F = eigen(S)

    refined_values = copy(values)
    refined_vectors = copy(right_reduced)
    refined_values[selected] .= ComplexF64.(F.values)
    refined_vectors[:, selected] .= X * F.vectors
    refined_values, refined_vectors, ratios
end

function reduced_polynomial_extraction(
    coeffs,
    Xbasis,
    Ybasis,
    center,
    radius;
    refinement=:none,
    newton_steps=1,
    newton_lift=nothing,
    newton_selection=:inside,
)
    size(Xbasis, 2) == size(Ybasis, 2) || error("reduced extraction needs square left/right bases")
    reduced_coeffs = [Ybasis' * A * Xbasis for A in coeffs]
    values, _, reduced_residuals = companion(reduced_coeffs)
    finite = finite_eigenvalue_mask(values)
    values = ComplexF64.(values[finite])
    reduced_residuals = reduced_residuals[finite]

    k = length(values)
    nred = size(Xbasis, 2)
    Vred = zeros(ComplexF64, nred, k)
    for j in eachindex(values)
        Tλ = polynomial_matrix(reduced_coeffs, values[j])
        F = svd(Tλ)
        Vred[:, j] .= F.V[:, end]
    end
    refinement_ratios = Float64[]
    if refinement === :block_newton
        values, Vred, refinement_ratios = reduced_polynomial_newton_refine(
            reduced_coeffs,
            values,
            Vred,
            center,
            radius;
            steps=newton_steps,
            lift=newton_lift,
            selection=newton_selection,
        )
    elseif refinement !== :none
        error("unknown reduced polynomial refinement: $refinement")
    end

    Ured = zeros(ComplexF64, nred, k)
    reduced_residuals = zeros(Float64, k)
    for j in eachindex(values)
        Tλ = polynomial_matrix(reduced_coeffs, values[j])
        F = svd(Tλ)
        Ured[:, j] .= F.U[:, end]
        reduced_residuals[j] = isempty(F.S) ? Inf : minimum(F.S) / max(maximum(F.S), eps(Float64))
    end

    Xfull = Xbasis * Vred
    Yfull = Ybasis * Ured
    normalize_columns_local!(Xfull)
    normalize_columns_local!(Yfull)
    right_residuals = polynomial_vector_residuals(coeffs, values, Xfull)
    left_residuals = polynomial_left_vector_residuals(coeffs, values, Yfull)
    combined_residuals = max.(right_residuals, left_residuals)
    inside = FEASTSolver.in_contour(values, center, radius)
    (
        values=values,
        right_vectors=Xfull,
        left_vectors=Yfull,
        inside=inside,
        residuals=combined_residuals,
        right_residuals=right_residuals,
        left_residuals=left_residuals,
        reduced_residuals=reduced_residuals,
        refinement=refinement,
        refinement_ratios=refinement_ratios,
    )
end

function physical_basis_from_columns(X; ranktol=1e-10)
    F = svd(X)
    isempty(F.S) && return zeros(ComplexF64, size(X, 1), 0), Float64[]
    rank = count(F.S ./ F.S[1] .> ranktol)
    rank = min(rank, size(X, 1), length(F.S))
    F.U[:, 1:rank], copy(F.S)
end

function subspace_projection_gap(A, B)
    if size(A, 2) == 0 && size(B, 2) == 0
        return 0.0
    elseif size(A, 2) == 0 || size(B, 2) == 0
        return 1.0
    end
    QA = Matrix(qr(A).Q)[:, 1:size(A, 2)]
    QB = Matrix(qr(B).Q)[:, 1:size(B, 2)]
    norm(QA * QA' - QB * QB')
end

function low_rank_column_factor(A; ranktol=1e-12)
    F = svd(A)
    if isempty(F.S) || F.S[1] <= eps(Float64)
        return zeros(ComplexF64, size(A, 1), 0), zeros(ComplexF64, 0, size(A, 2)), Float64[]
    end
    rank = count(F.S ./ F.S[1] .> ranktol)
    rank = min(rank, size(A, 1), size(A, 2), length(F.S))
    basis = F.U[:, 1:rank]
    coeffs = Diagonal(F.S[1:rank]) * F.V[:, 1:rank]'
    basis, coeffs, copy(F.S)
end

function dual_scalar_rii_step_polynomial(coeffs, values, Xright, Xleft, z_nodes, z_weights; residual_ranktol=0.0)
    n, k = size(Xright)
    Rright = zeros(ComplexF64, n, k)
    Rleft = zeros(ComplexF64, n, k)
    for j in eachindex(values)
        Tλ = polynomial_matrix(coeffs, values[j])
        Rright[:, j] .= Tλ * Xright[:, j]
        Rleft[:, j] .= Tλ' * Xleft[:, j]
    end

    if residual_ranktol > 0
        Rright_basis, Rright_coeffs, right_singulars = low_rank_column_factor(Rright; ranktol=residual_ranktol)
        Rleft_basis, Rleft_coeffs, left_singulars = low_rank_column_factor(Rleft; ranktol=residual_ranktol)
    else
        Rright_basis, Rright_coeffs, right_singulars = Rright, Matrix{ComplexF64}(I, k, k), svdvals(Rright)
        Rleft_basis, Rleft_coeffs, left_singulars = Rleft, Matrix{ComplexF64}(I, k, k), svdvals(Rleft)
    end

    Qright = zeros(ComplexF64, n, k)
    Qleft = zeros(ComplexF64, n, k)
    for (z, weight) in zip(z_nodes, z_weights)
        Tz = polynomial_matrix(coeffs, z)
        solved_right = isempty(Rright_basis) ? zeros(ComplexF64, n, k) : (Tz \ Rright_basis) * Rright_coeffs
        solved_left = isempty(Rleft_basis) ? zeros(ComplexF64, n, k) : (Tz' \ Rleft_basis) * Rleft_coeffs
        for j in eachindex(values)
            αright = weight / (z - values[j])
            αleft = conj(weight) / conj(z - values[j])
            Qright[:, j] .+= αright .* (Xright[:, j] .- solved_right[:, j])
            Qleft[:, j] .+= αleft .* (Xleft[:, j] .- solved_left[:, j])
        end
    end
    stats = (
        right_rank=size(Rright_basis, 2),
        left_rank=size(Rleft_basis, 2),
        right_singulars=right_singulars,
        left_singulars=left_singulars,
    )
    Qright, Qleft, stats
end

function dual_scalar_rii_step_generic(
    Tsolve,
    Tadjoint_solve,
    Tmatrix,
    values,
    Xright,
    Xleft,
    z_nodes,
    z_weights;
    residual_ranktol=0.0,
)
    n, k = size(Xright)
    Rright = zeros(ComplexF64, n, k)
    Rleft = zeros(ComplexF64, n, k)
    for j in eachindex(values)
        Tλ = Tmatrix(values[j])
        Rright[:, j] .= Tλ * Xright[:, j]
        Rleft[:, j] .= Tλ' * Xleft[:, j]
    end

    if residual_ranktol > 0
        Rright_basis, Rright_coeffs, right_singulars = low_rank_column_factor(Rright; ranktol=residual_ranktol)
        Rleft_basis, Rleft_coeffs, left_singulars = low_rank_column_factor(Rleft; ranktol=residual_ranktol)
    else
        Rright_basis, Rright_coeffs, right_singulars = Rright, Matrix{ComplexF64}(I, k, k), svdvals(Rright)
        Rleft_basis, Rleft_coeffs, left_singulars = Rleft, Matrix{ComplexF64}(I, k, k), svdvals(Rleft)
    end

    Qright = zeros(ComplexF64, n, k)
    Qleft = zeros(ComplexF64, n, k)
    for (z, weight) in zip(z_nodes, z_weights)
        solved_right = size(Rright_basis, 2) == 0 ? zeros(ComplexF64, n, k) : Tsolve(z, Rright_basis) * Rright_coeffs
        solved_left = size(Rleft_basis, 2) == 0 ? zeros(ComplexF64, n, k) : Tadjoint_solve(z, Rleft_basis) * Rleft_coeffs
        for j in eachindex(values)
            αright = weight / (z - values[j])
            αleft = conj(weight) / conj(z - values[j])
            Qright[:, j] .+= αright .* (Xright[:, j] .- solved_right[:, j])
            Qleft[:, j] .+= αleft .* (Xleft[:, j] .- solved_left[:, j])
        end
    end
    stats = (
        right_residual_rank=size(Rright_basis, 2),
        left_residual_rank=size(Rleft_basis, 2),
        right_candidate_cols=size(Qright, 2),
        left_candidate_cols=size(Qleft, 2),
        right_singulars=right_singulars,
        left_singulars=left_singulars,
    )
    Qright, Qleft, stats
end

function dual_scalar_rii_summary(extraction, expected; residual_tol, match_atol)
    inside = extraction.inside
    good = inside .& (extraction.residuals .<= residual_tol)
    matched = match_expected_count(extraction.values[good], expected; atol=match_atol)
    spurious_good = max(count(good) - matched, 0)
    (
        inside=count(inside),
        good=count(good),
        matched=matched,
        spurious_good=spurious_good,
        max_residual=any(inside) ? maximum(extraction.residuals[inside]) : Inf,
        max_right_residual=any(inside) ? maximum(extraction.right_residuals[inside]) : Inf,
        max_left_residual=any(inside) ? maximum(extraction.left_residuals[inside]) : Inf,
    )
end

function print_dual_scalar_rii_status(label, extraction, expected, center, radius; residual_tol, match_atol)
    summary = dual_scalar_rii_summary(extraction, expected; residual_tol=residual_tol, match_atol=match_atol)
    @printf(
        "  %s inside=%d good=%d matched=%d spurious_good=%d max_res=%.3e max_right=%.3e max_left=%.3e\n",
        label,
        summary.inside,
        summary.good,
        summary.matched,
        summary.spurious_good,
        summary.max_residual,
        summary.max_right_residual,
        summary.max_left_residual,
    )
end

function run_dual_scalar_rii_polynomial_experiment(;
    name="dual_sensitive_polynomial",
    make_problem=dual_sensitive_polynomial_problem,
    basis_moments=5,
    basis_nodes=24,
    rii_nodes=48,
    iterations=3,
    basis_ranktol=1e-10,
    compression_ranktol=1e-10,
    residual_tol=1e-6,
    match_atol=1e-3,
    extraction_mode=:dual,
    residual_ranktol=1e-12,
)
    problem = make_problem()
    coeffs, center, radius, n = problem[1], problem[2], problem[3], problem[4]
    expected = length(problem) >= 5 ? ComplexF64.(problem[5]) : companion_reference(coeffs, center, radius)
    z_nodes, z_weights = circular_rule(center, radius, basis_nodes)
    rii_z_nodes, rii_z_weights = circular_rule(center, radius, rii_nodes)

    Random.seed!(9601)
    Xprobe = rand(ComplexF64, n, n)
    Wprobe = rand(ComplexF64, n, n)
    Tsolve = (z, B) -> polynomial_matrix(coeffs, z) \ B
    right_moments = initial_moments_generic_scaled(Tsolve, Xprobe, z_nodes, z_weights, center, radius, basis_moments)
    left_moments = initial_adjoint_moments_polynomial_scaled(coeffs, Wprobe, z_nodes, z_weights, center, radius, basis_moments)
    Xbasis, _ = moment_block_basis(right_moments, basis_moments; ranktol=basis_ranktol)
    Ybasis, _ = moment_block_basis(left_moments, basis_moments; ranktol=basis_ranktol)
    if size(Xbasis, 2) != size(Ybasis, 2)
        common = min(size(Xbasis, 2), size(Ybasis, 2))
        Xbasis = Xbasis[:, 1:common]
        Ybasis = Ybasis[:, 1:common]
    end
    if extraction_mode === :biorth
        Xbasis, Ybasis, _ = biorthogonalize_bases(Xbasis, Ybasis)
    elseif extraction_mode !== :dual
        error("extraction_mode must be :dual or :biorth")
    end

    println()
    println("Dual scalar-RII polynomial iteration: $name")
    println("  expanded Ritz-vector correction followed by low-rank residual solves and physical SVD compression")
    @printf(
        "  basis_nodes=%d rii_nodes=%d basis=(%d,%d) match_tol=%.1e residual_tol=%.1e\n",
        basis_nodes,
        rii_nodes,
        size(Xbasis, 2),
        size(Ybasis, 2),
        match_atol,
        residual_tol,
    )

    extraction = reduced_polynomial_extraction(coeffs, Xbasis, Ybasis, center, radius)
    print_dual_scalar_rii_status("iter=0", extraction, expected, center, radius; residual_tol=residual_tol, match_atol=match_atol)
    for iteration in 1:iterations
        selected = extraction.inside
        if !any(selected)
            println("  iteration stopped: no Ritz values inside contour")
            break
        end
        Qright, Qleft, solve_stats = dual_scalar_rii_step_polynomial(
            coeffs,
            extraction.values[selected],
            extraction.right_vectors[:, selected],
            extraction.left_vectors[:, selected],
            rii_z_nodes,
            rii_z_weights,
            residual_ranktol=residual_ranktol,
        )
        Xbasis, right_singulars = physical_basis_from_columns(Qright; ranktol=compression_ranktol)
        Ybasis, left_singulars = physical_basis_from_columns(Qleft; ranktol=compression_ranktol)
        if size(Xbasis, 2) != size(Ybasis, 2)
            common = min(size(Xbasis, 2), size(Ybasis, 2))
            Xbasis = Xbasis[:, 1:common]
            Ybasis = Ybasis[:, 1:common]
        end
        extraction_mode === :biorth && ((Xbasis, Ybasis, _) = biorthogonalize_bases(Xbasis, Ybasis))
        extraction = reduced_polynomial_extraction(coeffs, Xbasis, Ybasis, center, radius)
        @printf(
            "    residual ranks right=%d left=%d; compressed ranks right=%d left=%d sigma_right=%.3e sigma_left=%.3e\n",
            solve_stats.right_rank,
            solve_stats.left_rank,
            size(Xbasis, 2),
            size(Ybasis, 2),
            isempty(right_singulars) ? NaN : right_singulars[end] / right_singulars[1],
            isempty(left_singulars) ? NaN : left_singulars[end] / left_singulars[1],
        )
        print_dual_scalar_rii_status(
            "iter=$iteration",
            extraction,
            expected,
            center,
            radius;
            residual_tol=residual_tol,
            match_atol=match_atol,
        )
    end
end

function residual_blocks_from_extraction(coeffs, extraction; residual_ranktol=1e-10)
    values = extraction.values[extraction.inside]
    Xright = extraction.right_vectors[:, extraction.inside]
    Xleft = extraction.left_vectors[:, extraction.inside]
    n, k = size(Xright)
    Rright = zeros(ComplexF64, n, k)
    Rleft = zeros(ComplexF64, n, k)
    for j in 1:k
        Tλ = polynomial_matrix(coeffs, values[j])
        Rright[:, j] .= Tλ * Xright[:, j]
        Rleft[:, j] .= Tλ' * Xleft[:, j]
    end
    Rright_basis, _, right_singulars = low_rank_column_factor(Rright; ranktol=residual_ranktol)
    Rleft_basis, _, left_singulars = low_rank_column_factor(Rleft; ranktol=residual_ranktol)
    Rright_basis, Rleft_basis, right_singulars, left_singulars
end

function moment_compressed_dual_rii_bases(
    coeffs,
    Xbasis,
    Ybasis,
    extraction,
    z_nodes,
    z_weights,
    center,
    radius;
    moment_count=2,
    residual_ranktol=1e-10,
    compression_ranktol=1e-10,
)
    Rright_basis, Rleft_basis, right_residual_singulars, left_residual_singulars =
        residual_blocks_from_extraction(coeffs, extraction; residual_ranktol=residual_ranktol)
    n = size(Xbasis, 1)
    right_moments = [zeros(ComplexF64, n, size(Rright_basis, 2)) for _ in 1:moment_count]
    left_moments = [zeros(ComplexF64, n, size(Rleft_basis, 2)) for _ in 1:moment_count]

    for (z, weight) in zip(z_nodes, z_weights)
        ζ = (z - center) / radius
        base = weight / (z - center)
        Tz = polynomial_matrix(coeffs, z)
        solved_right = isempty(Rright_basis) ? zeros(ComplexF64, n, 0) : Tz \ Rright_basis
        solved_left = isempty(Rleft_basis) ? zeros(ComplexF64, n, 0) : Tz' \ Rleft_basis
        right_power = one(ComplexF64)
        left_power = one(ComplexF64)
        for k in 1:moment_count
            right_moments[k] .+= (base * right_power) .* solved_right
            left_moments[k] .+= (conj(base) * left_power) .* solved_left
            right_power /= ζ
            left_power *= ζ
        end
    end

    right_blocks = Matrix{ComplexF64}[Matrix(Xbasis)]
    left_blocks = Matrix{ComplexF64}[Matrix(Ybasis)]
    append!(right_blocks, right_moments)
    append!(left_blocks, left_moments)
    Xcandidate = reduce(hcat, right_blocks)
    Ycandidate = reduce(hcat, left_blocks)
    Xnew, right_singulars = physical_basis_from_columns(Xcandidate; ranktol=compression_ranktol)
    Ynew, left_singulars = physical_basis_from_columns(Ycandidate; ranktol=compression_ranktol)
    if size(Xnew, 2) != size(Ynew, 2)
        common = min(size(Xnew, 2), size(Ynew, 2))
        Xnew = Xnew[:, 1:common]
        Ynew = Ynew[:, 1:common]
    end
    stats = (
        right_residual_rank=size(Rright_basis, 2),
        left_residual_rank=size(Rleft_basis, 2),
        right_residual_singulars=right_residual_singulars,
        left_residual_singulars=left_residual_singulars,
        right_candidate_cols=size(Xcandidate, 2),
        left_candidate_cols=size(Ycandidate, 2),
        right_singulars=right_singulars,
        left_singulars=left_singulars,
    )
    Xnew, Ynew, stats
end

function run_dual_moment_compressed_rii_polynomial_experiment(;
    name="dual_sensitive_polynomial_bad_initial",
    make_problem=dual_sensitive_polynomial_problem,
    basis_moments=5,
    basis_nodes=6,
    rii_nodes=48,
    update_moments=(1, 2, 3, 4),
    basis_ranktol=1e-6,
    compression_ranktol=1e-10,
    residual_ranktol=1e-10,
    residual_tol=1e-6,
    match_atol=1e-3,
)
    problem = make_problem()
    coeffs, center, radius, n = problem[1], problem[2], problem[3], problem[4]
    expected = length(problem) >= 5 ? ComplexF64.(problem[5]) : companion_reference(coeffs, center, radius)
    z_nodes, z_weights = circular_rule(center, radius, basis_nodes)
    rii_z_nodes, rii_z_weights = circular_rule(center, radius, rii_nodes)

    Random.seed!(9601)
    Xprobe = rand(ComplexF64, n, n)
    Wprobe = rand(ComplexF64, n, n)
    Tsolve = (z, B) -> polynomial_matrix(coeffs, z) \ B
    right_moments = initial_moments_generic_scaled(Tsolve, Xprobe, z_nodes, z_weights, center, radius, basis_moments)
    left_moments = initial_adjoint_moments_polynomial_scaled(coeffs, Wprobe, z_nodes, z_weights, center, radius, basis_moments)
    Xbasis, _ = moment_block_basis(right_moments, basis_moments; ranktol=basis_ranktol)
    Ybasis, _ = moment_block_basis(left_moments, basis_moments; ranktol=basis_ranktol)
    if size(Xbasis, 2) != size(Ybasis, 2)
        common = min(size(Xbasis, 2), size(Ybasis, 2))
        Xbasis = Xbasis[:, 1:common]
        Ybasis = Ybasis[:, 1:common]
    end

    println()
    println("Dual moment-compressed RII polynomial update: $name")
    println("  uses residual Laurent moments instead of persistent Ritz-vector columns")
    @printf(
        "  basis_nodes=%d rii_nodes=%d initial_basis=(%d,%d) match_tol=%.1e residual_tol=%.1e\n",
        basis_nodes,
        rii_nodes,
        size(Xbasis, 2),
        size(Ybasis, 2),
        match_atol,
        residual_tol,
    )

    extraction0 = reduced_polynomial_extraction(coeffs, Xbasis, Ybasis, center, radius)
    print_dual_scalar_rii_status("iter=0", extraction0, expected, center, radius; residual_tol=residual_tol, match_atol=match_atol)
    for moment_count in update_moments
        Xnew, Ynew, stats = moment_compressed_dual_rii_bases(
            coeffs,
            Xbasis,
            Ybasis,
            extraction0,
            rii_z_nodes,
            rii_z_weights,
            center,
            radius;
            moment_count=moment_count,
            residual_ranktol=residual_ranktol,
            compression_ranktol=compression_ranktol,
        )
        extraction = reduced_polynomial_extraction(coeffs, Xnew, Ynew, center, radius)
        @printf(
            "    update_moments=%d residual_ranks=(%d,%d) candidate_cols=(%d,%d) basis=(%d,%d) sigma=(%.3e, %.3e)\n",
            moment_count,
            stats.right_residual_rank,
            stats.left_residual_rank,
            stats.right_candidate_cols,
            stats.left_candidate_cols,
            size(Xnew, 2),
            size(Ynew, 2),
            isempty(stats.right_singulars) ? NaN : stats.right_singulars[end] / stats.right_singulars[1],
            isempty(stats.left_singulars) ? NaN : stats.left_singulars[end] / stats.left_singulars[1],
        )
        print_dual_scalar_rii_status(
            "moment_update=$moment_count",
            extraction,
            expected,
            center,
            radius;
            residual_tol=residual_tol,
            match_atol=match_atol,
        )
    end
end

function run_dual_residual_laurent_two_sided_update_control(;
    name="dual_sensitive_polynomial_bad_initial",
    make_problem=dual_sensitive_polynomial_problem,
    basis_moments=5,
    basis_nodes=6,
    rii_nodes=48,
    basis_ranktol=1e-6,
    compression_ranktol=1e-10,
    residual_ranktol=1e-10,
    residual_tol=1e-6,
    match_atol=1e-3,
    print_rows=true,
)
    problem = make_problem()
    coeffs, center, radius, n = problem[1], problem[2], problem[3], problem[4]
    expected = length(problem) >= 5 ? ComplexF64.(problem[5]) : companion_reference(coeffs, center, radius)
    z_nodes, z_weights = circular_rule(center, radius, basis_nodes)
    rii_z_nodes, rii_z_weights = circular_rule(center, radius, rii_nodes)

    Random.seed!(9601)
    Xprobe = rand(ComplexF64, n, n)
    Wprobe = rand(ComplexF64, n, n)
    Tsolve = (z, B) -> polynomial_matrix(coeffs, z) \ B
    right_moments = initial_moments_generic_scaled(Tsolve, Xprobe, z_nodes, z_weights, center, radius, basis_moments)
    left_moments = initial_adjoint_moments_polynomial_scaled(coeffs, Wprobe, z_nodes, z_weights, center, radius, basis_moments)
    Xbasis, _ = moment_block_basis(right_moments, basis_moments; ranktol=basis_ranktol)
    Ybasis, _ = moment_block_basis(left_moments, basis_moments; ranktol=basis_ranktol)
    if size(Xbasis, 2) != size(Ybasis, 2)
        common = min(size(Xbasis, 2), size(Ybasis, 2))
        Xbasis = Xbasis[:, 1:common]
        Ybasis = Ybasis[:, 1:common]
    end
    extraction0 = reduced_polynomial_extraction(coeffs, Xbasis, Ybasis, center, radius)
    Xnew, Ynew, stats = moment_compressed_dual_rii_bases(
        coeffs,
        Xbasis,
        Ybasis,
        extraction0,
        rii_z_nodes,
        rii_z_weights,
        center,
        radius;
        moment_count=1,
        residual_ranktol=residual_ranktol,
        compression_ranktol=compression_ranktol,
    )

    function summarize(label, Xraw, Yraw; truncated=false)
        common = min(size(Xraw, 2), size(Yraw, 2))
        Xtest = Xraw[:, 1:common]
        Ytest = Yraw[:, 1:common]
        extraction = reduced_polynomial_extraction(coeffs, Xtest, Ytest, center, radius)
        summary = dual_scalar_rii_summary(extraction, expected; residual_tol=residual_tol, match_atol=match_atol)
        (
            stage=label,
            expected=length(expected),
            inside=summary.inside,
            good=summary.good,
            matched=summary.matched,
            spurious_good=summary.spurious_good,
            max_residual=summary.max_residual,
            raw_right_basis=size(Xraw, 2),
            raw_left_basis=size(Yraw, 2),
            right_basis=size(Xtest, 2),
            left_basis=size(Ytest, 2),
            truncated=truncated || size(Xraw, 2) != size(Yraw, 2),
        )
    end

    rows = [
        summarize(:initial, Xbasis, Ybasis),
        summarize(:two_sided_residual_laurent, Xnew, Ynew),
        summarize(:right_only_truncated, Xnew, Ybasis; truncated=true),
        summarize(:left_only_truncated, Xbasis, Ynew; truncated=true),
    ]

    if print_rows
        println()
        println("Dual residual-Laurent two-sided update control: $name")
        println("  one-sided repair cannot form a square Petrov-Galerkin reduced NEP without truncating away the new side")
        @printf(
            "  initial_basis=(%d,%d) updated_basis=(%d,%d) residual_ranks=(%d,%d)\n",
            size(Xbasis, 2),
            size(Ybasis, 2),
            size(Xnew, 2),
            size(Ynew, 2),
            stats.right_residual_rank,
            stats.left_residual_rank,
        )
        for row in rows
            @printf(
                "  stage=%s matched=%d/%d good=%d spurious=%d max=%.3e raw_basis=(%d,%d) used_basis=(%d,%d) truncated=%s\n",
                string(row.stage),
                row.matched,
                row.expected,
                row.good,
                row.spurious_good,
                row.max_residual,
                row.raw_right_basis,
                row.raw_left_basis,
                row.right_basis,
                row.left_basis,
                string(row.truncated),
            )
        end
    end
    (
        rows=rows,
        expected=length(expected),
        initial_basis=(right=size(Xbasis, 2), left=size(Ybasis, 2)),
        updated_basis=(right=size(Xnew, 2), left=size(Ynew, 2)),
        residual_ranks=(right=stats.right_residual_rank, left=stats.left_residual_rank),
    )
end

function residual_blocks_from_matrix_extraction(Tmatrix, extraction; residual_ranktol=1e-10)
    values = extraction.values[extraction.inside]
    Xright = extraction.right_vectors[:, extraction.inside]
    Xleft = extraction.left_vectors[:, extraction.inside]
    n, k = size(Xright)
    Rright = zeros(ComplexF64, n, k)
    Rleft = zeros(ComplexF64, n, k)
    for j in 1:k
        Tλ = Tmatrix(values[j])
        Rright[:, j] .= Tλ * Xright[:, j]
        Rleft[:, j] .= Tλ' * Xleft[:, j]
    end
    Rright_basis, _, right_singulars = low_rank_column_factor(Rright; ranktol=residual_ranktol)
    Rleft_basis, _, left_singulars = low_rank_column_factor(Rleft; ranktol=residual_ranktol)
    Rright_basis, Rleft_basis, right_singulars, left_singulars
end

function residual_laurent_moment_blocks_generic(
    Tsolve,
    Tadjoint_solve,
    Rright_basis,
    Rleft_basis,
    z_nodes,
    z_weights,
    center,
    radius;
    moment_count,
)
    n = size(Rright_basis, 1)
    right_moments = [zeros(ComplexF64, n, size(Rright_basis, 2)) for _ in 1:moment_count]
    left_moments = [zeros(ComplexF64, n, size(Rleft_basis, 2)) for _ in 1:moment_count]

    for (z, weight) in zip(z_nodes, z_weights)
        ζ = (z - center) / radius
        base = weight / (z - center)
        solved_right = size(Rright_basis, 2) == 0 ? zeros(ComplexF64, n, 0) : Tsolve(z, Rright_basis)
        solved_left = size(Rleft_basis, 2) == 0 ? zeros(ComplexF64, n, 0) : Tadjoint_solve(z, Rleft_basis)
        right_power = one(ComplexF64)
        left_power = one(ComplexF64)
        for k in 1:moment_count
            right_moments[k] .+= (base * right_power) .* solved_right
            left_moments[k] .+= (conj(base) * left_power) .* solved_left
            right_power /= ζ
            left_power *= ζ
        end
    end
    right_moments, left_moments
end

function residual_laurent_remote_worker_step(
    Tsolve,
    Tadjoint_solve,
    Rright_basis,
    Rleft_basis,
    z_nodes,
    z_weights,
    center,
    radius,
    moment_count,
)
    start_ns = time_ns()
    right_moments, left_moments = residual_laurent_moment_blocks_generic(
        Tsolve,
        Tadjoint_solve,
        Rright_basis,
        Rleft_basis,
        z_nodes,
        z_weights,
        center,
        radius;
        moment_count=moment_count,
    )
    (
        right_moments=right_moments,
        left_moments=left_moments,
        nodes=length(z_nodes),
        elapsed_ns=time_ns() - start_ns,
        pid=myid(),
    )
end

function init_residual_laurent_remote_worker!(
    key::Symbol,
    Tsolve,
    Tadjoint_solve,
    z_nodes,
    z_weights,
    center,
    radius,
    moment_count,
)
    _MOMENT_RII_REMOTE_WORKSPACES[key] = (
        Tsolve=Tsolve,
        Tadjoint_solve=Tadjoint_solve,
        z_nodes=ComplexF64.(z_nodes),
        z_weights=ComplexF64.(z_weights),
        center=ComplexF64(center),
        radius=Float64(radius),
        moment_count=Int(moment_count),
    )
    (
        pid=myid(),
        nodes=length(z_nodes),
        key=key,
    )
end

function residual_laurent_persistent_remote_worker_step(key::Symbol, Rright_basis, Rleft_basis)
    ws = _MOMENT_RII_REMOTE_WORKSPACES[key]
    start_ns = time_ns()
    right_moments, left_moments = residual_laurent_moment_blocks_generic(
        ws.Tsolve,
        ws.Tadjoint_solve,
        Rright_basis,
        Rleft_basis,
        ws.z_nodes,
        ws.z_weights,
        ws.center,
        ws.radius;
        moment_count=ws.moment_count,
    )
    (
        right_moments=right_moments,
        left_moments=left_moments,
        nodes=length(ws.z_nodes),
        elapsed_ns=time_ns() - start_ns,
        pid=myid(),
    )
end

function cleanup_residual_laurent_remote_worker!(key::Symbol)
    existed = haskey(_MOMENT_RII_REMOTE_WORKSPACES, key)
    delete!(_MOMENT_RII_REMOTE_WORKSPACES, key)
    existed
end

function init_sparse_factor_residual_laurent_remote_worker!(
    key::Symbol,
    Tmatrix,
    z_nodes,
    z_weights,
    center,
    radius,
    moment_count,
)
    _MOMENT_RII_REMOTE_WORKSPACES[key] = (
        Tmatrix=Tmatrix,
        z_nodes=ComplexF64.(z_nodes),
        z_weights=ComplexF64.(z_weights),
        center=ComplexF64(center),
        radius=Float64(radius),
        moment_count=Int(moment_count),
        right_factors=Dict{ComplexF64, Any}(),
        left_factors=Dict{ComplexF64, Any}(),
        right_solutions=Dict{ComplexF64, Matrix{ComplexF64}}(),
        left_solutions=Dict{ComplexF64, Matrix{ComplexF64}}(),
        right_solves=Ref(0),
        left_solves=Ref(0),
        right_solution_buffers=Ref(0),
        left_solution_buffers=Ref(0),
    )
    (
        pid=myid(),
        nodes=length(z_nodes),
        key=key,
    )
end

function sparse_factor_workspace_solve!(ws, z, B; adjoint=false)
    key = ComplexF64(z)
    factors = adjoint ? ws.left_factors : ws.right_factors
    solutions = adjoint ? ws.left_solutions : ws.right_solutions
    if !haskey(factors, key)
        factors[key] = lu(adjoint ? ws.Tmatrix(z)' : ws.Tmatrix(z))
    end
    if !haskey(solutions, key) || size(solutions[key]) != size(B)
        solutions[key] = similar(B, ComplexF64)
        if adjoint
            ws.left_solution_buffers[] += 1
        else
            ws.right_solution_buffers[] += 1
        end
    end
    if adjoint
        ws.left_solves[] += 1
    else
        ws.right_solves[] += 1
    end
    ldiv!(solutions[key], factors[key], B)
    solutions[key]
end

function residual_laurent_sparse_factor_remote_worker_step(key::Symbol, Rright_basis, Rleft_basis)
    ws = _MOMENT_RII_REMOTE_WORKSPACES[key]
    start_ns = time_ns()
    n = size(Rright_basis, 1)
    right_moments = [zeros(ComplexF64, n, size(Rright_basis, 2)) for _ in 1:ws.moment_count]
    left_moments = [zeros(ComplexF64, n, size(Rleft_basis, 2)) for _ in 1:ws.moment_count]

    for (z, weight) in zip(ws.z_nodes, ws.z_weights)
        ζ = (z - ws.center) / ws.radius
        base = weight / (z - ws.center)
        solved_right = size(Rright_basis, 2) == 0 ?
            zeros(ComplexF64, n, 0) :
            sparse_factor_workspace_solve!(ws, z, Rright_basis; adjoint=false)
        solved_left = size(Rleft_basis, 2) == 0 ?
            zeros(ComplexF64, n, 0) :
            sparse_factor_workspace_solve!(ws, z, Rleft_basis; adjoint=true)
        right_power = one(ComplexF64)
        left_power = one(ComplexF64)
        for k in 1:ws.moment_count
            right_moments[k] .+= (base * right_power) .* solved_right
            left_moments[k] .+= (conj(base) * left_power) .* solved_left
            right_power /= ζ
            left_power *= ζ
        end
    end

    (
        right_moments=right_moments,
        left_moments=left_moments,
        nodes=length(ws.z_nodes),
        elapsed_ns=time_ns() - start_ns,
        pid=myid(),
        right_factorizations=length(ws.right_factors),
        left_factorizations=length(ws.left_factors),
        right_solution_buffers=ws.right_solution_buffers[],
        left_solution_buffers=ws.left_solution_buffers[],
        right_solves=ws.right_solves[],
        left_solves=ws.left_solves[],
    )
end

function compress_residual_laurent_candidates(
    Xbasis,
    Ybasis,
    right_moments,
    left_moments;
    compression_ranktol,
)
    right_blocks = Matrix{ComplexF64}[Matrix(Xbasis)]
    left_blocks = Matrix{ComplexF64}[Matrix(Ybasis)]
    append!(right_blocks, right_moments)
    append!(left_blocks, left_moments)
    Xcandidate = reduce(hcat, right_blocks)
    Ycandidate = reduce(hcat, left_blocks)
    Xnew, right_singulars = physical_basis_from_columns(Xcandidate; ranktol=compression_ranktol)
    Ynew, left_singulars = physical_basis_from_columns(Ycandidate; ranktol=compression_ranktol)
    if size(Xnew, 2) != size(Ynew, 2)
        common = min(size(Xnew, 2), size(Ynew, 2))
        Xnew = Xnew[:, 1:common]
        Ynew = Ynew[:, 1:common]
    end
    Xnew, Ynew, Xcandidate, Ycandidate, right_singulars, left_singulars
end

function moment_compressed_dual_rii_bases_generic(
    Tsolve,
    Tadjoint_solve,
    Tmatrix,
    Xbasis,
    Ybasis,
    extraction,
    z_nodes,
    z_weights,
    center,
    radius;
    moment_count=2,
    residual_ranktol=1e-10,
    compression_ranktol=1e-10,
)
    Rright_basis, Rleft_basis, right_residual_singulars, left_residual_singulars =
        residual_blocks_from_matrix_extraction(Tmatrix, extraction; residual_ranktol=residual_ranktol)
    right_moments, left_moments = residual_laurent_moment_blocks_generic(
        Tsolve,
        Tadjoint_solve,
        Rright_basis,
        Rleft_basis,
        z_nodes,
        z_weights,
        center,
        radius;
        moment_count=moment_count,
    )
    Xnew, Ynew, Xcandidate, Ycandidate, right_singulars, left_singulars =
        compress_residual_laurent_candidates(
            Xbasis,
            Ybasis,
            right_moments,
            left_moments;
            compression_ranktol=compression_ranktol,
        )
    stats = (
        right_residual_rank=size(Rright_basis, 2),
        left_residual_rank=size(Rleft_basis, 2),
        right_residual_singulars=right_residual_singulars,
        left_residual_singulars=left_residual_singulars,
        right_candidate_cols=size(Xcandidate, 2),
        left_candidate_cols=size(Ycandidate, 2),
        right_singulars=right_singulars,
        left_singulars=left_singulars,
    )
    Xnew, Ynew, stats
end

function run_dual_moment_compressed_rii_analytic_experiment(;
    cases=(scalar_sine_case(), scalar_cosine_case(), scalar_shifted_sine_case(), scalar_expm1_case()),
    center=0.0 + 0.0im,
    radius=20.0,
    operator_builder=similarity_analytic_tools,
    operator_label="similarity",
    basis_moments=4,
    basis_nodes=8,
    rii_nodes=256,
    update_moments=(1, 2, 4),
    basis_ranktol=0.5,
    compression_ranktol=1e-10,
    residual_ranktol=1e-10,
    determinant_nodes=2048,
    determinant_capacity=96,
    extractor=:determinant,
    reduced_moments=8,
    reduced_nodes=256,
    reduced_ranktol=1e-10,
    reduced_ss_mode=:similarity,
    loewner_points=4,
    loewner_radius=1.6,
    loewner_phase=0.0,
    residual_normalization=:operator,
    reduced_refinement=:none,
    refinement_steps=4,
    refinement_nodes=256,
    component_scaling=:none,
    component_scaling_nodes=64,
    residual_tol=1e-8,
    match_atol=1e-6,
)
    labels = join((case.name for case in cases), ",")
    component_scales = analytic_component_scales(cases, center, radius; mode=component_scaling, nodes=component_scaling_nodes)
    Tmatrix, Tderivative, Tsolve, Tadjoint_solve, expected_roots =
        operator_builder(cases; component_scales=component_scales)
    expected = expected_roots(center, radius)
    n = length(cases)
    z_nodes, z_weights = circular_rule(center, radius, basis_nodes)
    rii_z_nodes, rii_z_weights = circular_rule(center, radius, rii_nodes)

    Random.seed!(9701 + round(Int, radius * 10))
    Xprobe = rand(ComplexF64, n, n)
    Wprobe = rand(ComplexF64, n, n)
    right_moments = initial_moments_generic_scaled(Tsolve, Xprobe, z_nodes, z_weights, center, radius, basis_moments)
    left_moments = initial_adjoint_moments_generic_scaled(Tadjoint_solve, Wprobe, z_nodes, z_weights, center, radius, basis_moments)
    Xbasis, right_basis_singulars = moment_block_basis(right_moments, basis_moments; ranktol=basis_ranktol)
    Ybasis, left_basis_singulars = moment_block_basis(left_moments, basis_moments; ranktol=basis_ranktol)
    if size(Xbasis, 2) != size(Ybasis, 2)
        common = min(size(Xbasis, 2), size(Ybasis, 2))
        Xbasis = Xbasis[:, 1:common]
        Ybasis = Ybasis[:, 1:common]
    end

    println()
    println("Dual moment-compressed RII analytic update: [$labels]")
    println("  operator=$operator_label")
    println("  extractor=$extractor for the small reduced NEP; no polynomial companion linearization")
    @printf(
        "  radius=%.3g expected=%d basis_nodes=%d rii_nodes=%d initial_basis=(%d,%d) basis_sigma=(%.3e, %.3e)\n",
        radius,
        length(expected),
        basis_nodes,
        rii_nodes,
        size(Xbasis, 2),
        size(Ybasis, 2),
        isempty(right_basis_singulars) ? NaN : right_basis_singulars[end] / right_basis_singulars[1],
        isempty(left_basis_singulars) ? NaN : left_basis_singulars[end] / left_basis_singulars[1],
    )
    if size(Xbasis, 2) == 0
        println("  skipped: rank truncation removed the whole initial basis")
        return
    end

    extraction0 = reduced_analytic_extraction(
        Tmatrix,
        Tderivative,
        Xbasis,
        Ybasis,
        center,
        radius;
        extractor=extractor,
        determinant_nodes=determinant_nodes,
        determinant_capacity=determinant_capacity,
        reduced_moments=reduced_moments,
        reduced_nodes=reduced_nodes,
        reduced_ranktol=reduced_ranktol,
        reduced_ss_mode=reduced_ss_mode,
        loewner_points=loewner_points,
        loewner_radius=loewner_radius,
        loewner_phase=loewner_phase,
        residual_normalization=residual_normalization,
        refinement=reduced_refinement,
        refinement_steps=refinement_steps,
        refinement_nodes=refinement_nodes,
    )
    @printf("    reduced_count=%d count_error=%.3e\n", extraction0.count_estimate, extraction0.count_error)
    print_dual_scalar_rii_status("iter=0", extraction0, expected, center, radius; residual_tol=residual_tol, match_atol=match_atol)
    for moment_count in update_moments
        Xnew, Ynew, stats = moment_compressed_dual_rii_bases_generic(
            Tsolve,
            Tadjoint_solve,
            Tmatrix,
            Xbasis,
            Ybasis,
            extraction0,
            rii_z_nodes,
            rii_z_weights,
            center,
            radius;
            moment_count=moment_count,
            residual_ranktol=residual_ranktol,
            compression_ranktol=compression_ranktol,
        )
        extraction = reduced_analytic_extraction(
            Tmatrix,
            Tderivative,
            Xnew,
            Ynew,
            center,
            radius;
            extractor=extractor,
            determinant_nodes=determinant_nodes,
            determinant_capacity=determinant_capacity,
            reduced_moments=reduced_moments,
            reduced_nodes=reduced_nodes,
            reduced_ranktol=reduced_ranktol,
            reduced_ss_mode=reduced_ss_mode,
            loewner_points=loewner_points,
            loewner_radius=loewner_radius,
            loewner_phase=loewner_phase,
            residual_normalization=residual_normalization,
            refinement=reduced_refinement,
            refinement_steps=refinement_steps,
            refinement_nodes=refinement_nodes,
        )
        @printf(
            "    update_moments=%d reduced_count=%d residual_ranks=(%d,%d) candidate_cols=(%d,%d) basis=(%d,%d) sigma=(%.3e, %.3e)\n",
            moment_count,
            extraction.count_estimate,
            stats.right_residual_rank,
            stats.left_residual_rank,
            stats.right_candidate_cols,
            stats.left_candidate_cols,
            size(Xnew, 2),
            size(Ynew, 2),
            isempty(stats.right_singulars) ? NaN : stats.right_singulars[end] / stats.right_singulars[1],
            isempty(stats.left_singulars) ? NaN : stats.left_singulars[end] / stats.left_singulars[1],
        )
        print_dual_scalar_rii_status(
            "moment_update=$moment_count",
            extraction,
            expected,
            center,
            radius;
            residual_tol=residual_tol,
            match_atol=match_atol,
        )
    end
end

function run_dual_moment_compressed_rii_analytic_iteration(;
    cases=(scalar_sine_case(), scalar_cosine_case()),
    center=0.0 + 0.0im,
    radius=20.0,
    operator_builder=similarity_analytic_tools,
    operator_label="similarity",
    basis_moments=4,
    basis_nodes=8,
    rii_nodes=256,
    update_moment_count=1,
    iterations=3,
    basis_ranktol=0.5,
    compression_ranktol=1e-10,
    residual_ranktol=1e-10,
    determinant_nodes=2048,
    determinant_capacity=96,
    extractor=:determinant,
    reduced_moments=8,
    reduced_nodes=256,
    reduced_ranktol=1e-10,
    reduced_ss_mode=:similarity,
    loewner_points=4,
    loewner_radius=1.6,
    loewner_phase=0.0,
    residual_normalization=:operator,
    reduced_refinement=:none,
    refinement_steps=4,
    refinement_nodes=256,
    component_scaling=:none,
    component_scaling_nodes=64,
    residual_tol=1e-8,
    match_atol=1e-6,
    biorthogonalize=false,
    update_mode=:moment_compressed,
    pipeline_config=nothing,
    basis_config=nothing,
    extraction_config=nothing,
    update_config=nothing,
    verbose=true,
)
    labels = join((case.name for case in cases), ",")
    chart = ContourChart(
        center,
        radius;
        component_scaling=component_scaling,
        component_scaling_nodes=component_scaling_nodes,
    )
    ctx = analytic_context(cases, chart, operator_builder)
    expected = ctx.expected
    if pipeline_config !== nothing
        if basis_config !== nothing || extraction_config !== nothing || update_config !== nothing
            error("pass either pipeline_config or individual basis/extraction/update configs")
        end
        basis_config = pipeline_config.basis
        extraction_config = pipeline_config.extractor
        update_config = pipeline_config.update
    end
    extractor_config = extraction_config === nothing ? ReducedExtractorConfig(
            extractor=extractor,
            determinant_nodes=determinant_nodes,
            determinant_capacity=determinant_capacity,
            reduced_moments=reduced_moments,
            reduced_nodes=reduced_nodes,
            reduced_ranktol=reduced_ranktol,
            reduced_ss_mode=reduced_ss_mode,
            loewner_points=loewner_points,
            loewner_radius=loewner_radius,
            loewner_phase=loewner_phase,
            residual_normalization=residual_normalization,
            refinement=reduced_refinement,
            refinement_steps=refinement_steps,
            refinement_nodes=refinement_nodes,
        ) : extraction_config
    extractor = extractor_config.extractor
    determinant_nodes = extractor_config.determinant_nodes
    determinant_capacity = extractor_config.determinant_capacity
    reduced_moments = extractor_config.reduced_moments
    reduced_nodes = extractor_config.reduced_nodes
    reduced_ranktol = extractor_config.reduced_ranktol
    reduced_ss_mode = extractor_config.reduced_ss_mode
    loewner_points = extractor_config.loewner_points
    loewner_radius = extractor_config.loewner_radius
    loewner_phase = extractor_config.loewner_phase
    residual_normalization = extractor_config.residual_normalization
    reduced_refinement = extractor_config.refinement
    refinement_steps = extractor_config.refinement_steps
    refinement_nodes = extractor_config.refinement_nodes
    residual_config = update_config === nothing ? ResidualUpdateConfig(
            moment_count=update_moment_count,
            rii_nodes=rii_nodes,
            residual_ranktol=residual_ranktol,
            compression_ranktol=compression_ranktol,
            mode=update_mode,
            biorthogonalize=biorthogonalize,
        ) : update_config
    update_moment_count = residual_config.moment_count
    rii_nodes = residual_config.rii_nodes
    residual_ranktol = residual_config.residual_ranktol
    compression_ranktol = residual_config.compression_ranktol
    update_mode = residual_config.mode
    biorthogonalize = residual_config.biorthogonalize
    default_seed = 9801 + round(Int, radius * 10) + 17 * length(cases)
    basis_config = basis_config === nothing ? MomentBasisConfig(
            moments=basis_moments,
            nodes=basis_nodes,
            ranktol=basis_ranktol,
            seed=default_seed,
            biorthogonalize=biorthogonalize,
        ) : MomentBasisConfig(
            moments=basis_config.moments,
            nodes=basis_config.nodes,
            ranktol=basis_config.ranktol,
            seed=basis_config.seed === nothing ? default_seed : basis_config.seed,
            biorthogonalize=basis_config.biorthogonalize,
        )
    basis_moments = basis_config.moments
    basis_nodes = basis_config.nodes
    basis_ranktol = basis_config.ranktol
    biorthogonalize = basis_config.biorthogonalize
    trial = initial_dual_trial_spaces(ctx, chart, basis_config)
    initial_summary = trial_space_summary(trial)

    if verbose
        println()
        println("Iterated dual moment-compressed RII analytic update: [$labels]")
        println("  operator=$operator_label; repeats the same residual Laurent chart update; extractor=$extractor supplies scalar Ritz data")
        @printf(
            "  center=%.6g%+.6gi radius=%.3g expected=%d basis_nodes=%d rii_nodes=%d update_moments=%d update_mode=%s component_scaling=%s initial_basis=(%d,%d) basis_sigma=(%.3e, %.3e)\n",
            real(center),
            imag(center),
            radius,
            length(expected),
            basis_nodes,
            rii_nodes,
            update_moment_count,
            string(update_mode),
            string(component_scaling),
            initial_summary.right_cols,
            initial_summary.left_cols,
            initial_summary.right_sigma,
            initial_summary.left_sigma,
        )
    end
    if size(trial.X, 2) == 0
        verbose && println("  skipped: rank truncation removed the whole initial basis")
        return (
            extraction=nothing,
            Xbasis=trial.X,
            Ybasis=trial.Y,
            expected=expected,
            summaries=NamedTuple[],
            center=center,
            radius=radius,
        )
    end

    summaries = NamedTuple[]
    extraction = extract_reduced_nep(ctx, trial, chart, extractor_config)
    extractions = Any[extraction]
    push!(summaries, merge((iteration=0,), dual_scalar_rii_summary(extraction, expected; residual_tol=residual_tol, match_atol=match_atol)))
    if verbose
        @printf("    iter=0 reduced_count=%d count_error=%.3e\n", extraction.count_estimate, extraction.count_error)
        print_dual_scalar_rii_status("iter=0", extraction, expected, center, radius; residual_tol=residual_tol, match_atol=match_atol)
    end
    for iteration in 1:iterations
        if update_mode === :moment_compressed
            trial, stats = residual_laurent_update(ctx, trial, extraction, chart, residual_config)
        elseif update_mode === :scalar_expanded
            selected = extraction.inside
            if !any(selected)
                verbose && println("  iteration stopped: no Ritz values inside contour")
                break
            end
            rii_z_nodes, rii_z_weights = circular_rule(chart, rii_nodes)
            Qright, Qleft, solve_stats = dual_scalar_rii_step_generic(
                ctx.Tsolve,
                ctx.Tadjoint_solve,
                ctx.Tmatrix,
                extraction.values[selected],
                extraction.right_vectors[:, selected],
                extraction.left_vectors[:, selected],
                rii_z_nodes,
                rii_z_weights;
                residual_ranktol=residual_ranktol,
            )
            Xcandidate, right_singulars = physical_basis_from_columns(Qright; ranktol=compression_ranktol)
            Ycandidate, left_singulars = physical_basis_from_columns(Qleft; ranktol=compression_ranktol)
            trial = common_square_trial_spaces(TrialSpaces(
                X=Xcandidate,
                Y=Ycandidate,
                right_singulars=Float64.(right_singulars),
                left_singulars=Float64.(left_singulars),
                source=:scalar_expanded_rii_update,
            ))
            if biorthogonalize
                Xbi, Ybi, cross_singulars = biorthogonalize_bases(trial.X, trial.Y)
                trial = TrialSpaces(
                    X=Xbi,
                    Y=Ybi,
                    right_singulars=Float64.(cross_singulars),
                    left_singulars=Float64.(cross_singulars),
                    source=:biorthogonalized_scalar_expanded_rii_update,
                )
            end
            stats = merge(solve_stats, (right_singulars=right_singulars, left_singulars=left_singulars))
        else
            error("unknown analytic RII update_mode: $update_mode")
        end
        extraction = extract_reduced_nep(ctx, trial, chart, extractor_config)
        push!(extractions, extraction)
        push!(
            summaries,
            merge(
                (
                    iteration=iteration,
                    reduced_count=extraction.count_estimate,
                    right_residual_rank=stats.right_residual_rank,
                    left_residual_rank=stats.left_residual_rank,
                    right_candidate_cols=stats.right_candidate_cols,
                    left_candidate_cols=stats.left_candidate_cols,
                    right_basis_cols=size(trial.X, 2),
                    left_basis_cols=size(trial.Y, 2),
                ),
                dual_scalar_rii_summary(extraction, expected; residual_tol=residual_tol, match_atol=match_atol),
            ),
        )
        if verbose
            @printf(
                "    iter=%d reduced_count=%d residual_ranks=(%d,%d) candidate_cols=(%d,%d) basis=(%d,%d) sigma=(%.3e, %.3e)\n",
                iteration,
                extraction.count_estimate,
                stats.right_residual_rank,
                stats.left_residual_rank,
                stats.right_candidate_cols,
                stats.left_candidate_cols,
                size(trial.X, 2),
                size(trial.Y, 2),
                isempty(stats.right_singulars) ? NaN : stats.right_singulars[end] / stats.right_singulars[1],
                isempty(stats.left_singulars) ? NaN : stats.left_singulars[end] / stats.left_singulars[1],
            )
            print_dual_scalar_rii_status(
                "iter=$iteration",
                extraction,
                expected,
                center,
                radius;
                residual_tol=residual_tol,
                match_atol=match_atol,
            )
        end
    end
    (
        extraction=extraction,
        extractions=extractions,
        Xbasis=trial.X,
        Ybasis=trial.Y,
        expected=expected,
        summaries=summaries,
        center=center,
        radius=radius,
    )
end

function run_reduced_loewner_extractor_comparison(;
    cases=(scalar_sine_case(), scalar_cosine_case(), scalar_shifted_sine_case()),
    center=0.0 + 0.0im,
    radius=20.0,
    operator_builder=similarity_analytic_tools,
    operator_label="similarity",
    basis_moments=4,
    basis_nodes=8,
    rii_nodes=128,
    basis_ranktol=0.5,
    compression_ranktol=1e-10,
    residual_ranktol=1e-10,
    update_moment_count=1,
    determinant_nodes=512,
    determinant_capacity=80,
    reduced_moments=16,
    reduced_nodes=512,
    reduced_ranktol=1e-10,
    loewner_radii=(1.3, 1.6, 2.0),
    residual_normalization=:operator,
    component_scaling=:none,
    component_scaling_nodes=64,
    residual_tol=1e-8,
    match_atol=1e-6,
)
    labels = join((case.name for case in cases), ",")
    chart = ContourChart(
        center,
        radius;
        component_scaling=component_scaling,
        component_scaling_nodes=component_scaling_nodes,
    )
    ctx = analytic_context(cases, chart, operator_builder)
    expected = ctx.expected
    trial = initial_dual_trial_spaces(
        ctx,
        chart;
        basis_moments=basis_moments,
        basis_nodes=basis_nodes,
        basis_ranktol=basis_ranktol,
        seed=9901 + round(Int, radius * 10) + 19 * length(cases),
    )
    ss_config = ReducedExtractorConfig(
        extractor=:ss_counted,
        determinant_nodes=determinant_nodes,
        determinant_capacity=determinant_capacity,
        reduced_moments=reduced_moments,
        reduced_nodes=reduced_nodes,
        reduced_ranktol=reduced_ranktol,
        residual_normalization=residual_normalization,
    )
    update_config = ResidualUpdateConfig(
        moment_count=update_moment_count,
        rii_nodes=rii_nodes,
        residual_ranktol=residual_ranktol,
        compression_ranktol=compression_ranktol,
    )
    initial_summary = trial_space_summary(trial)

    println()
    println("Reduced Loewner extractor comparison: [$labels]")
    println("  operator=$operator_label; compares Hankel/SS and Loewner assembly using the same reduced contour solve data")
    @printf(
        "  radius=%.3g expected=%d basis_nodes=%d rii_nodes=%d initial_basis=(%d,%d) basis_sigma=(%.3e, %.3e)\n",
        radius,
        length(expected),
        basis_nodes,
        rii_nodes,
        initial_summary.right_cols,
        initial_summary.left_cols,
        initial_summary.right_sigma,
        initial_summary.left_sigma,
    )
    if size(trial.X, 2) == 0
        println("  skipped: rank truncation removed the whole initial basis")
        return
    end

    function score(label, extraction)
        summary = dual_scalar_rii_summary(extraction, expected; residual_tol=residual_tol, match_atol=match_atol)
        sigma_ratio = if hasproperty(extraction, :singular_values) && !isempty(extraction.singular_values)
            extraction.singular_values[min(length(extraction.singular_values), max(extraction.count_estimate, 1))] / extraction.singular_values[1]
        else
            NaN
        end
        @printf(
            "  %-28s count=%d inside=%d good=%d matched=%d spurious=%d max=%.3e sigma_at_count=%.3e\n",
            label,
            extraction.count_estimate,
            summary.inside,
            summary.good,
            summary.matched,
            summary.spurious_good,
            summary.max_residual,
            sigma_ratio,
        )
    end

    extraction0 = extract_reduced_nep(ctx, trial, chart, ss_config)
    score("initial ss_counted", extraction0)
    for loewner_radius in loewner_radii
        loewner_config = ReducedExtractorConfig(
            extractor=:loewner_counted,
            determinant_nodes=determinant_nodes,
            determinant_capacity=determinant_capacity,
            reduced_nodes=reduced_nodes,
            reduced_ranktol=reduced_ranktol,
            loewner_radius=loewner_radius,
            residual_normalization=residual_normalization,
        )
        extraction = extract_reduced_nep(ctx, trial, chart, loewner_config)
        score("initial loewner rho=$loewner_radius", extraction)
    end

    updated_trial, stats = residual_laurent_update(ctx, trial, extraction0, chart, update_config)
    @printf(
        "  residual update: ranks=(%d,%d) candidate_cols=(%d,%d) basis=(%d,%d)\n",
        stats.right_residual_rank,
        stats.left_residual_rank,
        stats.right_candidate_cols,
        stats.left_candidate_cols,
        size(updated_trial.X, 2),
        size(updated_trial.Y, 2),
    )
    extraction = extract_reduced_nep(ctx, updated_trial, chart, ss_config)
    score("updated ss_counted", extraction)
    for loewner_radius in loewner_radii
        loewner_config = ReducedExtractorConfig(
            extractor=:loewner_counted,
            determinant_nodes=determinant_nodes,
            determinant_capacity=determinant_capacity,
            reduced_nodes=reduced_nodes,
            reduced_ranktol=reduced_ranktol,
            loewner_radius=loewner_radius,
            residual_normalization=residual_normalization,
        )
        extraction = extract_reduced_nep(ctx, updated_trial, chart, loewner_config)
        score("updated loewner rho=$loewner_radius", extraction)
    end
end

function run_reduced_analytic_refinement_comparison(;
    cases=(scalar_sine_case(), scalar_cosine_case(), scalar_shifted_sine_case()),
    center=0.0 + 0.0im,
    radius=20.0,
    operator_builder=similarity_analytic_tools,
    operator_label="similarity",
    extractor=:ss_counted,
    basis_moments=4,
    basis_nodes=8,
    rii_nodes=128,
    basis_ranktol=0.5,
    compression_ranktol=1e-10,
    residual_ranktol=1e-10,
    update_moment_count=1,
    determinant_nodes=512,
    determinant_capacity=80,
    reduced_moments=16,
    reduced_nodes=512,
    reduced_ranktol=1e-10,
    residual_normalization=:operator,
    component_scaling=:none,
    component_scaling_nodes=64,
    refinement_modes=(:none, :scalar_newton, :block_newton),
    refinement_steps=4,
    refinement_nodes=256,
    residual_tol=1e-8,
    match_atol=1e-6,
    print_rows=true,
)
    labels = join((case.name for case in cases), ",")
    chart = ContourChart(
        center,
        radius;
        component_scaling=component_scaling,
        component_scaling_nodes=component_scaling_nodes,
    )
    ctx = analytic_context(cases, chart, operator_builder)
    expected = ctx.expected
    trial = initial_dual_trial_spaces(
        ctx,
        chart;
        basis_moments=basis_moments,
        basis_nodes=basis_nodes,
        basis_ranktol=basis_ranktol,
        seed=9917 + round(Int, radius * 10) + 23 * length(cases),
    )
    update_config = ResidualUpdateConfig(
        moment_count=update_moment_count,
        rii_nodes=rii_nodes,
        residual_ranktol=residual_ranktol,
        compression_ranktol=compression_ranktol,
    )

    if print_rows
        println()
        println("Reduced analytic refinement comparison: [$labels]")
        println("  operator=$operator_label; compares reduced extraction with scalar and invariant-pair cleanup")
        @printf(
            "  extractor=%s radius=%.3g expected=%d basis=(%d,%d)\n",
            string(extractor),
            radius,
            length(expected),
            size(trial.X, 2),
            size(trial.Y, 2),
        )
    end
    size(trial.X, 2) == 0 && return

    function config(refinement)
        ReducedExtractorConfig(
            extractor=extractor,
            determinant_nodes=determinant_nodes,
            determinant_capacity=determinant_capacity,
            reduced_moments=reduced_moments,
            reduced_nodes=reduced_nodes,
            reduced_ranktol=reduced_ranktol,
            residual_normalization=residual_normalization,
            refinement=refinement,
            refinement_steps=refinement_steps,
            refinement_nodes=refinement_nodes,
        )
    end

    function score(stage, refinement, extraction)
        summary = dual_scalar_rii_summary(extraction, expected; residual_tol=residual_tol, match_atol=match_atol)
        max_correction = hasproperty(extraction, :refinement_corrections) && !isempty(extraction.refinement_corrections) ?
            maximum(extraction.refinement_corrections) : NaN
        ratio_text = hasproperty(extraction, :refinement_ratios) && !isempty(extraction.refinement_ratios) ?
            join((@sprintf("%.2e", ratio) for ratio in extraction.refinement_ratios), ",") : "n/a"
        label = "$(stage) $(refinement)"
        if print_rows
            @printf(
                "  %-26s count=%d inside=%d good=%d matched=%d spurious=%d max=%.3e max_correction=%.3e newton_ratios=%s\n",
                label,
                extraction.count_estimate,
                summary.inside,
                summary.good,
                summary.matched,
                summary.spurious_good,
                summary.max_residual,
                max_correction,
                ratio_text,
            )
        end
        (
            stage=stage,
            refinement=refinement,
            count=extraction.count_estimate,
            inside=summary.inside,
            good=summary.good,
            matched=summary.matched,
            spurious=summary.spurious_good,
            max_residual=summary.max_residual,
            max_correction=max_correction,
            newton_ratios=hasproperty(extraction, :refinement_ratios) ? extraction.refinement_ratios : Float64[],
        )
    end

    rows = NamedTuple[]
    extractions0 = Dict{Symbol, Any}()
    for refinement in refinement_modes
        extraction = extract_reduced_nep(ctx, trial, chart, config(refinement))
        extractions0[refinement] = extraction
        push!(rows, score(:initial, refinement, extraction))
    end

    updated_trial, stats = residual_laurent_update(ctx, trial, extractions0[first(refinement_modes)], chart, update_config)
    if print_rows
        @printf(
            "  residual update: ranks=(%d,%d) candidate_cols=(%d,%d) basis=(%d,%d)\n",
            stats.right_residual_rank,
            stats.left_residual_rank,
            stats.right_candidate_cols,
            stats.left_candidate_cols,
            size(updated_trial.X, 2),
            size(updated_trial.Y, 2),
        )
    end
    for refinement in refinement_modes
        push!(rows, score(:updated, refinement, extract_reduced_nep(ctx, updated_trial, chart, config(refinement))))
    end
    (rows=rows, expected=length(expected), initial_basis=size(trial.X, 2), updated_basis=size(updated_trial.X, 2))
end

function sorted_unique_values(values; atol=1e-6)
    unique_values(sort(ComplexF64.(values); by=z -> (real(z), imag(z))); atol=atol)
end

function disk_grid_centers(center, radius, spacing)
    kmax = floor(Int, radius / spacing)
    offsets = spacing .* (-kmax:kmax)
    xs = real(center) .+ offsets
    ys = imag(center) .+ offsets
    ComplexF64[
        x + im * y
        for x in xs, y in ys
        if abs((x + im * y) - center) <= radius
    ]
end

function run_dual_local_chart_sweep_analytic(;
    cases=(scalar_sine_case(), scalar_cosine_case(), scalar_shifted_sine_case(), scalar_expm1_case()),
    outer_center=0.0 + 0.0im,
    outer_radius=20.0,
    operator_builder=similarity_analytic_tools,
    operator_label="similarity",
    centers=nothing,
    radii=(0.5, 0.8, 1.2),
    basis_moments=4,
    basis_nodes=16,
    rii_nodes=256,
    update_moment_count=1,
    iterations=1,
    basis_ranktol=0.5,
    compression_ranktol=1e-10,
    residual_ranktol=1e-10,
    determinant_nodes=1024,
    determinant_capacity=96,
    extractor=:ss_counted,
    reduced_moments=16,
    reduced_nodes=1024,
    reduced_ranktol=1e-10,
    reduced_ss_mode=:similarity,
    loewner_points=4,
    loewner_radius=1.6,
    loewner_phase=0.0,
    residual_normalization=:vector,
    reduced_refinement=:none,
    refinement_steps=4,
    component_scaling=:contour_max,
    component_scaling_nodes=64,
    residual_tol=1e-8,
    match_atol=1e-6,
    biorthogonalize=false,
    update_mode=:moment_compressed,
    selection=:matched,
    skip_empty_expected=centers === nothing,
    print_charts=true,
    print_summary=true,
    pipeline_config=nothing,
    basis_config=nothing,
    extraction_config=nothing,
    update_config=nothing,
)
    if pipeline_config !== nothing
        if basis_config !== nothing || extraction_config !== nothing || update_config !== nothing
            error("pass either pipeline_config or individual basis/extraction/update configs")
        end
        basis_config = pipeline_config.basis
        extraction_config = pipeline_config.extractor
        update_config = pipeline_config.update
    end
    if basis_config !== nothing
        basis_moments = basis_config.moments
        basis_nodes = basis_config.nodes
        basis_ranktol = basis_config.ranktol
        biorthogonalize = basis_config.biorthogonalize
    end
    if extraction_config !== nothing
        extractor = extraction_config.extractor
        determinant_nodes = extraction_config.determinant_nodes
        determinant_capacity = extraction_config.determinant_capacity
        reduced_moments = extraction_config.reduced_moments
        reduced_nodes = extraction_config.reduced_nodes
        reduced_ranktol = extraction_config.reduced_ranktol
        reduced_ss_mode = extraction_config.reduced_ss_mode
        loewner_points = extraction_config.loewner_points
        loewner_radius = extraction_config.loewner_radius
        loewner_phase = extraction_config.loewner_phase
        residual_normalization = extraction_config.residual_normalization
        reduced_refinement = extraction_config.refinement
        refinement_steps = extraction_config.refinement_steps
    end
    if update_config !== nothing
        rii_nodes = update_config.rii_nodes
        update_moment_count = update_config.moment_count
        compression_ranktol = update_config.compression_ranktol
        residual_ranktol = update_config.residual_ranktol
        update_mode = update_config.mode
        biorthogonalize = update_config.biorthogonalize
    end

    _, _, _, _, expected_roots = operator_builder(cases; component_scales=nothing)
    expected_global = expected_roots(outer_center, outer_radius)
    chart_centers = centers === nothing ? expected_global : ComplexF64.(centers)
    chart_centers = sorted_unique_values(chart_centers; atol=match_atol)
    source = centers === nothing ? "known roots" : "supplied centers"
    purpose = if centers === nothing
        "supervised exact-root centers; validates local chart sufficiency"
    elseif selection === :residual && !skip_empty_expected
        "residual-scored supplied centers; validates automatic chart-cover merging"
    else
        "supplied centers; validates local chart sufficiency"
    end

    if print_summary
        println()
        println("Dual local-chart sweep analytic update")
        println("  operator=$operator_label; centers=$source; $purpose")
        @printf(
            "  outer_center=%.6g%+.6gi outer_radius=%.3g expected_unique=%d radii=%s residual_normalization=%s component_scaling=%s\n",
            real(outer_center),
            imag(outer_center),
            outer_radius,
            length(expected_global),
            string(collect(radii)),
            string(residual_normalization),
            string(component_scaling),
        )
    end

    found = ComplexF64[]
    found_entries = NamedTuple[]
    records = NamedTuple[]
    selected_records = NamedTuple[]
    for chart_center in chart_centers
        best_record = nothing
        best_values = ComplexF64[]
        best_entries = NamedTuple[]
        best_score = (-1, -1, Inf)
        for radius in radii
            local_expected = expected_roots(chart_center, radius)
            skip_empty_expected && isempty(local_expected) && continue
            result = try
                run_dual_moment_compressed_rii_analytic_iteration(;
                    cases=cases,
                    center=chart_center,
                    radius=radius,
                    operator_builder=operator_builder,
                    operator_label=operator_label,
                    basis_moments=basis_moments,
                    basis_nodes=basis_nodes,
                    rii_nodes=rii_nodes,
                    update_moment_count=update_moment_count,
                    iterations=iterations,
                    basis_ranktol=basis_ranktol,
                    compression_ranktol=compression_ranktol,
                    residual_ranktol=residual_ranktol,
                    determinant_nodes=determinant_nodes,
                    determinant_capacity=determinant_capacity,
                    extractor=extractor,
                    reduced_moments=reduced_moments,
                    reduced_nodes=reduced_nodes,
                    reduced_ranktol=reduced_ranktol,
                    reduced_ss_mode=reduced_ss_mode,
                    loewner_points=loewner_points,
                    loewner_radius=loewner_radius,
                    loewner_phase=loewner_phase,
                    residual_normalization=residual_normalization,
                    reduced_refinement=reduced_refinement,
                    refinement_steps=refinement_steps,
                    component_scaling=component_scaling,
                    component_scaling_nodes=component_scaling_nodes,
                    residual_tol=residual_tol,
                    match_atol=match_atol,
                    biorthogonalize=biorthogonalize,
                    update_mode=update_mode,
                    basis_config=basis_config,
                    extraction_config=extraction_config,
                    update_config=update_config,
                    verbose=false,
                )
            catch err
                record = (
                    center=chart_center,
                    radius=radius,
                    local_expected=length(local_expected),
                    inside=0,
                    good=0,
                    matched=0,
                    max_residual=Inf,
                    count_estimate=0,
                    count_error=Inf,
                    failed=true,
                    error=string(typeof(err)),
                )
                push!(records, record)
                continue
            end
            extraction = result.extraction
            if extraction === nothing
                record = (
                    center=chart_center,
                    radius=radius,
                    local_expected=length(local_expected),
                    inside=0,
                    good=0,
                    matched=0,
                    max_residual=Inf,
                    count_estimate=0,
                    count_error=Inf,
                    failed=true,
                    error="empty_basis",
                )
                push!(records, record)
                continue
            end
            values = good_extraction_values(extraction; residual_tol=residual_tol)
            entries = good_extraction_entries(extraction, chart_center, radius; residual_tol=residual_tol)
            matched = match_expected_count(values, local_expected; atol=match_atol)
            summary = dual_scalar_rii_summary(extraction, local_expected; residual_tol=residual_tol, match_atol=match_atol)
            record = (
                center=chart_center,
                radius=radius,
                local_expected=length(local_expected),
                inside=summary.inside,
                good=summary.good,
                matched=matched,
                max_residual=summary.max_residual,
                count_estimate=getproperty(extraction, :count_estimate),
                count_error=getproperty(extraction, :count_error),
                failed=false,
                error="",
            )
            push!(records, record)
            score = if selection === :matched
                (matched, summary.good, -summary.max_residual)
            elseif selection === :residual
                (summary.good, matched, -summary.max_residual)
            else
                error("unknown local chart selection mode: $selection")
            end
            if score > best_score
                best_score = score
                best_record = record
                best_values = values
                best_entries = entries
            end
        end
        if best_record === nothing
            if print_charts
                @printf(
                    "    center=%+.6g%+.6gi no usable local chart\n",
                    real(chart_center),
                    imag(chart_center),
                )
            end
            continue
        end
        append!(found, best_values)
        append!(found_entries, best_entries)
        push!(selected_records, best_record)
        if print_charts
            @printf(
                "    center=%+.6g%+.6gi best_r=%.3g local=%d good=%d matched=%d max_res=%.3e\n",
                real(chart_center),
                imag(chart_center),
                best_record.radius,
                best_record.local_expected,
                best_record.good,
                best_record.matched,
                best_record.max_residual,
            )
        end
    end

    found_unique = sorted_unique_values(found; atol=match_atol)
    matched_global = match_expected_count(found_unique, expected_global; atol=match_atol)
    support_clusters = chart_entry_clusters(found_entries; atol=match_atol)
    support2_values = supported_cluster_values(support_clusters; min_support=2)
    support2_global_values = globally_supported_cluster_values(
        support_clusters,
        outer_center,
        outer_radius;
        min_support=2,
        boundary_margin=10 * match_atol,
    )
    support2_matched = match_expected_count(support2_values, expected_global; atol=match_atol)
    support2_global_matched = match_expected_count(support2_global_values, expected_global; atol=match_atol)
    if print_summary
        @printf(
            "  union_good=%d matched_global=%d/%d support2_good=%d support2_matched=%d/%d support2_global_good=%d support2_global_matched=%d/%d\n",
            length(found_unique),
            matched_global,
            length(expected_global),
            length(support2_values),
            support2_matched,
            length(expected_global),
            length(support2_global_values),
            support2_global_matched,
            length(expected_global),
        )
    end
    (
        records=records,
        selected_records=selected_records,
        found=found_unique,
        found_entries=found_entries,
        support_clusters=support_clusters,
        support2_found=support2_values,
        support2_global_found=support2_global_values,
        expected=expected_global,
        matched=matched_global,
        support2_matched=support2_matched,
        support2_global_matched=support2_global_matched,
        centers=chart_centers,
    )
end

function run_dual_grid_chart_cover_analytic(;
    outer_center=0.0 + 0.0im,
    outer_radius=20.0,
    operator_label="similarity",
    spacing=2.4,
    chart_radius=1.8,
    chart_radii=(chart_radius,),
    print_summary=true,
    kwargs...,
)
    centers = disk_grid_centers(outer_center, outer_radius, spacing)
    if print_summary
        println()
        @printf(
            "Grid chart cover: outer_center=%.6g%+.6gi outer_radius=%.3g spacing=%.3g chart_radii=%s centers=%d\n",
            real(outer_center),
            imag(outer_center),
            outer_radius,
            spacing,
            string(collect(chart_radii)),
            length(centers),
        )
    end
    run_dual_local_chart_sweep_analytic(;
        outer_center=outer_center,
        outer_radius=outer_radius,
        operator_label=operator_label,
        centers=centers,
        radii=chart_radii,
        selection=:residual,
        skip_empty_expected=false,
        print_charts=false,
        print_summary=print_summary,
        kwargs...,
    )
end

function run_dual_grid_chart_cover_triangular_analytic(;
    coupling=1.0,
    kwargs...,
)
    run_dual_grid_chart_cover_analytic(;
        operator_builder=triangular_operator_builder(; coupling=coupling),
        operator_label="triangular(coupling=$coupling)",
        kwargs...,
    )
end

function run_local_chart_refinement_comparison(;
    cases=(scalar_sine_case(), scalar_cosine_case(), scalar_shifted_sine_case(), scalar_expm1_case()),
    outer_center=0.0 + 0.0im,
    outer_radius=10.0,
    operator_builder=triangular_operator_builder(; coupling=10.0),
    operator_label="triangular(coupling=10.0)",
    radii=(1.2, 1.8),
    refinement_modes=(:none, :scalar_newton),
    iterations=1,
    basis_ranktol=1e-8,
    compression_ranktol=1e-10,
    residual_ranktol=1e-10,
    basis_nodes=24,
    rii_nodes=256,
    determinant_nodes=512,
    reduced_nodes=512,
    residual_normalization=:vector,
    component_scaling=:contour_max,
    residual_tol=1e-8,
    match_atol=1e-6,
)
    println()
    println("Local chart reduced-refinement comparison")
    println("  supervised chart centers; isolates reduced scalar cleanup from chart-cover misses")
    summaries = NamedTuple[]
    for refinement in refinement_modes
        result = run_dual_local_chart_sweep_analytic(;
            cases=cases,
            outer_center=outer_center,
            outer_radius=outer_radius,
            operator_builder=operator_builder,
            operator_label=operator_label,
            radii=radii,
            iterations=iterations,
            basis_ranktol=basis_ranktol,
            compression_ranktol=compression_ranktol,
            residual_ranktol=residual_ranktol,
            basis_nodes=basis_nodes,
            rii_nodes=rii_nodes,
            determinant_nodes=determinant_nodes,
            reduced_nodes=reduced_nodes,
            residual_normalization=residual_normalization,
            component_scaling=component_scaling,
            residual_tol=residual_tol,
            match_atol=match_atol,
            reduced_refinement=refinement,
            print_charts=false,
        )
        summary = (
            refinement=refinement,
            matched=result.matched,
            expected=length(result.expected),
            found=length(result.found),
        )
        push!(summaries, summary)
        @printf(
            "  refinement=%s found=%d matched=%d/%d\n",
            string(refinement),
            summary.found,
            summary.matched,
            summary.expected,
        )
        support_text = join(
            (
                @sprintf("s%d=%d/%d", item.support, item.matched, item.count)
                for item in support_sweep_counts(result; atol=match_atol)
            ),
            ", ",
        )
        println("    support sweep: $support_text")
    end
    summaries
end

function run_dual_multiple_root_analytic_stress(;
    cases=(scalar_squared_sine_case(),),
    center=0.0 + 0.0im,
    radius=10.0,
    operator_builder=similarity_analytic_tools,
    operator_label="similarity",
    basis_moments=8,
    basis_nodes=64,
    rii_nodes=256,
    update_moment_count=1,
    iterations=1,
    basis_ranktol=1e-10,
    determinant_nodes=2048,
    determinant_capacity=64,
    extractor=:ss_counted,
    reduced_moments=16,
    reduced_nodes=1024,
    residual_normalization=:vector,
    component_scaling=:contour_max,
    residual_tol=1e-8,
    match_atol=1e-6,
)
    labels = join((case.name for case in cases), ",")
    unique_expected = begin
        _, _, _, _, expected_roots = operator_builder(cases; component_scales=nothing)
        expected_roots(center, radius)
    end
    algebraic_expected = expected_roots_counting_multiplicity(cases, center, radius)
    result = run_dual_moment_compressed_rii_analytic_iteration(;
        cases=cases,
        center=center,
        radius=radius,
        operator_builder=operator_builder,
        operator_label=operator_label,
        basis_moments=basis_moments,
        basis_nodes=basis_nodes,
        rii_nodes=rii_nodes,
        update_moment_count=update_moment_count,
        iterations=iterations,
        basis_ranktol=basis_ranktol,
        determinant_nodes=determinant_nodes,
        determinant_capacity=determinant_capacity,
        extractor=extractor,
        reduced_moments=reduced_moments,
        reduced_nodes=reduced_nodes,
        residual_normalization=residual_normalization,
        component_scaling=component_scaling,
        residual_tol=residual_tol,
        match_atol=match_atol,
        verbose=false,
    )
    extraction = result.extraction
    values = good_extraction_values(extraction; residual_tol=residual_tol)
    unique_matched = match_expected_count(values, unique_expected; atol=match_atol)
    algebraic_matched = match_expected_count_with_multiplicity(values, algebraic_expected; atol=match_atol)
    println()
    println("Dual multiple-root analytic stress: [$labels]")
    println("  diagnostic only: repeated/shared roots test algebraic count without making Jordan chains a core requirement")
    @printf(
        "  operator=%s radius=%.3g unique_expected=%d algebraic_expected=%d count_estimate=%d good=%d unique_matched=%d algebraic_matched=%d max_res=%.3e\n",
        operator_label,
        radius,
        length(unique_expected),
        length(algebraic_expected),
        extraction === nothing ? 0 : extraction.count_estimate,
        length(values),
        unique_matched,
        algebraic_matched,
        extraction === nothing || !any(extraction.inside) ? Inf : maximum(extraction.residuals[extraction.inside]),
    )
    result
end

function run_dual_reduced_polynomial_control(;
    name="many_eigenvalue_nonnormal_polynomial",
    make_problem=many_eigenvalue_nonnormal_polynomial_problem,
    basis_moments=5,
    basis_nodes=64,
    basis_ranktol=1e-10,
    residual_tol=1e-7,
    match_atol=1e-6,
    refinement_modes=(:none, :block_newton),
    newton_steps=2,
    print_rows=true,
)
    problem = make_problem()
    coeffs, center, radius, n = problem[1], problem[2], problem[3], problem[4]
    expected = length(problem) >= 5 ? ComplexF64.(problem[5]) : companion_reference(coeffs, center, radius)
    z_nodes, z_weights = circular_rule(center, radius, basis_nodes)
    Random.seed!(9501)
    Xprobe = rand(ComplexF64, n, n)
    Wprobe = rand(ComplexF64, n, n)
    Tsolve = (z, B) -> polynomial_matrix(coeffs, z) \ B
    right_moments = initial_moments_generic_scaled(Tsolve, Xprobe, z_nodes, z_weights, center, radius, basis_moments)
    left_moments = initial_adjoint_moments_polynomial_scaled(coeffs, Wprobe, z_nodes, z_weights, center, radius, basis_moments)
    Xbasis, right_singulars = moment_block_basis(right_moments, basis_moments; ranktol=basis_ranktol)
    Ybasis, left_singulars = moment_block_basis(left_moments, basis_moments; ranktol=basis_ranktol)
    Xbi, Ybi, cross_singulars = biorthogonalize_bases(Xbasis, Ybasis)

    print_rows && begin
        println()
        println("Dual reduced polynomial control: $name")
        println("  reduced polynomial extraction from left/right moment-filtered physical spaces")
        @printf("  root matching tolerance: %.1e\n", match_atol)
    end
    rows = NamedTuple[]
    for (label, Xtest, Ytest) in (("dual", Xbasis, Ybasis), ("dual_biorth", Xbi, Ybi), ("galerkin", Xbasis, Xbasis))
        if size(Xtest, 2) == 0 || size(Ytest, 2) == 0 || size(Xtest, 2) != size(Ytest, 2)
            print_rows && println("  mode=$label skipped: incompatible reduced basis sizes")
            continue
        end
        for refinement in refinement_modes
            extraction = reduced_polynomial_extraction(
                coeffs,
                Xtest,
                Ytest,
                center,
                radius;
                refinement=refinement,
                newton_steps=newton_steps,
            )
            inside = extraction.inside
            good = inside .& (extraction.residuals .<= residual_tol)
            matched = match_expected_count(extraction.values[good], expected; atol=match_atol)
            spurious_good = max(count(good) - matched, 0)
            ratio_text = isempty(extraction.refinement_ratios) ? "n/a" : join((@sprintf("%.2e", r) for r in extraction.refinement_ratios), ",")
            reduced_max = any(inside) ? maximum(extraction.reduced_residuals[inside]) : Inf
            original_max = any(inside) ? maximum(extraction.residuals[inside]) : Inf
            push!(
                rows,
                (
                    mode=Symbol(label),
                    refinement=refinement,
                    expected=length(expected),
                    returned_inside=count(inside),
                    good=count(good),
                    matched=matched,
                    spurious_good=spurious_good,
                    reduced_max=reduced_max,
                    original_max=original_max,
                    right_basis=size(Xtest, 2),
                    left_basis=size(Ytest, 2),
                    newton_ratios=Float64.(extraction.refinement_ratios),
                ),
            )
            print_rows && @printf(
                    "  mode=%s refinement=%s expected=%d returned_inside=%d good=%d matched=%d spurious_good=%d reduced_max=%.3e original_max=%.3e basis=(%d,%d) newton_ratios=%s\n",
                    label,
                    string(refinement),
                    length(expected),
                    count(inside),
                    count(good),
                    matched,
                    spurious_good,
                    reduced_max,
                    original_max,
                    size(Xtest, 2),
                    size(Ytest, 2),
                    ratio_text,
                )
        end
    end
    singular_ratios = (
        right=isempty(right_singulars) ? NaN : right_singulars[end] / right_singulars[1],
        left=isempty(left_singulars) ? NaN : left_singulars[end] / left_singulars[1],
        cross=isempty(cross_singulars) ? NaN : cross_singulars[end] / cross_singulars[1],
    )
    print_rows && @printf(
            "    basis singular ratios right=%.3e left=%.3e cross=%.3e\n",
            singular_ratios.right,
            singular_ratios.left,
            singular_ratios.cross,
        )
    (
        rows=rows,
        expected=length(expected),
        singular_ratios=singular_ratios,
    )
end

function match_expected_count(values, expected; atol)
    isempty(expected) && return 0
    count(expected) do λ
        !isempty(values) && minimum(abs.(values .- λ)) <= atol
    end
end

function match_expected_count_with_multiplicity(values, expected; atol)
    isempty(expected) && return 0
    used = falses(length(values))
    matched = 0
    for λ in expected
        best = 0
        best_distance = Inf
        for j in eachindex(values)
            used[j] && continue
            distance = abs(values[j] - λ)
            if distance < best_distance
                best = j
                best_distance = distance
            end
        end
        if best != 0 && best_distance <= atol
            used[best] = true
            matched += 1
        end
    end
    matched
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
    (
        name=name,
        expected=length(expected),
        probe_cols=probe_cols,
        left_cols=ell,
        moment_count=moment_count,
        maxrank=maxrank,
        feast_returned=length(feast_values),
        feast_converged=count(feast_residuals .<= residual_tol),
        feast_max_residual=isempty(feast_residuals) ? Inf : maximum(feast_residuals),
        wide_feast_returned=length(wide_feast_values),
        wide_feast_converged=count(wide_feast_residuals .<= residual_tol),
        wide_feast_max_residual=isempty(wide_feast_residuals) ? Inf : maximum(wide_feast_residuals),
        initial=h0,
        final=hend,
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

include("policy.jl")
include("experiment_matrix.jl")
include("fused_hard_cases.jl")

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
    run_dual_reduced_polynomial_control()
    run_dual_reduced_polynomial_control(;
        name="dual_sensitive_polynomial",
        make_problem=dual_sensitive_polynomial_problem,
        basis_nodes=48,
        basis_ranktol=1e-10,
        residual_tol=1e-6,
        match_atol=1e-3,
    )
    run_dual_scalar_rii_polynomial_experiment(;
        name="dual_sensitive_polynomial_bad_initial",
        make_problem=dual_sensitive_polynomial_problem,
        basis_nodes=6,
        rii_nodes=48,
        iterations=1,
        basis_ranktol=1e-6,
        residual_ranktol=1e-10,
        residual_tol=1e-6,
        match_atol=1e-3,
    )
    run_dual_moment_compressed_rii_polynomial_experiment(;
        name="dual_sensitive_polynomial_bad_initial",
        make_problem=dual_sensitive_polynomial_problem,
        basis_nodes=6,
        rii_nodes=48,
        update_moments=(1, 2),
        basis_ranktol=1e-6,
        residual_ranktol=1e-10,
        residual_tol=1e-6,
        match_atol=1e-3,
    )
    run_dual_moment_compressed_rii_polynomial_experiment(;
        name="many_eigenvalue_nonnormal_rank_deficient_initial",
        make_problem=many_eigenvalue_nonnormal_polynomial_problem,
        basis_nodes=6,
        rii_nodes=48,
        update_moments=(1,),
        basis_ranktol=1e-1,
        residual_ranktol=1e-10,
        residual_tol=1e-7,
        match_atol=1e-6,
    )
    run_dual_moment_compressed_rii_analytic_iteration(;
        cases=(scalar_sine_case(), scalar_cosine_case(), scalar_shifted_sine_case()),
        radius=20.0,
        basis_moments=4,
        basis_nodes=8,
        rii_nodes=256,
        update_moment_count=1,
        iterations=1,
        basis_ranktol=0.5,
        determinant_nodes=1024,
        determinant_capacity=80,
        extractor=:ss_counted,
        reduced_moments=16,
        reduced_nodes=1024,
        residual_tol=1e-8,
        match_atol=1e-6,
    )
    run_reduced_loewner_extractor_comparison()
    run_dual_grid_chart_cover_analytic(;
        spacing=2.4,
        chart_radius=1.8,
        basis_nodes=32,
        rii_nodes=512,
        iterations=2,
        basis_ranktol=1e-8,
        determinant_nodes=1024,
        reduced_nodes=1024,
        residual_normalization=:vector,
        component_scaling=:contour_max,
        residual_tol=1e-8,
        match_atol=1e-6,
    )
    run_dual_grid_chart_cover_triangular_analytic(;
        coupling=10.0,
        spacing=2.4,
        chart_radii=(0.8, 1.2, 1.8, 2.4, 3.0),
        basis_nodes=32,
        rii_nodes=512,
        iterations=2,
        basis_ranktol=1e-8,
        determinant_nodes=1024,
        reduced_nodes=1024,
        residual_normalization=:vector,
        component_scaling=:contour_max,
        residual_tol=1e-8,
        match_atol=1e-6,
    )
    run_dual_multiple_root_analytic_stress(;
        radius=10.0,
        basis_moments=8,
        basis_nodes=64,
        iterations=1,
        determinant_capacity=64,
        residual_tol=1e-8,
        match_atol=1e-6,
    )
    run_dual_multiple_root_analytic_stress(;
        cases=(scalar_sine_case(), scalar_expm1_case()),
        operator_builder=triangular_operator_builder(; coupling=3.0),
        operator_label="triangular(coupling=3.0)",
        radius=1.0,
        basis_moments=4,
        basis_nodes=32,
        iterations=1,
        determinant_capacity=16,
        reduced_moments=8,
        reduced_nodes=512,
        residual_tol=1e-8,
        match_atol=1e-6,
    )
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
    run_dual_reduced_determinant_diagonal_stress()
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
