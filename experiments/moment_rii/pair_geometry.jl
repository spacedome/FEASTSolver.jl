# Invariant-pair and local chart geometry helpers.
#
# These utilities are shared by the exploratory moment-RII algorithms.  Keeping
# them separate from the runners makes the remaining experiment files easier to
# read as algorithm narratives.

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

