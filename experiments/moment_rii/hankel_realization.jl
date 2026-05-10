# Hankel and shifted realizations for moment-based contour extraction.
#
# This is the algebraic extraction layer between contour moments and
# invariant-pair updates.  The runners choose which realization to use; these
# constructors deliberately avoid experiment policy.

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

