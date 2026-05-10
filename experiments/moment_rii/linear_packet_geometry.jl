# Linear FEAST packet geometry diagnostics.
#
# This file deliberately uses dense eigendecompositions. It is not an
# implementation path; it pins what the "packet" monitor should mean in the
# linear lower rung where the Riesz projector is canonical.

function linear_standard_packet_projector(A, center, radius)
    F = eigen(A)
    inside = FEASTSolver.in_contour(F.values, center, radius)
    X = Matrix(F.vectors[:, inside])
    left = eigen(A')
    selected = falses(length(left.values))
    for λ in F.values[inside]
        index = argmin(abs.(conj(λ) .- left.values) .+ (selected .* Inf))
        selected[index] = true
    end
    Y = Matrix(left.vectors[:, selected])
    P = X * pinv(Y' * X) * Y'
    P, X, Y, ComplexF64.(F.values[inside])
end

function linear_dual_packet_projector(A, B, center, radius)
    F = eigen(A, B)
    inside = FEASTSolver.in_contour(F.values, center, radius)
    X = Matrix(F.vectors[:, inside])
    left = eigen(A', B')
    selected = falses(length(left.values))
    for λ in F.values[inside]
        index = argmin(abs.(conj(λ) .- left.values) .+ (selected .* Inf))
        selected[index] = true
    end
    Y = Matrix(left.vectors[:, selected])
    P = X * pinv(Y' * B * X) * Y' * B
    Pleft = Y * pinv(X' * B' * Y) * X' * B'
    P, Pleft, X, Y, ComplexF64.(F.values[inside])
end

linear_b_oblique_packet_projector(X, Y, B) = X * pinv(Y' * B * X) * Y' * B

function contour_standard_filter_matrix(A, contour)
    n = size(A, 1)
    P = zeros(ComplexF64, n, n)
    Iₙ = Matrix{ComplexF64}(I, n, n)
    for (z, w) in zip(contour.nodes, contour.weights)
        P .+= w .* ((z .* Iₙ .- A) \ Iₙ)
    end
    P
end

function contour_dual_right_filter_matrix(A, B, contour)
    n = size(A, 1)
    P = zeros(ComplexF64, n, n)
    for (z, w) in zip(contour.nodes, contour.weights)
        P .+= w .* ((z .* B .- A) \ B)
    end
    P
end

function contour_dual_left_filter_matrix(A, B, contour)
    n = size(A, 1)
    P = zeros(ComplexF64, n, n)
    for (z, w) in zip(contour.nodes, contour.weights)
        P .+= conj(w) .* ((z .* B .- A)' \ B')
    end
    P
end

function linear_packet_projector_diagnostic(;
    n=9,
    center=1.0 + 0.0im,
    radius=0.62,
    nodes=256,
    seed=20260509,
    ranktol=1e-10,
    print_rows=true,
)
    Random.seed!(seed)
    λ = ComplexF64[-1.2, -0.6, 0.25, 0.72, 1.02, 1.33, 1.78, 2.3, 3.0]
    V = ComplexF64[exp(0.11im * i * j) / (1 + abs(i - j)) for i in 1:n, j in 1:n]
    A = V * Diagonal(λ) * inv(V)
    B = Diagonal(ComplexF64.(1 .+ 0.07 .* (1:n)))
    Adual = B * A
    contour = circular_contour_trapezoidal(center, radius, nodes)

    Pstd, Xstd, Ystd, values = linear_standard_packet_projector(A, center, radius)
    Pstd_filter = contour_standard_filter_matrix(A, contour)
    Qstd_filter, _ = physical_basis_from_columns(Pstd_filter * rand(ComplexF64, n, length(values) + 2); ranktol=ranktol)
    standard = (
        expected=length(values),
        idempotence=norm(Pstd * Pstd - Pstd),
        filter_gap=norm(Pstd_filter - Pstd),
        monitor_gap=norm(oblique_packet_projector(Xstd, Ystd; ranktol=ranktol) - Pstd),
        subspace_gap=subspace_projection_gap(Qstd_filter, Xstd),
    )

    Pdual, Pdual_left, Xdual, Ydual, values_dual = linear_dual_packet_projector(Adual, B, center, radius)
    Pdual_filter = contour_dual_right_filter_matrix(Adual, B, contour)
    Pdual_left_filter = contour_dual_left_filter_matrix(Adual, B, contour)
    Pmonitor = linear_b_oblique_packet_projector(Xdual, Ydual, B)
    Qdual_filter, _ = physical_basis_from_columns(Pdual_filter * rand(ComplexF64, n, length(values_dual) + 2); ranktol=ranktol)
    Qdual_left_filter, _ = physical_basis_from_columns(Pdual_left_filter * rand(ComplexF64, n, length(values_dual) + 2); ranktol=ranktol)
    dual = (
        expected=length(values_dual),
        idempotence=norm(Pdual * Pdual - Pdual),
        right_filter_gap=norm(Pdual_filter - Pdual),
        left_filter_gap=norm(Pdual_left_filter - Pdual_left),
        monitor_gap=norm(Pmonitor - Pdual),
        right_subspace_gap=subspace_projection_gap(Qdual_filter, Xdual),
        left_subspace_gap=subspace_projection_gap(Qdual_left_filter, Ydual),
    )

    if print_rows
        println()
        println("Linear packet projector diagnostic")
        println("  standard FEAST packet = Riesz spectral projector")
        println("  dual FEAST packet = B-oblique left/right Riesz projector")
        @printf("  standard: expected=%d idempotence=%.3e filter_gap=%.3e monitor_gap=%.3e subspace_gap=%.3e\n",
            standard.expected, standard.idempotence, standard.filter_gap, standard.monitor_gap, standard.subspace_gap)
        @printf("  dual:     expected=%d idempotence=%.3e right_filter_gap=%.3e left_filter_gap=%.3e monitor_gap=%.3e\n",
            dual.expected, dual.idempotence, dual.right_filter_gap, dual.left_filter_gap, dual.monitor_gap)
    end
    (
        standard=standard,
        dual=dual,
        conclusion=:linear_packet_is_riesz_or_oblique_riesz_projector,
    )
end
