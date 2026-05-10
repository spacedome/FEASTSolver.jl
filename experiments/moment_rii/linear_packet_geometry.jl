# Linear FEAST packet geometry diagnostics.
#
# This file deliberately uses dense eigendecompositions. It is not an
# implementation path; it pins what the "packet" monitor should mean in the
# linear lower rung where the Riesz projector is canonical.

function linear_dual_rii_reduction_row(name, A, center, radius; seed, trial_cols, determinant_nodes, rii_nodes)
    n = size(A, 1)
    I_n = Matrix{ComplexF64}(I, n, n)
    Tmatrix(z) = z * I_n - A
    Tderivative(z) = I_n
    Tsolve(z, B) = Tmatrix(z) \ B
    Tadjoint_solve(z, B) = adjoint(Tmatrix(z)) \ B

    Random.seed!(seed)
    Xprobe = Matrix(qr(randn(ComplexF64, n, trial_cols)).Q[:, 1:trial_cols])
    Yprobe = Matrix(qr(randn(ComplexF64, n, trial_cols)).Q[:, 1:trial_cols])
    chart = ContourChart(center, radius)
    ctx = (
        Tmatrix=Tmatrix,
        Tderivative=Tderivative,
        Tsolve=Tsolve,
        Tadjoint_solve=Tadjoint_solve,
        expected=linear_reference_inside(A, center, radius),
        n=n,
    )
    trial = common_square_trial_spaces(TrialSpaces(X=Xprobe, Y=Yprobe, source=:manual_linear_trial))
    extraction = extract_reduced_nep(
        ctx,
        trial,
        chart,
        ReducedExtractorConfig(;
            extractor=:determinant,
            determinant_nodes=determinant_nodes,
            determinant_capacity=max(32, 2 * trial_cols),
            residual_normalization=:vector,
        ),
    )

    z_nodes, z_weights = circular_rule(center, radius, rii_nodes)
    inside = extraction.inside
    values = extraction.values[inside]
    Xright = extraction.right_vectors[:, inside]
    Xleft = extraction.left_vectors[:, inside]
    Qright, Qleft, stats = dual_scalar_rii_step_generic(
        Tsolve,
        Tadjoint_solve,
        Tmatrix,
        values,
        Xright,
        Xleft,
        z_nodes,
        z_weights,
    )

    Pright = zeros(ComplexF64, n, size(Xright, 2))
    Pleft = zeros(ComplexF64, n, size(Xleft, 2))
    for (z, weight) in zip(z_nodes, z_weights)
        Pright .+= weight .* Tsolve(z, Xright)
        Pleft .+= conj(weight) .* Tadjoint_solve(z, Xleft)
    end
    Qright_basis, _ = physical_basis_from_columns(Qright; ranktol=1e-12)
    Qleft_basis, _ = physical_basis_from_columns(Qleft; ranktol=1e-12)
    Pright_basis, _ = physical_basis_from_columns(Pright; ranktol=1e-12)
    Pleft_basis, _ = physical_basis_from_columns(Pleft; ranktol=1e-12)
    (
        name=name,
        n=n,
        expected=length(ctx.expected),
        extracted=count(inside),
        max_extracted_residual=any(inside) ? maximum(extraction.residuals[inside]) : Inf,
        right_projection_gap=opnorm(Qright_basis * Qright_basis' - Pright_basis * Pright_basis'),
        left_projection_gap=opnorm(Qleft_basis * Qleft_basis' - Pleft_basis * Pleft_basis'),
        right_relative_error=norm(Qright - Pright) / max(norm(Pright), eps(Float64)),
        left_relative_error=norm(Qleft - Pleft) / max(norm(Pleft), eps(Float64)),
        right_residual_rank=stats.right_residual_rank,
        left_residual_rank=stats.left_residual_rank,
    )
end

function run_linear_dual_rii_reduction_diagnostic(;
    determinant_nodes=512,
    rii_nodes=512,
    print_rows=true,
)
    A_diag, c_diag, r_diag, _ = many_eigenvalue_diagonal_matrix()
    A_grcar, c_grcar, r_grcar, _ = grcar_linear_matrix()
    rows = [
        linear_dual_rii_reduction_row(
            "many_eigenvalue_diagonal",
            A_diag,
            c_diag,
            r_diag;
            seed=1201,
            trial_cols=10,
            determinant_nodes=determinant_nodes,
            rii_nodes=rii_nodes,
        ),
        linear_dual_rii_reduction_row(
            "grcar_nonnormal",
            A_grcar,
            c_grcar,
            r_grcar;
            seed=1201,
            trial_cols=10,
            determinant_nodes=determinant_nodes,
            rii_nodes=rii_nodes,
        ),
    ]
    result = (rows=rows, max_projection_gap=maximum(max(row.right_projection_gap, row.left_projection_gap) for row in rows))
    if print_rows
        println()
        println("Linear dual RII reduction diagnostic")
        println("  verifies scalar residual-inverse iteration equals the FEAST contour filter for T(z)=zI-A")
        for row in rows
            @printf(
                "  %-24s extracted=%d expected=%d projection_gap=(%.3e, %.3e) relative_error=(%.3e, %.3e)\n",
                row.name,
                row.extracted,
                row.expected,
                row.right_projection_gap,
                row.left_projection_gap,
                row.right_relative_error,
                row.left_relative_error,
            )
        end
    end
    result
end

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

function orthogonal_projector_from_columns(X; ranktol=1e-10)
    Q, _ = physical_basis_from_columns(X; ranktol=ranktol)
    Q * Q'
end

function visible_projector_leakage(T, P)
    n = size(P, 1)
    Q = Matrix{ComplexF64}(I, n, n) - P
    scale = max(opnorm(T), eps(Float64))
    visible_to_hidden = opnorm(P * T * Q) / scale
    hidden_to_visible = opnorm(Q * T * P) / scale
    (
        visible_to_hidden=visible_to_hidden,
        hidden_to_visible=hidden_to_visible,
        max=max(visible_to_hidden, hidden_to_visible),
        idempotence=norm(P * P - P),
    )
end

function visible_projector_leakage_row(label, T, P, Preference; expected=:unknown)
    leakage = visible_projector_leakage(T, P)
    (
        label=label,
        expected=expected,
        leakage=leakage.max,
        visible_to_hidden=leakage.visible_to_hidden,
        hidden_to_visible=leakage.hidden_to_visible,
        idempotence=leakage.idempotence,
        projector_gap=norm(P - Preference),
    )
end

function run_linear_visible_projector_leakage_diagnostic(;
    n=9,
    center=1.0 + 0.0im,
    radius=0.62,
    z=center + 0.31im,
    perturbation=1e-3,
    seed=20260510,
    ranktol=1e-10,
    print_rows=true,
)
    Random.seed!(seed)
    λ = ComplexF64[-1.2, -0.6, 0.25, 0.72, 1.02, 1.33, 1.78, 2.3, 3.0]
    V = ComplexF64[exp(0.11im * i * j) / (1 + abs(i - j)) for i in 1:n, j in 1:n]
    A = V * Diagonal(λ) * inv(V)
    B = Diagonal(ComplexF64.(1 .+ 0.07 .* (1:n)))
    Adual = B * A
    I_n = Matrix{ComplexF64}(I, n, n)

    Pstd, Xstd, Ystd, _ = linear_standard_packet_projector(A, center, radius)
    Tstd = z .* I_n .- A
    Pstd_right_only = orthogonal_projector_from_columns(Xstd; ranktol=ranktol)
    Pstd_perturbed = oblique_packet_projector(
        Xstd .+ perturbation .* randn(ComplexF64, size(Xstd)),
        Ystd;
        ranktol=ranktol,
    )

    Pdual, _, Xdual, Ydual, _ = linear_dual_packet_projector(Adual, B, center, radius)
    # The generalized FEAST projector is spectral for B \ A, not for the raw
    # pencil matrix zB-A.  Measuring leakage on zB-A would incorrectly flag the
    # exact B-oblique spectral projector because T maps the right invariant space
    # through B.
    Tdual = z .* I_n .- (B \ Adual)
    Pdual_right_only = orthogonal_projector_from_columns(Xdual; ranktol=ranktol)
    Pdual_perturbed = linear_b_oblique_packet_projector(
        Xdual .+ perturbation .* randn(ComplexF64, size(Xdual)),
        Ydual,
        B,
    )

    rows = (
        visible_projector_leakage_row(:standard_oblique_exact, Tstd, Pstd, Pstd; expected=:clean),
        visible_projector_leakage_row(:standard_right_orthogonal, Tstd, Pstd_right_only, Pstd; expected=:one_sided_leakage),
        visible_projector_leakage_row(:standard_oblique_perturbed, Tstd, Pstd_perturbed, Pstd; expected=:chart_leakage),
        visible_projector_leakage_row(:dual_b_oblique_exact, Tdual, Pdual, Pdual; expected=:clean),
        visible_projector_leakage_row(:dual_right_orthogonal, Tdual, Pdual_right_only, Pdual; expected=:one_sided_leakage),
        visible_projector_leakage_row(:dual_b_oblique_perturbed, Tdual, Pdual_perturbed, Pdual; expected=:chart_leakage),
    )

    clean = filter(row -> row.expected === :clean, rows)
    one_sided = filter(row -> row.expected === :one_sided_leakage, rows)
    perturbed = filter(row -> row.expected === :chart_leakage, rows)
    result = (
        rows=rows,
        clean_maximum=maximum(row.leakage for row in clean),
        one_sided_minimum=minimum(row.leakage for row in one_sided),
        perturbed_minimum=minimum(row.leakage for row in perturbed),
        conclusion=:visible_projector_leakage_detects_one_sided_and_chart_defects,
    )

    if print_rows
        println()
        println("Linear visible-projector leakage diagnostic")
        println("  off-block leakage P*T*Q and Q*T*P is zero for the spectral projector, not for one-sided right-space projectors")
        for row in rows
            @printf(
                "  %-28s leakage=%.3e projector_gap=%.3e idempotence=%.3e expected=%s\n",
                string(row.label),
                row.leakage,
                row.projector_gap,
                row.idempotence,
                string(row.expected),
            )
        end
    end

    result
end

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
