# Cross-problem experiment matrix for the moment-RII research line.
#
# This file depends on the prototype functions in run.jl. It deliberately keeps
# the comparison layer separate from those prototypes so we can stabilize the
# algorithmic story without turning every historical helper into public API.

function nearest_expected_distance(values, expected)
    (isempty(values) || isempty(expected)) && return Inf
    maximum(value -> minimum(abs.(value .- expected)), values)
end

function experiment_matrix_row(problem, problem_class, stage, extraction, expected; residual_tol, match_atol, dimension=nothing, notes="")
    if extraction === nothing
        return (
            problem=problem,
            problem_class=problem_class,
            stage=stage,
            dimension=dimension,
            expected=length(expected),
            inside=0,
            good=0,
            matched=0,
            spurious=0,
            max_residual=Inf,
            nearest_expected=Inf,
            success=false,
            notes=notes,
        )
    end
    summary = dual_scalar_rii_summary(extraction, expected; residual_tol=residual_tol, match_atol=match_atol)
    good = extraction.inside .& (extraction.residuals .<= residual_tol)
    nearest = nearest_expected_distance(ComplexF64.(extraction.values[good]), expected)
    (
        problem=problem,
        problem_class=problem_class,
        stage=stage,
        dimension=dimension,
        expected=length(expected),
        inside=summary.inside,
        good=summary.good,
        matched=summary.matched,
        spurious=summary.spurious_good,
        max_residual=summary.max_residual,
        nearest_expected=nearest,
        success=summary.matched == length(expected) && summary.spurious_good == 0,
        notes=notes,
    )
end

function experiment_matrix_support_row(problem, problem_class, stage, result; match_atol, dimension=nothing, notes="")
    (
        problem=problem,
        problem_class=problem_class,
        stage=stage,
        dimension=dimension,
        expected=length(result.expected),
        inside=length(result.found),
        good=length(result.support2_found),
        matched=result.support2_matched,
        spurious=max(length(result.support2_found) - result.support2_matched, 0),
        max_residual=NaN,
        nearest_expected=nearest_expected_distance(ComplexF64.(result.support2_found), result.expected),
        success=result.support2_matched == length(result.expected),
        notes=notes,
    )
end

function print_experiment_matrix(rows)
    println()
    println("Moment-RII experiment matrix")
    println("  success means all expected roots matched with no residual-small spurious values unless noted")
    @printf(
        "  %-32s %-20s %5s %-24s %8s %8s %8s %8s %10s %10s  %s\n",
        "problem",
        "class",
        "n",
        "stage",
        "matched",
        "good",
        "inside",
        "spurious",
        "max_res",
        "near",
        "notes",
    )
    for row in rows
        max_text = isfinite(row.max_residual) ? @sprintf("%.2e", row.max_residual) : string(row.max_residual)
        near_text = isfinite(row.nearest_expected) ? @sprintf("%.2e", row.nearest_expected) : string(row.nearest_expected)
        matched_text = @sprintf("%d/%d", row.matched, row.expected)
        status = row.success ? "ok" : "check"
        dimension_text = row.dimension === nothing ? "?" : string(row.dimension)
        @printf(
            "  %-32s %-20s %5s %-24s %8s %8d %8d %8d %10s %10s  %s %s\n",
            row.problem,
            row.problem_class,
            dimension_text,
            row.stage,
            matched_text,
            row.good,
            row.inside,
            row.spurious,
            max_text,
            near_text,
            status,
            row.notes,
        )
    end
end

function initial_chebyshev_moments_generic_scaled(
    Tsolve,
    Xprobe,
    z_nodes,
    z_weights,
    center,
    radius,
    moment_count;
    coordinate_shift=0.0,
    coordinate_kind=:direct,
)
    n, m = size(Xprobe)
    moments = [zeros(ComplexF64, n, m) for _ in 1:(moment_count + 1)]
    for (z, weight) in zip(z_nodes, z_weights)
        solved = Tsolve(z, Xprobe)
        μ = contour_coordinate(z, center, radius, coordinate_shift, coordinate_kind)
        moments[1] .+= weight .* solved
        moment_count == 0 && continue
        tprev = one(ComplexF64)
        tcurr = μ
        moments[2] .+= (weight * tcurr) .* solved
        for q in 3:length(moments)
            tnext = 2 * μ * tcurr - tprev
            moments[q] .+= (weight * tnext) .* solved
            tprev, tcurr = tcurr, tnext
        end
    end
    moments
end

function reduced_hankel_variant_extraction(
    Tmatrix,
    Tderivative,
    Xbasis,
    Ybasis,
    center,
    radius;
    variant=:hankel,
    reduced_moments=8,
    reduced_nodes=512,
    ranktol=1e-10,
    maxrank=typemax(Int),
    moment_offset=0,
    coordinate_kind=:direct,
    coordinate_shift=0.0,
    residual_normalization=:vector,
    refinement=:none,
    refinement_steps=4,
)
    size(Xbasis, 2) == size(Ybasis, 2) || error("reduced Hankel variant extraction needs square left/right bases")
    variant in (:hankel, :hankel_balanced, :shifted, :balanced_shifted, :chebyshev_shifted) ||
        error("unknown reduced Hankel variant: $variant")
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
    moment_offsets = moment_offset isa Integer ? (moment_offset,) : Tuple(moment_offset)
    max_moment_offset = maximum(moment_offsets)
    if variant === :chebyshev_shifted
        moments = initial_chebyshev_moments_generic_scaled(
            Tsolve_red,
            probe,
            z_nodes,
            z_weights,
            center,
            radius,
            reduced_moments;
            coordinate_shift=coordinate_shift,
            coordinate_kind=coordinate_kind,
        )
        _, Scoord, rank, singular_values = chebyshev_shifted_pair_identity(
            moments,
            reduced_moments;
            ranktol=ranktol,
            maxrank=maxrank,
        )
    else
        moments = initial_moments_generic_scaled(
            Tsolve_red,
            probe,
            z_nodes,
            z_weights,
            center,
            radius,
            reduced_moments;
            coordinate_shift=coordinate_shift,
            coordinate_kind=coordinate_kind,
            moment_offset=max_moment_offset,
        )
        if variant === :hankel
            _, Scoord, rank, singular_values = hankel_pair_identity_offsets(
                moments,
                reduced_moments,
                moment_offsets;
                ranktol=ranktol,
                maxrank=maxrank,
            )
        elseif variant === :hankel_balanced
            _, Scoord, rank, singular_values = balanced_hankel_pair_identity_offsets(
                moments,
                reduced_moments,
                moment_offsets;
                ranktol=ranktol,
                maxrank=maxrank,
            )
        elseif variant === :shifted
            _, Scoord, rank, singular_values = shifted_pair_identity_offsets(
                moments,
                reduced_moments,
                moment_offsets;
                ranktol=ranktol,
                maxrank=maxrank,
            )
        else
            _, Scoord, rank, singular_values = balanced_shifted_pair_identity(
                moments,
                reduced_moments;
                ranktol=ranktol,
                maxrank=maxrank,
            )
        end
    end
    Sλ = lambda_matrix_from_scaled(Scoord, center, radius, coordinate_shift, coordinate_kind)
    values = ComplexF64.(eigvals(Sλ))
    values = refine_determinant_roots(Tred, Tred_derivative, values; step_limit=0.25 * radius)
    finite = finite_eigenvalue_mask(values)
    values = ComplexF64.(values[finite])
    refinement_corrections = Float64[]
    if refinement === :scalar_newton
        values, refinement_corrections = refine_reduced_analytic_triplets(
            Tred,
            Tred_derivative,
            values;
            steps=refinement_steps,
            step_limit=0.25 * radius,
        )
    elseif refinement !== :none
        error("unknown reduced Hankel variant refinement: $refinement")
    end
    left_reduced, right_reduced = reduced_left_right_singular_vectors(Tred, values)
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
        count_estimate=rank,
        count_error=NaN,
        singular_values=Float64.(singular_values),
        variant=variant,
        moment_offset=moment_offset,
        coordinate_kind=coordinate_kind,
        coordinate_shift=coordinate_shift,
        refinement=refinement,
        refinement_corrections=refinement_corrections,
    )
end

function run_matrix_polynomial_case(;
    name,
    problem_class=:polynomial,
    make_problem,
    basis_moments,
    basis_nodes,
    rii_nodes,
    update_moment_count=1,
    basis_ranktol=1e-10,
    compression_ranktol=1e-10,
    residual_ranktol=1e-10,
    residual_tol=1e-8,
    match_atol=1e-6,
    newton_steps=2,
)
    problem = make_problem()
    coeffs, center, radius = problem[1], problem[2], problem[3]
    n = size(coeffs[1], 1)
    expected = length(problem) >= 5 ? ComplexF64.(problem[5]) : companion_reference(coeffs, center, radius)
    z_nodes, z_weights = circular_rule(center, radius, basis_nodes)
    rii_z_nodes, rii_z_weights = circular_rule(center, radius, rii_nodes)

    Random.seed!(9701 + basis_nodes + 17 * basis_moments)
    Xprobe = rand(ComplexF64, n, n)
    Wprobe = rand(ComplexF64, n, n)
    Tsolve = (z, B) -> polynomial_matrix(coeffs, z) \ B
    right_moments = initial_moments_generic_scaled(Tsolve, Xprobe, z_nodes, z_weights, center, radius, basis_moments)
    left_moments = initial_adjoint_moments_polynomial_scaled(coeffs, Wprobe, z_nodes, z_weights, center, radius, basis_moments)
    Xbasis, _ = moment_block_basis(right_moments, basis_moments; ranktol=basis_ranktol)
    Ybasis, _ = moment_block_basis(left_moments, basis_moments; ranktol=basis_ranktol)
    trial = common_square_trial_spaces(TrialSpaces(X=Xbasis, Y=Ybasis, source=:polynomial_initial))
    size(trial.X, 2) == 0 && return NamedTuple[]

    rows = NamedTuple[]
    extraction0 = reduced_polynomial_extraction(coeffs, trial.X, trial.Y, center, radius)
    push!(
        rows,
        experiment_matrix_row(
            name,
            problem_class,
            :initial_extraction,
            extraction0,
            expected;
            residual_tol=residual_tol,
            match_atol=match_atol,
            dimension=n,
            notes="dual projected polynomial NEP",
        ),
    )
    extraction0_refined = reduced_polynomial_extraction(
        coeffs,
        trial.X,
        trial.Y,
        center,
        radius;
        refinement=:block_newton,
        newton_steps=newton_steps,
    )
    push!(
        rows,
        experiment_matrix_row(
            name,
            problem_class,
            :block_newton_cleanup,
            extraction0_refined,
            expected;
            residual_tol=residual_tol,
            match_atol=match_atol,
            dimension=n,
            notes="reduced invariant-pair cleanup",
        ),
    )

    Xupdated, Yupdated, stats = moment_compressed_dual_rii_bases(
        coeffs,
        trial.X,
        trial.Y,
        extraction0,
        rii_z_nodes,
        rii_z_weights,
        center,
        radius;
        moment_count=update_moment_count,
        residual_ranktol=residual_ranktol,
        compression_ranktol=compression_ranktol,
    )
    extraction1 = reduced_polynomial_extraction(coeffs, Xupdated, Yupdated, center, radius)
    push!(
        rows,
        experiment_matrix_row(
            name,
            problem_class,
            :laurent_update,
            extraction1,
            expected;
            residual_tol=residual_tol,
            match_atol=match_atol,
            dimension=n,
            notes=@sprintf("residual ranks=(%d,%d)", stats.right_residual_rank, stats.left_residual_rank),
        ),
    )
    rows
end

function run_matrix_analytic_case(;
    name,
    problem_class=:analytic,
    cases,
    center=0.0 + 0.0im,
    radius,
    operator_builder=similarity_analytic_tools,
    operator_label="similarity",
    iterations=1,
    basis_moments=4,
    basis_nodes=16,
    rii_nodes=128,
    update_moment_count=1,
    basis_ranktol=1e-8,
    compression_ranktol=1e-10,
    residual_ranktol=1e-10,
    determinant_nodes=512,
    determinant_capacity=96,
    extractor=:ss_counted,
    reduced_moments=8,
    reduced_nodes=512,
    reduced_ranktol=1e-10,
    reduced_ss_mode=:similarity,
    loewner_points=4,
    loewner_radius=1.6,
    loewner_phase=0.0,
    residual_normalization=:operator,
    component_scaling=:contour_max,
    residual_tol=1e-8,
    match_atol=1e-6,
)
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
        component_scaling=component_scaling,
        residual_tol=residual_tol,
        match_atol=match_atol,
        verbose=false,
    )
    rows = NamedTuple[]
    for (index, summary) in pairs(result.summaries)
        extraction = hasproperty(result, :extractions) ? result.extractions[index] : nothing
        push!(
            rows,
            (
                problem=name,
                problem_class=problem_class,
                stage=summary.iteration == 0 ? :initial_extraction : Symbol("laurent_update_", summary.iteration),
                dimension=size(result.Xbasis, 1),
                expected=length(result.expected),
                inside=summary.inside,
                good=summary.good,
                matched=summary.matched,
                spurious=summary.spurious_good,
                max_residual=summary.max_residual,
                nearest_expected=extraction === nothing ? NaN :
                    nearest_expected_distance(
                        ComplexF64.(extraction.values[extraction.inside .& (extraction.residuals .<= residual_tol)]),
                        result.expected,
                    ),
                success=summary.matched == length(result.expected) && summary.spurious_good == 0,
                notes=@sprintf("basis=(%d,%d)", size(result.Xbasis, 2), size(result.Ybasis, 2)),
            ),
        )
    end
    rows
end

function hankel_variant_summary(
    label,
    extraction,
    expected;
    residual_tol=1e-8,
    match_atol=1e-6,
    notes="",
)
    summary = dual_scalar_rii_summary(extraction, expected; residual_tol=residual_tol, match_atol=match_atol)
    good = extraction.inside .& (extraction.residuals .<= residual_tol)
    singular_ratio = hasproperty(extraction, :singular_values) && !isempty(extraction.singular_values) ?
        extraction.singular_values[end] / extraction.singular_values[1] : NaN
    (
        label=String(label),
        inside=summary.inside,
        good=summary.good,
        matched=summary.matched,
        expected=length(expected),
        spurious=summary.spurious_good,
        max_residual=summary.max_residual,
        nearest_expected=nearest_expected_distance(ComplexF64.(extraction.values[good]), expected),
        count_estimate=hasproperty(extraction, :count_estimate) ? extraction.count_estimate : length(extraction.values),
        singular_ratio=singular_ratio,
        success=summary.matched == length(expected) && summary.spurious_good == 0,
        notes=notes,
    )
end

function failed_hankel_variant_summary(label, expected, err)
    (
        label=String(label),
        inside=0,
        good=0,
        matched=0,
        expected=length(expected),
        spurious=0,
        max_residual=Inf,
        nearest_expected=Inf,
        count_estimate=0,
        singular_ratio=NaN,
        success=false,
        notes="$(typeof(err)): $err",
    )
end

function print_hankel_variant_sweep(rows)
    println()
    println("Exponential reduced Hankel chart/gauge sweep")
    println("  same physical trial/test spaces; compares reduced realizations against the known exponential root lattice")
    @printf(
        "  %-34s %8s %8s %8s %8s %10s %10s %10s  %s\n",
        "variant",
        "matched",
        "good",
        "inside",
        "spurious",
        "max_res",
        "near",
        "sigma",
        "notes",
    )
    for row in rows
        max_text = isfinite(row.max_residual) ? @sprintf("%.2e", row.max_residual) : string(row.max_residual)
        near_text = isfinite(row.nearest_expected) ? @sprintf("%.2e", row.nearest_expected) : string(row.nearest_expected)
        sigma_text = isfinite(row.singular_ratio) ? @sprintf("%.2e", row.singular_ratio) : string(row.singular_ratio)
        matched_text = @sprintf("%d/%d", row.matched, row.expected)
        status = row.success ? "ok" : "check"
        @printf(
            "  %-34s %8s %8d %8d %8d %10s %10s %10s  %s %s\n",
            row.label,
            matched_text,
            row.good,
            row.inside,
            row.spurious,
            max_text,
            near_text,
            sigma_text,
            status,
            row.notes,
        )
    end
end

function exponential_hankel_sweep_trial(;
    cases=exponential_many_root_cases(),
    center=0.0 + 0.0im,
    radius=10.0,
    basis_moments=5,
    basis_nodes=24,
    basis_ranktol=1e-8,
    component_scaling=:none,
)
    chart = ContourChart(
        center,
        radius;
        component_scaling=component_scaling,
        component_scaling_nodes=64,
    )
    ctx = analytic_context(cases, chart, similarity_analytic_tools)
    trial = initial_dual_trial_spaces(
        ctx,
        chart;
        basis_moments=basis_moments,
        basis_nodes=basis_nodes,
        basis_ranktol=basis_ranktol,
        seed=9801 + round(Int, radius * 10) + 17 * length(cases),
    )
    ctx, chart, trial
end

function run_exponential_hankel_variant_sweep(;
    reduced_moments=16,
    reduced_nodes=768,
    ranktol=1e-10,
    residual_tol=1e-8,
    match_atol=1e-6,
    print_rows=true,
)
    ctx, chart, trial = exponential_hankel_sweep_trial()
    expected = ctx.expected
    rows = NamedTuple[]

    reference_configs = (
        (
            label="loewner_counted",
            config=ReducedExtractorConfig(
                extractor=:loewner_counted,
                determinant_nodes=768,
                determinant_capacity=96,
                reduced_moments=reduced_moments,
                reduced_nodes=reduced_nodes,
                reduced_ranktol=ranktol,
                loewner_points=6,
                loewner_radius=1.3,
                residual_normalization=:vector,
            ),
        ),
        (
            label="ss_counted_similarity",
            config=ReducedExtractorConfig(
                extractor=:ss_counted,
                determinant_nodes=768,
                determinant_capacity=96,
                reduced_moments=reduced_moments,
                reduced_nodes=reduced_nodes,
                reduced_ranktol=ranktol,
                reduced_ss_mode=:similarity,
                residual_normalization=:vector,
            ),
        ),
        (
            label="ss_counted_generalized",
            config=ReducedExtractorConfig(
                extractor=:ss_counted,
                determinant_nodes=768,
                determinant_capacity=96,
                reduced_moments=reduced_moments,
                reduced_nodes=reduced_nodes,
                reduced_ranktol=ranktol,
                reduced_ss_mode=:generalized,
                residual_normalization=:vector,
            ),
        ),
    )
    for item in reference_configs
        try
            extraction = extract_reduced_nep(ctx, trial, chart, item.config)
            push!(rows, hankel_variant_summary(item.label, extraction, expected; residual_tol=residual_tol, match_atol=match_atol, notes="reference"))
        catch err
            push!(rows, failed_hankel_variant_summary(item.label, expected, err))
        end
    end

    specs = [
        (label="hankel_direct_o0", variant=:hankel, offset=0, coord=:direct, shift=0.0),
        (label="hankel_direct_o2", variant=:hankel, offset=2, coord=:direct, shift=0.0),
        (label="hankel_direct_o024", variant=:hankel, offset=(0, 2, 4), coord=:direct, shift=0.0),
        (label="hankel_balanced_o0", variant=:hankel_balanced, offset=0, coord=:direct, shift=0.0),
        (label="hankel_balanced_o024", variant=:hankel_balanced, offset=(0, 2, 4), coord=:direct, shift=0.0),
        (label="hankel_inverse_o0", variant=:hankel, offset=0, coord=:inverse, shift=0.0),
        (label="hankel_mobius_p25", variant=:hankel, offset=0, coord=:mobius, shift=0.25),
        (label="hankel_mobius_m25", variant=:hankel, offset=0, coord=:mobius, shift=-0.25),
        (label="shifted_direct_o0", variant=:shifted, offset=0, coord=:direct, shift=0.0),
        (label="shifted_direct_o024", variant=:shifted, offset=(0, 2, 4), coord=:direct, shift=0.0),
        (label="shifted_inverse_o0", variant=:shifted, offset=0, coord=:inverse, shift=0.0),
        (label="shifted_mobius_p25", variant=:shifted, offset=0, coord=:mobius, shift=0.25),
        (label="balanced_shifted", variant=:balanced_shifted, offset=0, coord=:direct, shift=0.0),
        (label="chebyshev_shifted", variant=:chebyshev_shifted, offset=0, coord=:direct, shift=0.0),
        (label="chebyshev_mobius_p25", variant=:chebyshev_shifted, offset=0, coord=:mobius, shift=0.25),
    ]
    for spec in specs
        try
            extraction = reduced_hankel_variant_extraction(
                ctx.Tmatrix,
                ctx.Tderivative,
                trial.X,
                trial.Y,
                chart.center,
                chart.radius;
                variant=spec.variant,
                reduced_moments=reduced_moments,
                reduced_nodes=reduced_nodes,
                ranktol=ranktol,
                moment_offset=spec.offset,
                coordinate_kind=spec.coord,
                coordinate_shift=spec.shift,
                residual_normalization=:vector,
            )
            notes = "coord=$(spec.coord) offset=$(spec.offset)"
            push!(rows, hankel_variant_summary(spec.label, extraction, expected; residual_tol=residual_tol, match_atol=match_atol, notes=notes))
        catch err
            push!(rows, failed_hankel_variant_summary(spec.label, expected, err))
        end
    end
    print_rows && print_hankel_variant_sweep(rows)
    rows
end

function find_hankel_variant_row(rows, label)
    for row in rows
        row.label == label && return row
    end
    error("missing Hankel variant row: $label")
end

function run_rational_coordinate_boundary_diagnostic(; print_rows=true)
    rows = run_exponential_hankel_variant_sweep(; print_rows=false)
    loewner = find_hankel_variant_row(rows, "loewner_counted")
    inverse = find_hankel_variant_row(rows, "hankel_inverse_o0")
    mobius_plus = find_hankel_variant_row(rows, "hankel_mobius_p25")
    mobius_minus = find_hankel_variant_row(rows, "hankel_mobius_m25")
    shifted_mobius = find_hankel_variant_row(rows, "shifted_mobius_p25")
    chebyshev_mobius = find_hankel_variant_row(rows, "chebyshev_mobius_p25")

    if print_rows
        println()
        println("Rational coordinate boundary diagnostic")
        println("  same exponential chart and trial spaces; asks whether inverse/Mobius coordinates replace Loewner/local charts")
        @printf(
            "  loewner=%d/%d spurious=%d inverse=%d/%d mobius(+)= %d/%d spurious=%d mobius(-)= %d/%d spurious=%d\n",
            loewner.matched,
            loewner.expected,
            loewner.spurious,
            inverse.matched,
            inverse.expected,
            mobius_plus.matched,
            mobius_plus.expected,
            mobius_plus.spurious,
            mobius_minus.matched,
            mobius_minus.expected,
            mobius_minus.spurious,
        )
        @printf(
            "  shifted_mobius=%d/%d chebyshev_mobius=%d/%d conclusion=%s\n",
            shifted_mobius.matched,
            shifted_mobius.expected,
            chebyshev_mobius.matched,
            chebyshev_mobius.expected,
            loewner.success && !mobius_plus.success && !mobius_minus.success ? "rational_coordinates_are_diagnostics" : "check",
        )
    end
    (
        rows=rows,
        loewner=loewner,
        inverse=inverse,
        mobius_plus=mobius_plus,
        mobius_minus=mobius_minus,
        shifted_mobius=shifted_mobius,
        chebyshev_mobius=chebyshev_mobius,
    )
end

function loewner_sweep_row(label, phase, rho, stage, extraction, expected; residual_tol, match_atol, notes="")
    summary = dual_scalar_rii_summary(extraction, expected; residual_tol=residual_tol, match_atol=match_atol)
    good = extraction.inside .& (extraction.residuals .<= residual_tol)
    singular_ratio = hasproperty(extraction, :singular_values) && !isempty(extraction.singular_values) ?
        extraction.singular_values[min(length(extraction.singular_values), max(summary.inside, 1))] / extraction.singular_values[1] : NaN
    (
        label=String(label),
        phase=Float64(phase),
        rho=Float64(rho),
        stage=stage,
        expected=length(expected),
        inside=summary.inside,
        good=summary.good,
        matched=summary.matched,
        spurious=summary.spurious_good,
        max_residual=summary.max_residual,
        nearest_expected=nearest_expected_distance(ComplexF64.(extraction.values[good]), expected),
        count_estimate=hasproperty(extraction, :count_estimate) ? extraction.count_estimate : length(extraction.values),
        singular_ratio=singular_ratio,
        success=summary.matched == length(expected) && summary.spurious_good == 0,
        notes=notes,
    )
end

function loewner_layout_entries(extraction, rho, phase; residual_tol)
    extraction === nothing && return NamedTuple[]
    good = extraction.inside .& (extraction.residuals .<= residual_tol)
    layout_id = ComplexF64(rho, phase)
    [
        (
            value=ComplexF64(extraction.values[j]),
            residual=Float64(extraction.residuals[j]),
            right_residual=Float64(extraction.right_residuals[j]),
            left_residual=Float64(extraction.left_residuals[j]),
            center=layout_id,
            radius=Float64(rho),
            phase=Float64(phase),
        )
        for j in eachindex(extraction.values) if good[j]
    ]
end

function loewner_layout_clusters(entries; atol=1e-6)
    raw_clusters = Vector{Vector{Any}}()
    for entry in entries
        index = findfirst(cluster -> minimum(abs.(entry.value .- [item.value for item in cluster])) <= atol, raw_clusters)
        if index === nothing
            push!(raw_clusters, Any[entry])
        else
            push!(raw_clusters[index], entry)
        end
    end
    map(raw_clusters) do cluster
        centers = unique_values([entry.center for entry in cluster]; atol=0.0)
        values = [entry.value for entry in cluster]
        residuals = [entry.residual for entry in cluster]
        (
            value=sum(values) / length(values),
            support=length(centers),
            observations=length(cluster),
            best_residual=minimum(residuals),
            spread=maximum(abs.(values .- (sum(values) / length(values)))),
        )
    end
end

function loewner_supported_cluster_values(clusters; min_support=2)
    ComplexF64[cluster.value for cluster in clusters if cluster.support >= min_support]
end

function loewner_layout_support_summary(entries, expected; match_atol, max_support=6)
    clusters = loewner_layout_clusters(entries; atol=match_atol)
    [
        (
            support=support,
            count=length(loewner_supported_cluster_values(clusters; min_support=support)),
            matched=match_expected_count(loewner_supported_cluster_values(clusters; min_support=support), expected; atol=match_atol),
            spurious=max(
                length(loewner_supported_cluster_values(clusters; min_support=support)) -
                match_expected_count(loewner_supported_cluster_values(clusters; min_support=support), expected; atol=match_atol),
                0,
            ),
        )
        for support in 1:min(max_support, length(unique_values([entry.center for entry in entries]; atol=0.0)))
    ]
end

function print_loewner_layout_support(label, summaries)
    isempty(summaries) && return
    println("  layout support $(label):")
    for row in summaries
        @printf(
            "    support>=%d count=%d matched=%d spurious=%d\n",
            row.support,
            row.count,
            row.matched,
            row.spurious,
        )
    end
end

function print_loewner_interpolation_sweep(rows; initial_support=NamedTuple[], updated_support=NamedTuple[])
    println()
    println("Loewner interpolation-point sweep")
    println("  same reduced NEP and trial spaces; varies outside-circle interpolation radius/phase")
    @printf(
        "  %-18s %-10s %6s %7s %8s %8s %8s %8s %10s %10s %10s  %s\n",
        "case",
        "stage",
        "rho",
        "phase",
        "matched",
        "good",
        "inside",
        "spurious",
        "max_res",
        "near",
        "sigma",
        "notes",
    )
    for row in rows
        max_text = isfinite(row.max_residual) ? @sprintf("%.2e", row.max_residual) : string(row.max_residual)
        near_text = isfinite(row.nearest_expected) ? @sprintf("%.2e", row.nearest_expected) : string(row.nearest_expected)
        sigma_text = isfinite(row.singular_ratio) ? @sprintf("%.2e", row.singular_ratio) : string(row.singular_ratio)
        matched_text = @sprintf("%d/%d", row.matched, row.expected)
        @printf(
            "  %-18s %-10s %6.2f %7.3f %8s %8d %8d %8d %10s %10s %10s  %s %s\n",
            row.label,
            string(row.stage),
            row.rho,
            row.phase,
            matched_text,
            row.good,
            row.inside,
            row.spurious,
            max_text,
            near_text,
            sigma_text,
            row.success ? "ok" : "check",
            row.notes,
        )
    end
    print_loewner_layout_support("initial", initial_support)
    print_loewner_layout_support("updated", updated_support)
end

function run_loewner_interpolation_sweep(;
    radii=(1.15, 1.3, 1.6, 2.0),
    phase_fractions=(0.0, 0.125, 0.25, 0.375),
    loewner_points=6,
    residual_tol=1e-8,
    match_atol=1e-6,
    print_rows=true,
)
    # The hard exponential global chart is where interpolation placement matters most.
    ctx, chart, trial = exponential_hankel_sweep_trial(; component_scaling=:none)
    expected = ctx.expected
    rows = NamedTuple[]
    initial_entries = NamedTuple[]
    updated_entries = NamedTuple[]
    initial_best = nothing
    initial_best_score = (-1, -1, Inf)
    for rho in radii, fraction in phase_fractions
        phase = 2pi * fraction / loewner_points
        config = ReducedExtractorConfig(
            extractor=:loewner_counted,
            determinant_nodes=768,
            determinant_capacity=96,
            reduced_moments=16,
            reduced_nodes=768,
            reduced_ranktol=1e-10,
            loewner_points=loewner_points,
            loewner_radius=rho,
            loewner_phase=phase,
            residual_normalization=:vector,
        )
        try
            extraction = extract_reduced_nep(ctx, trial, chart, config)
            append!(initial_entries, loewner_layout_entries(extraction, rho, phase; residual_tol=residual_tol))
            row = loewner_sweep_row("exp_global", phase, rho, :initial, extraction, expected; residual_tol=residual_tol, match_atol=match_atol)
            push!(rows, row)
            score = (row.matched, -row.spurious, -row.max_residual)
            if score > initial_best_score
                initial_best = extraction
                initial_best_score = score
            end
        catch err
            push!(
                rows,
                (
                    label="exp_global",
                    phase=Float64(phase),
                    rho=Float64(rho),
                    stage=:initial,
                    expected=length(expected),
                    inside=0,
                    good=0,
                    matched=0,
                    spurious=0,
                    max_residual=Inf,
                    nearest_expected=Inf,
                    count_estimate=0,
                    singular_ratio=NaN,
                    success=false,
                    notes="$(typeof(err)): $err",
                ),
            )
        end
    end
    if initial_best !== nothing
        update_config = ResidualUpdateConfig(moment_count=1, rii_nodes=128, residual_ranktol=1e-10, compression_ranktol=1e-10)
        updated_trial, stats = residual_laurent_update(ctx, trial, initial_best, chart, update_config)
        for rho in radii, fraction in phase_fractions
            phase = 2pi * fraction / loewner_points
            config = ReducedExtractorConfig(
                extractor=:loewner_counted,
                determinant_nodes=768,
                determinant_capacity=96,
                reduced_moments=16,
                reduced_nodes=768,
                reduced_ranktol=1e-10,
                loewner_points=loewner_points,
                loewner_radius=rho,
                loewner_phase=phase,
                residual_normalization=:vector,
            )
            try
                extraction = extract_reduced_nep(ctx, updated_trial, chart, config)
                append!(updated_entries, loewner_layout_entries(extraction, rho, phase; residual_tol=residual_tol))
                push!(
                    rows,
                    loewner_sweep_row(
                        "exp_global",
                        phase,
                        rho,
                        :updated,
                        extraction,
                        expected;
                        residual_tol=residual_tol,
                        match_atol=match_atol,
                        notes=@sprintf("residual ranks=(%d,%d)", stats.right_residual_rank, stats.left_residual_rank),
                    ),
                )
            catch err
                push!(
                    rows,
                    (
                        label="exp_global",
                        phase=Float64(phase),
                        rho=Float64(rho),
                        stage=:updated,
                        expected=length(expected),
                        inside=0,
                        good=0,
                        matched=0,
                        spurious=0,
                        max_residual=Inf,
                        nearest_expected=Inf,
                        count_estimate=0,
                        singular_ratio=NaN,
                        success=false,
                        notes="$(typeof(err)): $err",
                    ),
                )
            end
        end
    end
    initial_support = loewner_layout_support_summary(initial_entries, expected; match_atol=match_atol)
    updated_support = loewner_layout_support_summary(updated_entries, expected; match_atol=match_atol)
    print_rows && print_loewner_interpolation_sweep(rows; initial_support=initial_support, updated_support=updated_support)
    (rows=rows, initial_support=initial_support, updated_support=updated_support)
end

function run_global_loewner_interior_artifact_diagnostic(;
    radii=(1.15, 1.3, 1.6),
    phase_fractions=(0.0, 0.25),
    loewner_points=6,
    residual_tol=1e-8,
    match_atol=1e-6,
    print_rows=true,
)
    result = run_loewner_interpolation_sweep(;
        radii=radii,
        phase_fractions=phase_fractions,
        loewner_points=loewner_points,
        residual_tol=residual_tol,
        match_atol=match_atol,
        print_rows=false,
    )
    bad_rows = [row for row in result.rows if row.good > 0 && row.spurious > 0]
    support1_initial = result.initial_support[1]
    support2_initial = result.initial_support[2]
    support1_updated = result.updated_support[1]
    support2_updated = result.updated_support[2]
    summary = (
        expected=support2_updated.matched,
        layouts=length(radii) * length(phase_fractions),
        bad_single_layouts=length(bad_rows),
        worst_nearest_expected=isempty(bad_rows) ? 0.0 : maximum(row.nearest_expected for row in bad_rows),
        worst_bad_residual=isempty(bad_rows) ? 0.0 : maximum(row.max_residual for row in bad_rows),
        worst_bad_singular_ratio=isempty(bad_rows) ? 0.0 : maximum(row.singular_ratio for row in bad_rows),
        initial_support1_spurious=support1_initial.spurious,
        initial_support2_spurious=support2_initial.spurious,
        updated_support1_spurious=support1_updated.spurious,
        updated_support2_spurious=support2_updated.spurious,
        updated_support2_matched=support2_updated.matched,
        updated_support2_count=support2_updated.count,
        support2_prunes_artifacts=support2_initial.spurious == 0 &&
            support2_updated.spurious == 0 &&
            support2_updated.matched == support2_updated.count,
    )
    if print_rows
        println()
        println("Global Loewner in-target artifact diagnostic")
        println("  one oversized chart; residual-small values are already inside the target contour")
        @printf(
            "  layouts=%d bad_single_layouts=%d worst_nearest=%.3e worst_residual=%.3e worst_sigma=%.3e\n",
            summary.layouts,
            summary.bad_single_layouts,
            summary.worst_nearest_expected,
            summary.worst_bad_residual,
            summary.worst_bad_singular_ratio,
        )
        @printf(
            "  support initial: s1 spurious=%d s2 spurious=%d; updated: s1 spurious=%d s2=%d/%d spurious=%d status=%s\n",
            summary.initial_support1_spurious,
            summary.initial_support2_spurious,
            summary.updated_support1_spurious,
            summary.updated_support2_matched,
            summary.updated_support2_count,
            summary.updated_support2_spurious,
            summary.support2_prunes_artifacts ? "ok" : "check",
        )
    end
    (result=result, bad_rows=bad_rows, summary=summary)
end

function run_exponential_local_chart_diagnostic(;
    extractors=(:ss_counted, :loewner_counted),
    radii=(1.2, 2.0),
    residual_tol=1e-8,
    match_atol=1e-6,
    print_rows=true,
)
    rows = NamedTuple[]
    for extractor in extractors
        result = run_dual_local_chart_sweep_analytic(;
            cases=exponential_many_root_cases(),
            outer_radius=10.0,
            radii=radii,
            iterations=1,
            basis_moments=4,
            basis_nodes=16,
            rii_nodes=128,
            basis_ranktol=1e-8,
            determinant_nodes=512,
            determinant_capacity=32,
            extractor=extractor,
            reduced_moments=8,
            reduced_nodes=512,
            residual_normalization=:vector,
            component_scaling=:none,
            residual_tol=residual_tol,
            match_atol=match_atol,
            print_charts=false,
        )
        push!(
            rows,
            (
                extractor=extractor,
                expected=length(result.expected),
                union=length(result.found),
                union_matched=result.matched,
                support2=length(result.support2_found),
                support2_matched=result.support2_matched,
                success=result.support2_matched == length(result.expected) && length(result.support2_found) == length(result.expected),
            ),
        )
    end
    if print_rows
        println()
        println("Exponential local-chart diagnostic")
        println("  supervised root-centered charts; support>=2 is the relevant merged result")
        @printf("  %-18s %8s %8s %8s %10s %10s\n", "extractor", "union", "matched", "support2", "support2_ok", "status")
        for row in rows
            @printf(
                "  %-18s %8d %4d/%-3d %8d %4d/%-5d %s\n",
                string(row.extractor),
                row.union,
                row.union_matched,
                row.expected,
                row.support2,
                row.support2_matched,
                row.expected,
                row.success ? "ok" : "check",
            )
        end
    end
    rows
end

function run_exponential_local_chart_loewner_layout_sweep(;
    loewner_radii=(1.15, 1.3, 1.6),
    phase_fractions=(0.0, 0.25),
    loewner_points=6,
    chart_radii=(1.2, 2.0),
    residual_tol=1e-8,
    match_atol=1e-6,
    print_rows=true,
)
    rows = NamedTuple[]
    for rho in loewner_radii, fraction in phase_fractions
        phase = 2pi * fraction / loewner_points
        result = run_dual_local_chart_sweep_analytic(;
            cases=exponential_many_root_cases(),
            outer_radius=10.0,
            radii=chart_radii,
            iterations=1,
            basis_moments=4,
            basis_nodes=16,
            rii_nodes=128,
            basis_ranktol=1e-8,
            determinant_nodes=512,
            determinant_capacity=32,
            extractor=:loewner_counted,
            reduced_moments=8,
            reduced_nodes=512,
            reduced_ranktol=1e-10,
            loewner_points=loewner_points,
            loewner_radius=rho,
            loewner_phase=phase,
            residual_normalization=:vector,
            component_scaling=:none,
            residual_tol=residual_tol,
            match_atol=match_atol,
            print_charts=false,
        )
        support_rows = support_sweep_counts(result; atol=match_atol)
        support1 = isempty(support_rows) ? (count=length(result.found), matched=result.matched) : support_rows[1]
        support2 = length(support_rows) >= 2 ? support_rows[2] : (count=length(result.support2_found), matched=result.support2_matched)
        push!(
            rows,
            (
                rho=Float64(rho),
                phase=Float64(phase),
                expected=length(result.expected),
                union=length(result.found),
                union_matched=result.matched,
                support1=support1.count,
                support1_matched=support1.matched,
                support2=support2.count,
                support2_matched=support2.matched,
                success=result.support2_matched == length(result.expected) && length(result.support2_found) == length(result.expected),
            ),
        )
    end
    if print_rows
        println()
        println("Exponential local-chart Loewner-layout sweep")
        println("  supervised root-centered charts; varies Loewner outside interpolation layout inside each local chart")
        @printf("  %6s %7s %8s %8s %10s %10s %8s\n", "rho", "phase", "union", "matched", "support2", "support2_ok", "status")
        for row in rows
            @printf(
                "  %6.2f %7.3f %8d %4d/%-3d %8d %4d/%-5d %s\n",
                row.rho,
                row.phase,
                row.union,
                row.union_matched,
                row.expected,
                row.support2,
                row.support2_matched,
                row.expected,
                row.success ? "ok" : "check",
            )
        end
    end
    rows
end

function run_exponential_grid_chart_loewner_spacing_sweep(;
    spacings=(2.4, 1.8, 1.2),
    loewner_radius=1.3,
    loewner_phase=0.0,
    loewner_points=6,
    chart_radii=(1.2, 2.0),
    residual_tol=1e-8,
    match_atol=1e-6,
    print_rows=true,
)
    rows = NamedTuple[]
    for spacing in spacings
        result = run_dual_grid_chart_cover_analytic(;
            cases=exponential_many_root_cases(),
            outer_radius=10.0,
            spacing=spacing,
            chart_radii=chart_radii,
            iterations=1,
            basis_moments=4,
            basis_nodes=16,
            rii_nodes=128,
            basis_ranktol=1e-8,
            determinant_nodes=512,
            determinant_capacity=32,
            extractor=:loewner_counted,
            reduced_moments=8,
            reduced_nodes=512,
            reduced_ranktol=1e-10,
            loewner_points=loewner_points,
            loewner_radius=loewner_radius,
            loewner_phase=loewner_phase,
            residual_normalization=:vector,
            component_scaling=:none,
            residual_tol=residual_tol,
            match_atol=match_atol,
        )
        support_rows = support_sweep_counts(result; atol=match_atol)
        support1 = isempty(support_rows) ? (count=length(result.found), matched=result.matched) : support_rows[1]
        support2 = length(support_rows) >= 2 ? support_rows[2] : (count=length(result.support2_found), matched=result.support2_matched)
        support3 = length(support_rows) >= 3 ? support_rows[3] : (count=0, matched=0)
        push!(
            rows,
            (
                spacing=Float64(spacing),
                centers=length(result.centers),
                expected=length(result.expected),
                union=length(result.found),
                union_matched=result.matched,
                support1=support1.count,
                support1_matched=support1.matched,
                support2=support2.count,
                support2_matched=support2.matched,
                support3=support3.count,
                support3_matched=support3.matched,
                success_union=result.matched == length(result.expected),
                success_support2=result.support2_matched == length(result.expected) && length(result.support2_found) == length(result.expected),
            ),
        )
    end
    if print_rows
        println()
        println("Exponential grid-chart Loewner spacing sweep")
        println("  unsupervised Cartesian grid centers; varies grid spacing while keeping local radii and Loewner layout fixed")
        @printf(
            "  %7s %8s %8s %8s %10s %10s %10s %8s\n",
            "spacing",
            "centers",
            "union",
            "matched",
            "support2",
            "support2_ok",
            "support3_ok",
            "status",
        )
        for row in rows
            @printf(
                "  %7.3f %8d %8d %4d/%-3d %8d %4d/%-5d %4d/%-5d %s\n",
                row.spacing,
                row.centers,
                row.union,
                row.union_matched,
                row.expected,
                row.support2,
                row.support2_matched,
                row.expected,
                row.support3_matched,
                row.expected,
                row.success_union ? (row.success_support2 ? "ok" : "union-only") : "check",
            )
        end
    end
    rows
end

function adaptive_grid_refinement_row(stage, result, center_count, added_count; match_atol)
    support_rows = support_sweep_counts(result; atol=match_atol)
    support2 = length(support_rows) >= 2 ? support_rows[2] : (count=length(result.support2_found), matched=result.support2_matched)
    support3 = length(support_rows) >= 3 ? support_rows[3] : (count=0, matched=0)
    (
        stage=stage,
        centers=center_count,
        added=added_count,
        expected=length(result.expected),
        union=length(result.found),
        union_matched=result.matched,
        support2=support2.count,
        support2_matched=support2.matched,
        support2_global=length(result.support2_global_found),
        support2_global_matched=result.support2_global_matched,
        support3=support3.count,
        support3_matched=support3.matched,
        success_support2=support2.matched == length(result.expected) && support2.count == length(result.expected),
        success_support2_global=result.support2_global_matched == length(result.expected) &&
            length(result.support2_global_found) == length(result.expected),
    )
end

function run_adaptive_grid_loewner_refinement(;
    label="Adaptive grid Loewner refinement",
    cases=(scalar_sine_case(), scalar_cosine_case(), scalar_shifted_sine_case(), scalar_expm1_case()),
    outer_center=0.0 + 0.0im,
    outer_radius=10.0,
    operator_builder=similarity_analytic_tools,
    operator_label="similarity",
    base_spacing=2.4,
    target_support=2,
    refinement_rounds=1,
    loewner_radius=1.3,
    loewner_phase=0.0,
    loewner_points=6,
    chart_radii=(1.2, 2.0),
    iterations=1,
    basis_moments=4,
    basis_nodes=16,
    update_moment_count=1,
    rii_nodes=128,
    basis_ranktol=1e-8,
    residual_ranktol=1e-10,
    compression_ranktol=1e-10,
    determinant_nodes=512,
    determinant_capacity=32,
    extractor=:loewner_counted,
    reduced_moments=8,
    reduced_nodes=512,
    reduced_ranktol=1e-10,
    reduced_refinement=:none,
    refinement_steps=4,
    residual_normalization=:vector,
    component_scaling=:none,
    residual_tol=1e-8,
    match_atol=1e-6,
    refine_inside_target_only=true,
    print_rows=true,
)
    centers = disk_grid_centers(outer_center, outer_radius, base_spacing)
    results = Any[]
    rows = NamedTuple[]
    added_by_round = Vector{ComplexF64}[]

    for round in 0:refinement_rounds
        result = run_dual_local_chart_sweep_analytic(;
            cases=cases,
            outer_center=outer_center,
            outer_radius=outer_radius,
            operator_builder=operator_builder,
            operator_label=operator_label,
            centers=centers,
            radii=chart_radii,
            iterations=iterations,
            basis_moments=basis_moments,
            basis_nodes=basis_nodes,
            rii_nodes=rii_nodes,
            update_moment_count=update_moment_count,
            basis_ranktol=basis_ranktol,
            residual_ranktol=residual_ranktol,
            compression_ranktol=compression_ranktol,
            determinant_nodes=determinant_nodes,
            determinant_capacity=determinant_capacity,
            extractor=extractor,
            reduced_moments=reduced_moments,
            reduced_nodes=reduced_nodes,
            reduced_ranktol=reduced_ranktol,
            reduced_refinement=reduced_refinement,
            refinement_steps=refinement_steps,
            loewner_points=loewner_points,
            loewner_radius=loewner_radius,
            loewner_phase=loewner_phase,
            residual_normalization=residual_normalization,
            component_scaling=component_scaling,
            residual_tol=residual_tol,
            match_atol=match_atol,
            selection=:residual,
            skip_empty_expected=false,
            print_charts=false,
        )
        push!(results, result)
        stage = round == 0 ? :base_grid : Symbol("candidate_refined_$round")
        added_count = round == 0 ? 0 : length(added_by_round[end])
        push!(rows, adaptive_grid_refinement_row(stage, result, length(centers), added_count; match_atol=match_atol))

        round == refinement_rounds && break
        weak_values = sorted_unique_values(
            ComplexF64[cluster.value for cluster in result.support_clusters if cluster.support < target_support];
            atol=match_atol,
        )
        if refine_inside_target_only
            weak_values = ComplexF64[
                value for value in weak_values
                if abs(value - outer_center) <= outer_radius + 10 * match_atol
            ]
        end
        added = ComplexF64[
            value for value in weak_values if all(abs(value - center) > match_atol for center in centers)
        ]
        new_centers = sorted_unique_values(vcat(ComplexF64.(centers), added); atol=match_atol)
        push!(added_by_round, ComplexF64.(added))
        centers = new_centers
        isempty(added) && break
    end

    if print_rows
        println()
        println(label)
        println("  starts from a coarse grid, then adds residual-small weak-support candidate values as new chart centers")
        @printf(
            "  %-20s %8s %8s %8s %8s %10s %10s %12s %8s\n",
            "stage",
            "centers",
            "added",
            "union",
            "matched",
            "support2",
            "support2_ok",
            "global2_ok",
            "status",
        )
        for row in rows
            @printf(
                "  %-20s %8d %8d %8d %4d/%-3d %8d %4d/%-5d %4d/%-7d %s\n",
                string(row.stage),
                row.centers,
                row.added,
                row.union,
                row.union_matched,
                row.expected,
                row.support2,
                row.support2_matched,
                row.expected,
                row.support2_global_matched,
                row.expected,
                row.success_support2_global ? "ok" : (row.success_support2 ? "supported-extra" : "check"),
            )
        end
    end

    (
        rows=rows,
        results=results,
        base=first(results),
        refined=last(results),
        added_centers=isempty(added_by_round) ? ComplexF64[] : reduce(vcat, added_by_round),
        added_by_round=added_by_round,
    )
end

function count_driven_refinement_row(stage, result, center_count, added_count, target_count, target_count_error; match_atol)
    support2_global = ComplexF64.(result.support2_global_found)
    validation_matched = match_expected_count(support2_global, result.expected; atol=match_atol)
    (
        stage=stage,
        centers=center_count,
        added=added_count,
        target_count=target_count,
        target_count_error=target_count_error,
        retained=length(support2_global),
        count_complete=length(support2_global) == target_count,
        expected=length(result.expected),
        validation_matched=validation_matched,
        validation_success=validation_matched == length(result.expected) &&
            length(support2_global) == length(result.expected),
    )
end

function count_driven_chart_diagnostic_summary(
    result;
    outer_center=0.0 + 0.0im,
    outer_radius,
    match_atol=1e-6,
    count_error_tol=1e-2,
)
    function stage_summary(stage_result)
        clusters = stage_result.support_clusters
        inside_clusters = [
            cluster for cluster in clusters
            if abs(cluster.value - outer_center) <= outer_radius + 10 * match_atol
        ]
        selected = stage_result.selected_records
        usable_selected = [record for record in selected if !record.failed && record.good > 0]
        count_errors = Float64[record.count_error for record in usable_selected]
        (
            union_good=length(stage_result.found),
            retained=length(stage_result.support2_global_found),
            inside_clusters=length(inside_clusters),
            weak_inside_clusters=count(cluster -> cluster.support < 2, inside_clusters),
            selected_records=length(selected),
            selected_good_records=length(usable_selected),
            selected_count_deficit=count(record -> record.good < record.count_estimate, usable_selected),
            selected_count_error_bad=count(record -> record.count_error > count_error_tol, usable_selected),
            max_selected_count_error=isempty(count_errors) ? 0.0 : maximum(count_errors),
        )
    end
    (
        base=stage_summary(result.base),
        final=stage_summary(result.refined),
        rows=result.rows,
        stop_reason=result.stop_reason,
        target_count=result.count.count_estimate,
        target_count_error=result.count.count_error,
        algebraic_retained_count=result.algebraic_retained_count,
    )
end

Base.@kwdef struct CountDrivenPolicyConfig
    base_spacing::Float64 = 2.4
    target_support::Int = 2
    max_refinement_rounds::Int = 4
    chart_radii::Tuple{Vararg{Float64}} = (1.2, 2.0)
    chart_radii_stages::Union{Nothing, Tuple{Vararg{Tuple{Vararg{Float64}}}}} = nothing
    residual_tol::Float64 = 1e-8
    match_atol::Float64 = 1e-6
    count_error_tol::Float64 = 1e-2
    refine_inside_target_only::Bool = true
end

function with_chart_radii(policy::CountDrivenPolicyConfig, chart_radii)
    CountDrivenPolicyConfig(;
        base_spacing=policy.base_spacing,
        target_support=policy.target_support,
        max_refinement_rounds=policy.max_refinement_rounds,
        chart_radii=Tuple(Float64.(chart_radii)),
        chart_radii_stages=policy.chart_radii_stages,
        residual_tol=policy.residual_tol,
        match_atol=policy.match_atol,
        count_error_tol=policy.count_error_tol,
        refine_inside_target_only=policy.refine_inside_target_only,
    )
end

Base.@kwdef struct CountDrivenNumericsConfig
    iterations::Int = 1
    basis_moments::Int = 4
    basis_nodes::Int = 16
    update_moment_count::Int = 1
    rii_nodes::Int = 128
    basis_ranktol::Float64 = 1e-8
    residual_ranktol::Float64 = 1e-10
    compression_ranktol::Float64 = 1e-10
    determinant_nodes::Int = 512
    determinant_capacity::Int = 32
    extractor::Symbol = :loewner_counted
    reduced_moments::Int = 8
    reduced_nodes::Int = 512
    reduced_ranktol::Float64 = 1e-10
    reduced_refinement::Symbol = :none
    refinement_steps::Int = 4
    loewner_radius::Float64 = 1.3
    loewner_phase::Float64 = 0.0
    loewner_points::Int = 6
    residual_normalization::Symbol = :vector
    component_scaling::Symbol = :none
    component_scaling_nodes::Int = 64
    update_mode::Symbol = :moment_compressed
    biorthogonalize::Bool = false
end

function reduced_extractor_config(numerics::CountDrivenNumericsConfig)
    ReducedExtractorConfig(;
        extractor=numerics.extractor,
        determinant_nodes=numerics.determinant_nodes,
        determinant_capacity=numerics.determinant_capacity,
        reduced_moments=numerics.reduced_moments,
        reduced_nodes=numerics.reduced_nodes,
        reduced_ranktol=numerics.reduced_ranktol,
        loewner_points=numerics.loewner_points,
        loewner_radius=numerics.loewner_radius,
        loewner_phase=numerics.loewner_phase,
        residual_normalization=numerics.residual_normalization,
        refinement=numerics.reduced_refinement,
        refinement_steps=numerics.refinement_steps,
    )
end

function residual_update_config(numerics::CountDrivenNumericsConfig)
    ResidualUpdateConfig(;
        moment_count=numerics.update_moment_count,
        rii_nodes=numerics.rii_nodes,
        residual_ranktol=numerics.residual_ranktol,
        compression_ranktol=numerics.compression_ranktol,
        mode=numerics.update_mode,
        biorthogonalize=numerics.biorthogonalize,
    )
end

function local_cluster_multiplicity_estimates(
    cases,
    values;
    outer_center=0.0 + 0.0im,
    outer_radius,
    operator_builder=similarity_analytic_tools,
    component_scaling=:none,
    component_scaling_nodes=64,
    local_radius=0.08,
    determinant_nodes=2048,
    determinant_capacity=16,
)
    isempty(values) && return NamedTuple[]
    chart = ContourChart(
        outer_center,
        outer_radius;
        component_scaling=component_scaling,
        component_scaling_nodes=component_scaling_nodes,
    )
    ctx = analytic_context(cases, chart, operator_builder)
    retained = ComplexF64.(values)
    rows = NamedTuple[]
    for (index, value) in pairs(retained)
        nearest = length(retained) == 1 ? Inf :
            minimum(abs(value - retained[j]) for j in eachindex(retained) if j != index)
        boundary_room = outer_radius - abs(value - outer_center)
        radius = min(local_radius, 0.35 * nearest)
        boundary_room > 0 && (radius = min(radius, 0.45 * boundary_room))
        radius = max(radius, 100 * eps(Float64) * max(1.0, abs(value)))
        _, count_estimate, sums = determinant_power_sums(
            ctx.Tmatrix,
            ctx.Tderivative;
            center=value,
            radius=radius,
            nodes=determinant_nodes,
            capacity=determinant_capacity,
        )
        push!(
            rows,
            (
                value=value,
                radius=radius,
                multiplicity=count_estimate,
                count_error=abs(sums[1] - count_estimate),
            ),
        )
    end
    rows
end

function run_count_driven_adaptive_grid_refinement(;
    label="Count-driven adaptive grid refinement",
    cases=(scalar_sine_case(), scalar_cosine_case(), scalar_shifted_sine_case(), scalar_expm1_case()),
    outer_center=0.0 + 0.0im,
    outer_radius=10.0,
    operator_builder=similarity_analytic_tools,
    operator_label="similarity",
    policy=nothing,
    numerics=nothing,
    base_spacing=2.4,
    target_support=2,
    max_refinement_rounds=4,
    loewner_radius=1.3,
    loewner_phase=0.0,
    loewner_points=6,
    chart_radii=(1.2, 2.0),
    iterations=1,
    basis_moments=4,
    basis_nodes=16,
    update_moment_count=1,
    rii_nodes=128,
    basis_ranktol=1e-8,
    residual_ranktol=1e-10,
    compression_ranktol=1e-10,
    determinant_nodes=512,
    determinant_capacity=32,
    extractor=:loewner_counted,
    reduced_moments=8,
    reduced_nodes=512,
    reduced_ranktol=1e-10,
    reduced_refinement=:none,
    refinement_steps=4,
    residual_normalization=:vector,
    component_scaling=:none,
    component_scaling_nodes=64,
    update_mode=:moment_compressed,
    biorthogonalize=false,
    residual_tol=1e-8,
    match_atol=1e-6,
    count_error_tol=1e-2,
    refine_inside_target_only=true,
    print_rows=true,
)
    if policy !== nothing
        base_spacing = policy.base_spacing
        target_support = policy.target_support
        max_refinement_rounds = policy.max_refinement_rounds
        chart_radii = policy.chart_radii
        residual_tol = policy.residual_tol
        match_atol = policy.match_atol
        count_error_tol = policy.count_error_tol
        refine_inside_target_only = policy.refine_inside_target_only
    end
    if numerics !== nothing
        iterations = numerics.iterations
        basis_moments = numerics.basis_moments
        basis_nodes = numerics.basis_nodes
        update_moment_count = numerics.update_moment_count
        rii_nodes = numerics.rii_nodes
        basis_ranktol = numerics.basis_ranktol
        residual_ranktol = numerics.residual_ranktol
        compression_ranktol = numerics.compression_ranktol
        determinant_nodes = numerics.determinant_nodes
        determinant_capacity = numerics.determinant_capacity
        extractor = numerics.extractor
        reduced_moments = numerics.reduced_moments
        reduced_nodes = numerics.reduced_nodes
        reduced_ranktol = numerics.reduced_ranktol
        reduced_refinement = numerics.reduced_refinement
        refinement_steps = numerics.refinement_steps
        loewner_radius = numerics.loewner_radius
        loewner_phase = numerics.loewner_phase
        loewner_points = numerics.loewner_points
        residual_normalization = numerics.residual_normalization
        component_scaling = numerics.component_scaling
        component_scaling_nodes = numerics.component_scaling_nodes
        update_mode = numerics.update_mode
        biorthogonalize = numerics.biorthogonalize
    end
    extraction_config = numerics === nothing ? nothing : reduced_extractor_config(numerics)
    update_config = numerics === nothing ? nothing : residual_update_config(numerics)
    count = full_operator_count_estimate(
        cases;
        outer_center=outer_center,
        outer_radius=outer_radius,
        operator_builder=operator_builder,
        component_scaling=component_scaling,
        component_scaling_nodes=component_scaling_nodes,
        determinant_nodes=max(2048, determinant_nodes),
        determinant_capacity=max(128, determinant_capacity),
    )
    centers = disk_grid_centers(outer_center, outer_radius, base_spacing)
    results = Any[]
    rows = NamedTuple[]
    added_by_round = Vector{ComplexF64}[]
    stop_reason = :max_rounds
    multiplicity_rows = NamedTuple[]
    algebraic_retained_count = 0

    for round in 0:max_refinement_rounds
        result = run_dual_local_chart_sweep_analytic(;
            cases=cases,
            outer_center=outer_center,
            outer_radius=outer_radius,
            operator_builder=operator_builder,
            operator_label=operator_label,
            centers=centers,
            radii=chart_radii,
            iterations=iterations,
            basis_moments=basis_moments,
            basis_nodes=basis_nodes,
            rii_nodes=rii_nodes,
            update_moment_count=update_moment_count,
            basis_ranktol=basis_ranktol,
            residual_ranktol=residual_ranktol,
            compression_ranktol=compression_ranktol,
            determinant_nodes=determinant_nodes,
            determinant_capacity=determinant_capacity,
            extractor=extractor,
            reduced_moments=reduced_moments,
            reduced_nodes=reduced_nodes,
            reduced_ranktol=reduced_ranktol,
            reduced_refinement=reduced_refinement,
            refinement_steps=refinement_steps,
            loewner_points=loewner_points,
            loewner_radius=loewner_radius,
            loewner_phase=loewner_phase,
            residual_normalization=residual_normalization,
            component_scaling=component_scaling,
            component_scaling_nodes=component_scaling_nodes,
            biorthogonalize=biorthogonalize,
            update_mode=update_mode,
            extraction_config=extraction_config,
            update_config=update_config,
            residual_tol=residual_tol,
            match_atol=match_atol,
            selection=:residual,
            skip_empty_expected=false,
            print_charts=false,
        )
        push!(results, result)
        stage = round == 0 ? :base_grid : Symbol("count_refined_$round")
        added_count = round == 0 ? 0 : length(added_by_round[end])
        row = count_driven_refinement_row(
            stage,
            result,
            length(centers),
            added_count,
            count.count_estimate,
            count.count_error;
            match_atol=match_atol,
        )
        push!(rows, row)

        if count.count_error > count_error_tol
            stop_reason = :target_count_unreliable
            break
        elseif row.count_complete
            # Simple-root lower rung: unique support already satisfies the
            # algebraic target count, so no local multiplicity probes are
            # needed.
            multiplicity_rows = NamedTuple[]
            algebraic_retained_count = row.retained
            stop_reason = :target_count_complete
            break
        else
            multiplicity_rows = local_cluster_multiplicity_estimates(
                cases,
                result.support2_global_found;
                outer_center=outer_center,
                outer_radius=outer_radius,
                operator_builder=operator_builder,
                component_scaling=component_scaling,
                component_scaling_nodes=component_scaling_nodes,
                determinant_nodes=max(2048, determinant_nodes),
                determinant_capacity=max(16, determinant_capacity),
            )
            algebraic_retained_count = sum(item.multiplicity for item in multiplicity_rows; init=0)
            if algebraic_retained_count == count.count_estimate &&
                    all(item.count_error <= count_error_tol for item in multiplicity_rows)
                stop_reason = :target_algebraic_count_complete
                break
            end
        end
        if round == max_refinement_rounds
            stop_reason = :max_rounds
            break
        end

        weak_values = sorted_unique_values(
            ComplexF64[cluster.value for cluster in result.support_clusters if cluster.support < target_support];
            atol=match_atol,
        )
        if refine_inside_target_only
            weak_values = ComplexF64[
                value for value in weak_values
                if abs(value - outer_center) <= outer_radius + 10 * match_atol
            ]
        end
        added = ComplexF64[
            value for value in weak_values if all(abs(value - center) > match_atol for center in centers)
        ]
        if isempty(added)
            push!(added_by_round, ComplexF64[])
            stop_reason = if length(result.support2_global_found) < count.count_estimate &&
                    algebraic_retained_count == count.count_estimate &&
                    all(item.count_error <= count_error_tol for item in multiplicity_rows)
                :target_algebraic_count_complete
            elseif length(result.support2_global_found) < count.count_estimate
                :count_multiplicity_or_unresolved_defect
            else
                :no_new_weak_target_centers
            end
            break
        end
        push!(added_by_round, ComplexF64.(added))
        centers = sorted_unique_values(vcat(ComplexF64.(centers), added); atol=match_atol)
    end

    if print_rows
        println()
        println(label)
        println("  stops from the full-operator argument-principle count, not from known roots")
        @printf(
            "  %-18s %8s %8s %12s %12s %12s %12s %8s\n",
            "stage",
            "centers",
            "added",
            "retained",
            "target",
            "count_ok",
            "valid_ok",
            "status",
        )
        for row in rows
            @printf(
                "  %-18s %8d %8d %8d %4d/%-7d %11s %4d/%-7d %s\n",
                string(row.stage),
                row.centers,
                row.added,
                row.retained,
                row.retained,
                row.target_count,
                string(row.count_complete),
                row.validation_matched,
                row.expected,
                row.validation_success ? "ok" : "check",
            )
        end
        @printf(
            "  stop=%s target_count_error=%.3e algebraic_retained=%d\n",
            string(stop_reason),
            count.count_error,
            algebraic_retained_count,
        )
    end

    (
        rows=rows,
        results=results,
        base=first(results),
        refined=last(results),
        count=count,
        stop_reason=stop_reason,
        multiplicities=multiplicity_rows,
        algebraic_retained_count=algebraic_retained_count,
        added_centers=isempty(added_by_round) ? ComplexF64[] : reduce(vcat, added_by_round),
        added_by_round=added_by_round,
    )
end

function run_count_driven_radius_ladder(;
    chart_radii_stages,
    runner,
    print_rows=true,
)
    stage_results = Any[]
    for (index, radii) in pairs(chart_radii_stages)
        result = runner(radii)
        push!(stage_results, (stage=index, chart_radii=Tuple(radii), result=result))
        if print_rows
            final = result.rows[end]
            @printf(
                "  radius_stage=%d radii=%s stop=%s retained=%d/%d rounds=%d\n",
                index,
                string(collect(radii)),
                string(result.stop_reason),
                final.retained,
                final.target_count,
                length(result.rows),
            )
        end
        result.stop_reason in (:target_count_complete, :target_algebraic_count_complete, :target_count_unreliable) && break
    end
    final = stage_results[end].result
    (
        stages=stage_results,
        result=final,
        count=final.count,
        stop_reason=final.stop_reason,
        algebraic_retained_count=final.algebraic_retained_count,
        rows=final.rows,
    )
end

function run_count_driven_policy_ladder(;
    policy::CountDrivenPolicyConfig,
    numerics=nothing,
    print_rows=true,
    kwargs...,
)
    stages = policy.chart_radii_stages
    stages === nothing && error("policy chart_radii_stages must be set for radius-ladder execution")
    run_count_driven_radius_ladder(;
        chart_radii_stages=stages,
        print_rows=print_rows,
        runner=radii -> run_count_driven_adaptive_grid_refinement(;
            policy=with_chart_radii(policy, radii),
            numerics=numerics,
            print_rows=false,
            kwargs...,
        ),
    )
end

function run_count_driven_policy_diagnostic(;
    outer_center=0.0 + 0.0im,
    outer_radius,
    policy=CountDrivenPolicyConfig(),
    numerics=nothing,
    print_rows=true,
    diagnostic_label="policy diagnostic",
    kwargs...,
)
    ladder = policy.chart_radii_stages === nothing ? nothing : run_count_driven_policy_ladder(;
        outer_center=outer_center,
        outer_radius=outer_radius,
        policy=policy,
        numerics=numerics,
        print_rows=print_rows,
        kwargs...,
    )
    result = ladder === nothing ? run_count_driven_adaptive_grid_refinement(;
            outer_center=outer_center,
            outer_radius=outer_radius,
            policy=policy,
            numerics=numerics,
            print_rows=print_rows,
            kwargs...,
        ) : ladder.result
    diagnostic = count_driven_chart_diagnostic_summary(
        result;
        outer_center=outer_center,
        outer_radius=outer_radius,
        match_atol=policy.match_atol,
        count_error_tol=policy.count_error_tol,
    )
    if print_rows
        println("  $diagnostic_label:")
        @printf(
            "    base union=%d retained=%d weak=%d count_warnings=%d max_count_error=%.3e target=%d\n",
            diagnostic.base.union_good,
            diagnostic.base.retained,
            diagnostic.base.weak_inside_clusters,
            diagnostic.base.selected_count_error_bad,
            diagnostic.base.max_selected_count_error,
            result.count.count_estimate,
        )
        @printf(
            "    final union=%d retained=%d weak=%d count_warnings=%d\n",
            diagnostic.final.union_good,
            diagnostic.final.retained,
            diagnostic.final.weak_inside_clusters,
            diagnostic.final.selected_count_error_bad,
        )
    end
    (
        result=result,
        diagnostic=diagnostic,
        rows=result.rows,
        count=result.count,
        stop_reason=result.stop_reason,
        algebraic_retained_count=result.algebraic_retained_count,
        added_centers=result.added_centers,
        stages=ladder === nothing ? nothing : ladder.stages,
    )
end

function run_canonical_nlfeast_limit_diagnostic(;
    center=0.0 + 0.0im,
    radius=1.2,
    residual_tol=1e-8,
    match_atol=1e-6,
    print_rows=true,
)
    cases = (
        scalar_rational_case(; root=-0.5, pole=3.0, name="canonical_r1"),
        scalar_rational_case(; root=0.2 + 0.3im, pole=3.0 + 0.5im, name="canonical_r2"),
        scalar_rational_case(; root=0.6 - 0.2im, pole=3.2 - 0.4im, name="canonical_r3"),
    )
    common_kwargs = (
        cases=cases,
        center=center,
        radius=radius,
        operator_builder=similarity_analytic_tools,
        operator_label="canonical similarity rational",
        basis_moments=1,
        basis_nodes=16,
        rii_nodes=128,
        update_moment_count=1,
        iterations=2,
        extractor=:loewner_counted,
        reduced_moments=6,
        reduced_nodes=256,
        determinant_capacity=16,
        basis_ranktol=1e-10,
        residual_tol=residual_tol,
        match_atol=match_atol,
        verbose=false,
    )
    scalar = run_dual_moment_compressed_rii_analytic_iteration(; common_kwargs..., update_mode=:scalar_expanded)
    compressed = run_dual_moment_compressed_rii_analytic_iteration(; common_kwargs..., update_mode=:moment_compressed)

    chart = ContourChart(center, radius)
    ctx = analytic_context(cases, chart, similarity_analytic_tools)
    Random.seed!(1234)
    X = rand(ComplexF64, ctx.n, length(ctx.expected))
    values, _, residuals = FEASTSolver.nlfeast!(
        ctx.Tmatrix,
        X,
        16,
        2;
        c=center,
        r=radius,
        ϵ=residual_tol,
        store=true,
    )
    inside = FEASTSolver.in_contour(values, center, radius)
    canonical_values = ComplexF64.(values[inside])
    canonical_residuals = Float64.(residuals[inside])
    canonical_matched = match_expected_count(canonical_values, ctx.expected; atol=match_atol)
    canonical_summary = (
        inside=count(inside),
        good=count(canonical_residuals .<= residual_tol),
        matched=canonical_matched,
        spurious_good=max(count(canonical_residuals .<= residual_tol) - canonical_matched, 0),
        max_residual=isempty(canonical_residuals) ? Inf : maximum(canonical_residuals),
    )

    scalar_summary = last(scalar.summaries)
    compressed_summary = last(compressed.summaries)
    if print_rows
        println()
        println("Canonical NLFEAST limit diagnostic")
        println("  one root per component; compares existing nlfeast!, scalar expanded RII, and compressed residual-Laurent update")
        for row in (
            (method=:nlfeast, summary=canonical_summary),
            (method=:scalar_expanded, summary=scalar_summary),
            (method=:moment_compressed, summary=compressed_summary),
        )
            @printf(
                "  %-18s good=%d matched=%d/%d spurious=%d max=%.3e\n",
                string(row.method),
                row.summary.good,
                row.summary.matched,
                length(ctx.expected),
                row.summary.spurious_good,
                row.summary.max_residual,
            )
        end
    end
    (
        expected=ctx.expected,
        canonical_values=canonical_values,
        canonical_summary=canonical_summary,
        scalar=scalar,
        scalar_summary=scalar_summary,
        compressed=compressed,
        compressed_summary=compressed_summary,
    )
end

function run_three_function_count_driven_adaptive_refinement(;
    outer_radius=20.0,
    base_spacing=3.0,
    target_support=2,
    max_refinement_rounds=4,
    loewner_radius=1.3,
    loewner_phase=0.0,
    loewner_points=6,
    chart_radii=(1.5, 2.4),
    residual_tol=1e-8,
    match_atol=1e-6,
    print_rows=true,
)
    run_count_driven_adaptive_grid_refinement(;
        label="Three-function count-driven adaptive refinement",
        cases=(scalar_sine_case(), scalar_cosine_case(), scalar_shifted_sine_case()),
        outer_radius=outer_radius,
        base_spacing=base_spacing,
        target_support=target_support,
        max_refinement_rounds=max_refinement_rounds,
        loewner_radius=loewner_radius,
        loewner_phase=loewner_phase,
        loewner_points=loewner_points,
        chart_radii=chart_radii,
        residual_tol=residual_tol,
        match_atol=match_atol,
        print_rows=print_rows,
    )
end

function run_triangular_count_driven_adaptive_refinement(;
    coupling=10.0,
    outer_radius=6.0,
    base_spacing=2.4,
    target_support=2,
    max_refinement_rounds=4,
    chart_radii=(1.2, 1.8),
    residual_tol=1e-8,
    match_atol=1e-6,
    print_rows=true,
)
    run_count_driven_adaptive_grid_refinement(;
        label="Triangular count-driven adaptive refinement",
        cases=(scalar_sine_case(), scalar_cosine_case(), scalar_shifted_sine_case(), scalar_expm1_case()),
        outer_radius=outer_radius,
        operator_builder=triangular_operator_builder(; coupling=coupling),
        operator_label="triangular(coupling=$coupling)",
        base_spacing=base_spacing,
        target_support=target_support,
        max_refinement_rounds=max_refinement_rounds,
        chart_radii=chart_radii,
        iterations=2,
        basis_nodes=24,
        rii_nodes=128,
        determinant_nodes=256,
        determinant_capacity=96,
        reduced_nodes=256,
        residual_normalization=:operator,
        component_scaling=:contour_max,
        residual_tol=residual_tol,
        match_atol=match_atol,
        print_rows=print_rows,
    )
end

function run_squared_sine_count_driven_adaptive_refinement(;
    outer_radius=10.0,
    base_spacing=2.4,
    target_support=2,
    max_refinement_rounds=4,
    chart_radii=(1.5, 2.4),
    residual_tol=1e-8,
    match_atol=1e-6,
    print_rows=true,
)
    run_count_driven_adaptive_grid_refinement(;
        label="Squared-sine count-driven adaptive refinement",
        cases=(scalar_squared_sine_case(),),
        outer_radius=outer_radius,
        base_spacing=base_spacing,
        target_support=target_support,
        max_refinement_rounds=max_refinement_rounds,
        chart_radii=chart_radii,
        basis_moments=8,
        basis_nodes=64,
        determinant_nodes=512,
        determinant_capacity=64,
        reduced_moments=16,
        reduced_nodes=512,
        component_scaling=:contour_max,
        residual_tol=residual_tol,
        match_atol=match_atol,
        print_rows=print_rows,
    )
end

function run_delay_count_driven_adaptive_refinement(;
    outer_radius=6.0,
    base_spacing=1.8,
    target_support=2,
    max_refinement_rounds=4,
    chart_radii=(1.2, 2.0),
    residual_tol=1e-8,
    match_atol=1e-6,
    print_rows=true,
)
    run_count_driven_adaptive_grid_refinement(;
        label="Delay count-driven adaptive refinement",
        cases=(scalar_delay_case(),),
        outer_radius=outer_radius,
        base_spacing=base_spacing,
        target_support=target_support,
        max_refinement_rounds=max_refinement_rounds,
        chart_radii=chart_radii,
        basis_moments=6,
        basis_nodes=48,
        determinant_nodes=512,
        determinant_capacity=64,
        reduced_moments=12,
        reduced_nodes=512,
        component_scaling=:none,
        residual_tol=residual_tol,
        match_atol=match_atol,
        print_rows=print_rows,
    )
end

function run_multi_delay_count_driven_adaptive_refinement(;
    coupling=5.0,
    outer_radius=6.0,
    base_spacing=1.8,
    target_support=2,
    max_refinement_rounds=4,
    chart_radii=(1.2, 2.0),
    residual_tol=1e-8,
    match_atol=1e-6,
    print_rows=true,
)
    run_count_driven_adaptive_grid_refinement(;
        label="Multi-delay count-driven adaptive refinement",
        cases=(
            scalar_delay_case(; a=0.4, b=2.0, tau=1.0),
            scalar_delay_case(; a=-0.2, b=1.4, tau=0.8),
            scalar_delay_case(; a=0.9, b=2.5, tau=1.2),
        ),
        outer_radius=outer_radius,
        operator_builder=triangular_operator_builder(; coupling=coupling),
        operator_label="triangular multi delay(coupling=$coupling)",
        base_spacing=base_spacing,
        target_support=target_support,
        max_refinement_rounds=max_refinement_rounds,
        chart_radii=chart_radii,
        iterations=2,
        basis_moments=6,
        basis_nodes=48,
        determinant_nodes=512,
        determinant_capacity=128,
        reduced_moments=12,
        reduced_nodes=512,
        residual_normalization=:operator,
        component_scaling=:contour_max,
        residual_tol=residual_tol,
        match_atol=match_atol,
        print_rows=print_rows,
    )
end

function run_two_delay_count_driven_adaptive_refinement(;
    outer_radius=6.0,
    base_spacing=1.8,
    target_support=2,
    max_refinement_rounds=4,
    chart_radii=(1.2, 2.0),
    residual_tol=1e-8,
    match_atol=1e-6,
    print_rows=true,
)
    run_count_driven_adaptive_grid_refinement(;
        label="Two-delay count-driven adaptive refinement",
        cases=(scalar_two_delay_case(),),
        outer_radius=outer_radius,
        base_spacing=base_spacing,
        target_support=target_support,
        max_refinement_rounds=max_refinement_rounds,
        chart_radii=chart_radii,
        basis_moments=8,
        basis_nodes=64,
        determinant_nodes=768,
        determinant_capacity=96,
        reduced_moments=16,
        reduced_nodes=768,
        component_scaling=:none,
        residual_tol=residual_tol,
        match_atol=match_atol,
        print_rows=print_rows,
    )
end

function coupled_two_delay_cases()
    (
        scalar_two_delay_case(; a=0.15, b=1.6, tau=0.7, c=0.85, sigma=1.4),
        scalar_two_delay_case(; a=-0.25, b=1.25, tau=0.9, c=0.55, sigma=1.7),
    )
end

function coupled_two_delay_operator_builder(; coupling=0.15)
    function builder(cases; component_scales=nothing)
        length(cases) == 2 || error("coupled two-delay control expects exactly two cases")
        scales = component_scales === nothing ? ones(Float64, 2) : Float64.(component_scales)
        function row_scaled(M)
            Diagonal(ComplexF64.(1 ./ scales)) * M
        end
        function Tmatrix(z)
            row_scaled(ComplexF64[
                cases[1].f(z) coupling * (1 + 0.2z)
                coupling * exp(-0.3z) cases[2].f(z)
            ])
        end
        function Tderivative(z)
            row_scaled(ComplexF64[
                cases[1].df(z) 0.2 * coupling
                -0.3 * coupling * exp(-0.3z) cases[2].df(z)
            ])
        end
        Tsolve(z, B) = Tmatrix(z) \ B
        Tadjoint_solve(z, B) = Tmatrix(z)' \ B
        expected_roots(center, radius) = ComplexF64[]
        Tmatrix, Tderivative, Tsolve, Tadjoint_solve, expected_roots
    end
end

function dense_multi_delay_cases()
    (
        scalar_delay_case(; a=0.4, b=2.0, tau=1.0),
        scalar_delay_case(; a=-0.2, b=1.4, tau=0.8),
        scalar_two_delay_case(; a=0.1, b=1.2, tau=0.9, c=0.6, sigma=1.5),
    )
end

function dense_multi_delay_operator_builder(; coupling=0.5)
    function builder(cases; component_scales=nothing)
        length(cases) == 3 || error("dense multi-delay control expects exactly three cases")
        scales = component_scales === nothing ? ones(Float64, 3) : Float64.(component_scales)
        function row_scaled(M)
            Diagonal(ComplexF64.(1 ./ scales)) * M
        end
        function C(z)
            ComplexF64[
                0 1 + 0.1z exp(-0.2z)
                0.7 * exp(-0.15z) 0 0.8 - 0.05z
                0.4 + 0.08z 0.6 * exp(-0.25z) 0
            ]
        end
        function Cderivative(z)
            ComplexF64[
                0 0.1 -0.2 * exp(-0.2z)
                -0.105 * exp(-0.15z) 0 -0.05
                0.08 -0.15 * exp(-0.25z) 0
            ]
        end
        function Tmatrix(z)
            row_scaled(Diagonal(ComplexF64[cases[i].f(z) for i in eachindex(cases)]) + coupling * C(z))
        end
        function Tderivative(z)
            row_scaled(Diagonal(ComplexF64[cases[i].df(z) for i in eachindex(cases)]) + coupling * Cderivative(z))
        end
        Tsolve(z, B) = Tmatrix(z) \ B
        Tadjoint_solve(z, B) = adjoint(Tmatrix(z)) \ B
        expected_roots(center, radius) = ComplexF64[]
        Tmatrix, Tderivative, Tsolve, Tadjoint_solve, expected_roots
    end
end

function run_coupled_two_delay_count_driven_adaptive_refinement(;
    coupling=0.15,
    outer_radius=6.0,
    base_spacing=1.8,
    target_support=2,
    max_refinement_rounds=4,
    chart_radii=(1.2, 2.0),
    residual_tol=1e-8,
    match_atol=1e-6,
    print_rows=true,
)
    run_count_driven_adaptive_grid_refinement(;
        label="Coupled two-delay count-driven adaptive refinement",
        cases=coupled_two_delay_cases(),
        outer_radius=outer_radius,
        operator_builder=coupled_two_delay_operator_builder(; coupling=coupling),
        operator_label="coupled two-delay(coupling=$coupling)",
        base_spacing=base_spacing,
        target_support=target_support,
        max_refinement_rounds=max_refinement_rounds,
        chart_radii=chart_radii,
        iterations=2,
        basis_moments=8,
        basis_nodes=64,
        rii_nodes=128,
        determinant_nodes=768,
        determinant_capacity=128,
        reduced_moments=16,
        reduced_nodes=768,
        residual_normalization=:operator,
        component_scaling=:none,
        residual_tol=residual_tol,
        match_atol=match_atol,
        print_rows=print_rows,
    )
end

function run_coupled_two_delay_radius_ladder_refinement(;
    coupling=1.0,
    outer_radius=12.0,
    base_spacing=4.0,
    chart_radii_stages=((1.2, 2.0), (1.2, 3.0)),
    max_refinement_rounds=4,
    residual_tol=1e-8,
    match_atol=1e-6,
    print_rows=true,
)
    policy = CountDrivenPolicyConfig(;
        base_spacing=base_spacing,
        max_refinement_rounds=max_refinement_rounds,
        chart_radii=first(chart_radii_stages),
        chart_radii_stages=Tuple(Tuple(Float64.(radii)) for radii in chart_radii_stages),
        residual_tol=residual_tol,
        match_atol=match_atol,
    )
    numerics = CountDrivenNumericsConfig(;
        iterations=2,
        basis_moments=8,
        basis_nodes=64,
        rii_nodes=128,
        determinant_nodes=768,
        determinant_capacity=128,
        reduced_moments=16,
        reduced_nodes=768,
        residual_normalization=:operator,
        component_scaling=:none,
    )
    run_count_driven_policy_diagnostic(;
        label="Coupled two-delay radius-ladder refinement",
        cases=coupled_two_delay_cases(),
        outer_radius=outer_radius,
        operator_builder=coupled_two_delay_operator_builder(; coupling=coupling),
        operator_label="coupled two-delay(coupling=$coupling)",
        policy=policy,
        numerics=numerics,
        print_rows=print_rows,
        diagnostic_label="radius-ladder diagnostic",
    )
end

function run_coupled_two_delay_mixed_policy_stress(;
    coupling=1.0,
    outer_radius=16.0,
    base_spacing=4.0,
    chart_radii=(1.5, 3.0),
    max_refinement_rounds=4,
    residual_tol=1e-8,
    match_atol=1e-6,
    print_rows=true,
)
    policy = CountDrivenPolicyConfig(;
        base_spacing=base_spacing,
        chart_radii=chart_radii,
        max_refinement_rounds=max_refinement_rounds,
        residual_tol=residual_tol,
        match_atol=match_atol,
    )
    numerics = CountDrivenNumericsConfig(;
        iterations=2,
        basis_moments=8,
        basis_nodes=64,
        rii_nodes=128,
        determinant_nodes=768,
        determinant_capacity=160,
        reduced_moments=16,
        reduced_nodes=768,
        residual_normalization=:operator,
        component_scaling=:none,
    )
    run_count_driven_policy_diagnostic(;
        label="Coupled two-delay mixed policy stress",
        cases=coupled_two_delay_cases(),
        outer_radius=outer_radius,
        operator_builder=coupled_two_delay_operator_builder(; coupling=coupling),
        operator_label="coupled two-delay(coupling=$coupling)",
        policy=policy,
        numerics=numerics,
        print_rows=print_rows,
        diagnostic_label="mixed diagnostic",
    )
end

function run_dense_multi_delay_weak_support_stress(;
    coupling=0.5,
    outer_radius=6.0,
    base_spacing=3.0,
    chart_radii=(1.2, 2.0),
    max_refinement_rounds=4,
    residual_tol=1e-8,
    match_atol=1e-6,
    print_rows=true,
)
    policy = CountDrivenPolicyConfig(;
        base_spacing=base_spacing,
        chart_radii=chart_radii,
        max_refinement_rounds=max_refinement_rounds,
        residual_tol=residual_tol,
        match_atol=match_atol,
    )
    numerics = CountDrivenNumericsConfig(;
        iterations=2,
        basis_moments=8,
        basis_nodes=64,
        rii_nodes=128,
        determinant_nodes=768,
        determinant_capacity=128,
        reduced_moments=16,
        reduced_nodes=768,
        residual_normalization=:operator,
        component_scaling=:none,
    )
    run_count_driven_policy_diagnostic(;
        label="Dense multi-delay weak-support stress",
        cases=dense_multi_delay_cases(),
        outer_radius=outer_radius,
        operator_builder=dense_multi_delay_operator_builder(; coupling=coupling),
        operator_label="dense multi-delay(coupling=$coupling)",
        policy=policy,
        numerics=numerics,
        print_rows=print_rows,
        diagnostic_label="dense diagnostic",
    )
end

function run_duplicate_delay_count_driven_adaptive_refinement(;
    coupling=5.0,
    outer_radius=6.0,
    base_spacing=1.8,
    target_support=2,
    max_refinement_rounds=4,
    chart_radii=(1.2, 2.0),
    residual_tol=1e-8,
    match_atol=1e-6,
    print_rows=true,
)
    delay = scalar_delay_case(; a=0.4, b=2.0, tau=1.0)
    run_count_driven_adaptive_grid_refinement(;
        label="Duplicate-delay count-driven adaptive refinement",
        cases=(delay, delay),
        outer_radius=outer_radius,
        operator_builder=triangular_operator_builder(; coupling=coupling),
        operator_label="triangular duplicate delay(coupling=$coupling)",
        base_spacing=base_spacing,
        target_support=target_support,
        max_refinement_rounds=max_refinement_rounds,
        chart_radii=chart_radii,
        iterations=2,
        basis_moments=6,
        basis_nodes=48,
        determinant_nodes=512,
        determinant_capacity=96,
        reduced_moments=12,
        reduced_nodes=512,
        residual_normalization=:operator,
        component_scaling=:contour_max,
        residual_tol=residual_tol,
        match_atol=match_atol,
        print_rows=print_rows,
    )
end

function run_near_pole_rational_count_driven_adaptive_refinement(;
    gap=0.01,
    coupling=5.0,
    outer_radius=1.0,
    base_spacing=0.35,
    target_support=2,
    max_refinement_rounds=4,
    chart_radii=(0.28, 0.45),
    residual_tol=1e-8,
    match_atol=1e-6,
    count_error_tol=1e-6,
    print_rows=true,
)
    run_count_driven_adaptive_grid_refinement(;
        label="Near-pole rational count-driven adaptive refinement",
        cases=oracle_free_near_pole_rational_cases(; gap=gap),
        outer_radius=outer_radius,
        operator_builder=triangular_operator_builder(; coupling=coupling),
        operator_label="triangular near-pole rational(coupling=$coupling,gap=$gap)",
        base_spacing=base_spacing,
        target_support=target_support,
        max_refinement_rounds=max_refinement_rounds,
        chart_radii=chart_radii,
        iterations=2,
        basis_moments=6,
        basis_nodes=48,
        determinant_nodes=512,
        determinant_capacity=64,
        reduced_moments=12,
        reduced_nodes=512,
        residual_normalization=:operator,
        component_scaling=:contour_max,
        residual_tol=residual_tol,
        match_atol=match_atol,
        count_error_tol=count_error_tol,
        print_rows=print_rows,
    )
end

function run_exponential_adaptive_grid_loewner_refinement(;
    base_spacing=2.4,
    target_support=2,
    loewner_radius=1.3,
    loewner_phase=0.0,
    loewner_points=6,
    chart_radii=(1.2, 2.0),
    residual_tol=1e-8,
    match_atol=1e-6,
    print_rows=true,
)
    run_adaptive_grid_loewner_refinement(;
        label="Exponential adaptive grid Loewner refinement",
        cases=exponential_many_root_cases(),
        outer_radius=10.0,
        base_spacing=base_spacing,
        target_support=target_support,
        refinement_rounds=1,
        loewner_radius=loewner_radius,
        loewner_phase=loewner_phase,
        loewner_points=loewner_points,
        chart_radii=chart_radii,
        residual_tol=residual_tol,
        match_atol=match_atol,
        print_rows=print_rows,
    )
end

function run_triangular_adaptive_grid_loewner_refinement(;
    coupling=10.0,
    outer_radius=6.0,
    base_spacing=2.4,
    target_support=2,
    refinement_rounds=2,
    chart_radii=(1.2, 1.8),
    residual_tol=1e-8,
    match_atol=1e-6,
    print_rows=true,
)
    run_adaptive_grid_loewner_refinement(;
        label="Triangular adaptive grid Loewner refinement",
        outer_radius=outer_radius,
        operator_builder=triangular_operator_builder(; coupling=coupling),
        operator_label="triangular(coupling=$coupling)",
        base_spacing=base_spacing,
        target_support=target_support,
        refinement_rounds=refinement_rounds,
        chart_radii=chart_radii,
        iterations=2,
        basis_nodes=24,
        rii_nodes=128,
        determinant_nodes=256,
        determinant_capacity=96,
        reduced_nodes=256,
        residual_normalization=:operator,
        component_scaling=:contour_max,
        residual_tol=residual_tol,
        match_atol=match_atol,
        print_rows=print_rows,
    )
end

function run_three_function_adaptive_grid_loewner_refinement(;
    outer_radius=20.0,
    base_spacing=3.0,
    target_support=2,
    refinement_rounds=2,
    loewner_radius=1.3,
    loewner_phase=0.0,
    loewner_points=6,
    chart_radii=(1.5, 2.4),
    residual_tol=1e-8,
    match_atol=1e-6,
    print_rows=true,
)
    run_adaptive_grid_loewner_refinement(;
        label="Three-function adaptive grid Loewner refinement",
        cases=(scalar_sine_case(), scalar_cosine_case(), scalar_shifted_sine_case()),
        outer_radius=outer_radius,
        base_spacing=base_spacing,
        target_support=target_support,
        refinement_rounds=refinement_rounds,
        loewner_radius=loewner_radius,
        loewner_phase=loewner_phase,
        loewner_points=loewner_points,
        chart_radii=chart_radii,
        residual_tol=residual_tol,
        match_atol=match_atol,
        print_rows=print_rows,
    )
end

function adaptive_retention_score_summary(
    result;
    outer_center=0.0 + 0.0im,
    outer_radius,
    match_atol=1e-6,
    count_error_tol=1e-2,
    target_count_estimate=nothing,
    target_count_error=nothing,
)
    final = result.refined
    expected = final.expected
    target_count = target_count_estimate === nothing ? length(expected) : Int(target_count_estimate)
    target_count_err = target_count_error === nothing ? 0.0 : Float64(target_count_error)
    clusters = final.support_clusters
    support1_global = globally_supported_cluster_values(
        clusters,
        outer_center,
        outer_radius;
        min_support=1,
        boundary_margin=10 * match_atol,
    )
    support2_global = globally_supported_cluster_values(
        clusters,
        outer_center,
        outer_radius;
        min_support=2,
        boundary_margin=10 * match_atol,
    )
    support3_global = globally_supported_cluster_values(
        clusters,
        outer_center,
        outer_radius;
        min_support=3,
        boundary_margin=10 * match_atol,
    )
    inside_clusters = [
        cluster for cluster in clusters
        if abs(cluster.value - outer_center) <= outer_radius + 10 * match_atol
    ]
    usable_records = [record for record in final.records if !record.failed]
    good_records = [record for record in usable_records if record.good > 0]
    selected_records = hasproperty(final, :selected_records) ? final.selected_records : final.records
    usable_selected_records = [record for record in selected_records if !record.failed]
    good_selected_records = [record for record in usable_selected_records if record.good > 0]
    (
        expected=length(expected),
        target_count_estimate=target_count,
        target_count_error=target_count_err,
        target_count_reliable=target_count_err <= count_error_tol,
        support1_global=length(support1_global),
        support1_global_matched=match_expected_count(support1_global, expected; atol=match_atol),
        support2_global=length(support2_global),
        support2_global_matched=match_expected_count(support2_global, expected; atol=match_atol),
        support2_count_complete=length(support2_global) == target_count,
        support3_global=length(support3_global),
        support3_global_matched=match_expected_count(support3_global, expected; atol=match_atol),
        support3_count_complete=length(support3_global) == target_count,
        weak_inside_clusters=count(cluster -> cluster.support < 2, inside_clusters),
        inside_cluster_count=length(inside_clusters),
        usable_records=length(usable_records),
        good_records=length(good_records),
        selected_usable_records=length(usable_selected_records),
        selected_good_records=length(good_selected_records),
        count_error_bad=count(record -> record.count_error > count_error_tol, good_records),
        count_deficit_records=count(record -> record.good < record.count_estimate, good_records),
        selected_count_error_bad=count(record -> record.count_error > count_error_tol, good_selected_records),
        selected_count_deficit_records=count(record -> record.good < record.count_estimate, good_selected_records),
        max_count_error=isempty(good_records) ? Inf : maximum(record.count_error for record in good_records),
        max_record_residual=isempty(good_records) ? Inf : maximum(record.max_residual for record in good_records),
        exact_support2_global=length(support2_global) == length(expected) &&
            match_expected_count(support2_global, expected; atol=match_atol) == length(expected),
    )
end

function full_operator_count_estimate(
    cases;
    outer_center=0.0 + 0.0im,
    outer_radius,
    operator_builder=similarity_analytic_tools,
    component_scaling=:none,
    component_scaling_nodes=64,
    determinant_nodes=2048,
    determinant_capacity=128,
)
    chart = ContourChart(
        outer_center,
        outer_radius;
        component_scaling=component_scaling,
        component_scaling_nodes=component_scaling_nodes,
    )
    ctx = analytic_context(cases, chart, operator_builder)
    _, count_estimate, sums = determinant_power_sums(
        ctx.Tmatrix,
        ctx.Tderivative;
        center=outer_center,
        radius=outer_radius,
        nodes=determinant_nodes,
        capacity=determinant_capacity,
    )
    (
        count_estimate=count_estimate,
        count_error=abs(sums[1] - count_estimate),
        expected=length(ctx.expected),
    )
end

function chart_policy_plan(
    result,
    summary;
    outer_center=0.0 + 0.0im,
    outer_radius,
    match_atol=1e-6,
    count_error_tol=1e-2,
)
    final = result.refined
    retained = globally_supported_cluster_values(
        final.support_clusters,
        outer_center,
        outer_radius;
        min_support=2,
        boundary_margin=10 * match_atol,
    )
    weak_target_centers = sorted_unique_values(
        ComplexF64[
            cluster.value for cluster in final.support_clusters
            if cluster.support < 2 && abs(cluster.value - outer_center) <= outer_radius + 10 * match_atol
        ];
        atol=match_atol,
    )
    selected_records = hasproperty(final, :selected_records) ? final.selected_records : final.records
    count_stressed = [
        record for record in selected_records
        if !record.failed && record.good > 0 &&
            (record.good < record.count_estimate || record.count_error > count_error_tol)
    ]
    split_records = NamedTuple[]
    for record in count_stressed
        child_radius = record.radius / 2
        # Four overlapping child charts preserve the disk geometry while
        # reducing local realization size around count-stressed regions.
        child_centers = ComplexF64[
            record.center + child_radius / 2,
            record.center - child_radius / 2,
            record.center + im * child_radius / 2,
            record.center - im * child_radius / 2,
        ]
        candidate_centers = sorted_unique_values(
            ComplexF64[
                entry.value for entry in final.found_entries
                if abs(entry.center - record.center) <= match_atol &&
                    abs(entry.radius - record.radius) <= 10 * eps(Float64) * max(1.0, record.radius)
            ];
            atol=match_atol,
        )
        refinement_centers = sorted_unique_values(
            vcat(candidate_centers, ComplexF64[record.center], child_centers);
            atol=match_atol,
        )
        stress = record.good < record.count_estimate ? :count_deficit : :count_error
        candidate_radii = stress === :count_deficit ?
            (record.radius / 4, 3 * record.radius / 8, 5 * record.radius / 8) :
            (record.radius / 2, 2 * record.radius / 3, record.radius)
        push!(
            split_records,
            (
                center=record.center,
                radius=record.radius,
                child_radius=child_radius,
                child_centers=child_centers,
                candidate_centers=candidate_centers,
                refinement_centers=refinement_centers,
                candidate_radii=candidate_radii,
                selected=true,
                stress=stress,
                good=record.good,
                count_estimate=record.count_estimate,
                count_error=record.count_error,
                max_residual=record.max_residual,
            ),
        )
    end
    (
        retained=retained,
        retained_count=length(retained),
        retained_expected=summary.expected,
        target_count_estimate=summary.target_count_estimate,
        target_count_error=summary.target_count_error,
        weak_target_centers=weak_target_centers,
        weak_target_count=length(weak_target_centers),
        count_stressed_records=split_records,
        count_stressed_count=length(split_records),
        actions=(
            isempty(weak_target_centers) ? :no_weak_target_centers : :add_weak_target_candidate_centers,
            isempty(split_records) ? :no_count_stressed_charts : :split_or_shrink_count_stressed_charts,
            !summary.support3_count_complete ?
                :do_not_raise_support_threshold_without_cover_density :
                :support_threshold_can_be_reconsidered,
        ),
    )
end

function run_three_function_retention_score_diagnostic(;
    outer_radius=20.0,
    base_spacing=3.0,
    chart_radii=(1.5, 2.4),
    loewner_radius=1.3,
    loewner_points=6,
    count_error_tol=1e-2,
    residual_tol=1e-8,
    match_atol=1e-6,
    include_layout_agreement=false,
    include_extractor_agreement=false,
    print_rows=true,
)
    result = run_three_function_adaptive_grid_loewner_refinement(;
        outer_radius=outer_radius,
        base_spacing=base_spacing,
        chart_radii=chart_radii,
        refinement_rounds=2,
        loewner_radius=loewner_radius,
        loewner_points=loewner_points,
        residual_tol=residual_tol,
        match_atol=match_atol,
        print_rows=false,
    )
    count = full_operator_count_estimate(
        (scalar_sine_case(), scalar_cosine_case(), scalar_shifted_sine_case());
        outer_radius=outer_radius,
        determinant_nodes=2048,
        determinant_capacity=128,
    )
    summary = adaptive_retention_score_summary(
        result;
        outer_radius=outer_radius,
        match_atol=match_atol,
        count_error_tol=count_error_tol,
        target_count_estimate=count.count_estimate,
        target_count_error=count.count_error,
    )
    plan = chart_policy_plan(
        result,
        summary;
        outer_radius=outer_radius,
        match_atol=match_atol,
        count_error_tol=count_error_tol,
    )
    layout_agreement = include_layout_agreement ? run_three_function_adaptive_grid_loewner_layout_sweep(;
        radii=(1.15, 1.3, 1.6),
        phase_fractions=(0.0,),
        loewner_points=loewner_points,
        residual_tol=residual_tol,
        match_atol=match_atol,
        print_rows=false,
    ) : nothing
    extractor_agreement = include_extractor_agreement ? run_three_function_adaptive_grid_extractor_agreement(;
        extractors=(:loewner_counted, :ss_counted),
        loewner_radius=loewner_radius,
        loewner_points=loewner_points,
        residual_tol=residual_tol,
        match_atol=match_atol,
        print_rows=false,
    ) : nothing
    evidence = (
        support_and_target=summary.exact_support2_global,
        oracle_free_count_complete=summary.target_count_reliable && summary.support2_count_complete,
        local_count_warning=summary.count_deficit_records > 0 || summary.count_error_bad > 0,
        residual_ok=summary.max_record_residual <= 10 * residual_tol,
        layout_agreement=layout_agreement === nothing ? missing : layout_agreement.summary.success,
        extractor_agreement=extractor_agreement === nothing ? missing : extractor_agreement.summary.success,
    )
    if print_rows
        println()
        println("Three-function adaptive retention-score diagnostic")
        println("  reports support thresholds, target-domain weak clusters, and local count-estimator stress")
        @printf(
            "  expected=%d target_count=%d count_error=%.3e support1_global=%d/%d support2_global=%d/%d support3_global=%d/%d weak_inside=%d\n",
            summary.expected,
            summary.target_count_estimate,
            summary.target_count_error,
            summary.support1_global_matched,
            summary.support1_global,
            summary.support2_global_matched,
            summary.support2_global,
            summary.support3_global_matched,
            summary.support3_global,
            summary.weak_inside_clusters,
        )
        @printf(
            "  local records usable=%d good=%d selected_good=%d count_error_bad=%d count_deficit=%d selected_count_deficit=%d max_count_error=%.3e max_residual=%.3e status=%s\n",
            summary.usable_records,
            summary.good_records,
            summary.selected_good_records,
            summary.count_error_bad,
            summary.count_deficit_records,
            summary.selected_count_deficit_records,
            summary.max_count_error,
            summary.max_record_residual,
            summary.exact_support2_global ? "ok" : "check",
        )
        println("  evidence=", evidence)
        println("  plan=", plan.actions)
    end
    (
        result=result,
        summary=summary,
        plan=plan,
        evidence=evidence,
        layout_agreement=layout_agreement,
        extractor_agreement=extractor_agreement,
    )
end

function run_count_stressed_split_probe(;
    residual_tol=1e-8,
    match_atol=1e-6,
    print_rows=true,
)
    diagnostic = run_three_function_retention_score_diagnostic(;
        residual_tol=residual_tol,
        match_atol=match_atol,
        print_rows=false,
    )
    isempty(diagnostic.plan.count_stressed_records) && error("no count-stressed charts available to probe")
    rows = NamedTuple[]
    for record in diagnostic.plan.count_stressed_records
        naive_child = run_dual_local_chart_sweep_analytic(;
            cases=(scalar_sine_case(), scalar_cosine_case(), scalar_shifted_sine_case()),
            outer_center=record.center,
            outer_radius=record.radius,
            centers=record.child_centers,
            radii=(record.child_radius,),
            iterations=1,
            extractor=:loewner_counted,
            loewner_points=6,
            loewner_radius=1.3,
            residual_normalization=:vector,
            component_scaling=:none,
            selection=:residual,
            skip_empty_expected=false,
            print_charts=false,
        )
        candidate_child = run_dual_local_chart_sweep_analytic(;
            cases=(scalar_sine_case(), scalar_cosine_case(), scalar_shifted_sine_case()),
            outer_center=record.center,
            outer_radius=record.radius,
            centers=record.refinement_centers,
            radii=record.candidate_radii,
            iterations=2,
            extractor=:loewner_counted,
            loewner_points=6,
            loewner_radius=1.3,
            residual_normalization=:vector,
            component_scaling=:none,
            selection=:residual,
            skip_empty_expected=false,
            print_charts=false,
        )
        push!(
            rows,
            (
                parent=record,
                naive_child=naive_child,
                candidate_child=candidate_child,
                expected=length(candidate_child.expected),
            ),
        )
    end
    record = rows[1].parent
    naive_child = rows[1].naive_child
    candidate_child = rows[1].candidate_child
    if print_rows
        println()
        println("Count-stressed split/shrink probe")
        println("  compares blind geometric children with residual candidates plus overlapping cover anchors")
        for row in rows
            @printf(
                "  parent_center=%.6g%+.6gi parent_r=%.3g parent_good=%d parent_count=%d naive=%d/%d refined=%d/%d refined_support2=%d/%d\n",
                real(row.parent.center),
                imag(row.parent.center),
                row.parent.radius,
                row.parent.good,
                row.parent.count_estimate,
                row.naive_child.matched,
                row.expected,
                row.candidate_child.matched,
                row.expected,
                row.candidate_child.support2_global_matched,
                row.expected,
            )
        end
    end
    (
        base=diagnostic,
        rows=rows,
        parent=record,
        naive_child=naive_child,
        candidate_child=candidate_child,
        child_centers=record.child_centers,
        child_radius=record.child_radius,
        candidate_centers=record.candidate_centers,
        refinement_centers=record.refinement_centers,
        candidate_radii=record.candidate_radii,
    )
end

function run_triangular_count_error_split_probe(;
    coupling=10.0,
    outer_radius=6.0,
    residual_tol=1e-8,
    match_atol=1e-6,
    print_rows=true,
)
    result = run_triangular_adaptive_grid_loewner_refinement(;
        coupling=coupling,
        outer_radius=outer_radius,
        residual_tol=residual_tol,
        match_atol=match_atol,
        print_rows=false,
    )
    summary = adaptive_retention_score_summary(
        (refined=result.refined,);
        outer_radius=outer_radius,
        match_atol=match_atol,
    )
    plan = chart_policy_plan(
        (refined=result.refined,),
        summary;
        outer_radius=outer_radius,
        match_atol=match_atol,
    )
    record_index = findfirst(record -> record.stress === :count_error, plan.count_stressed_records)
    record_index === nothing && error("no count-error-only triangular chart available to probe")
    record = plan.count_stressed_records[record_index]
    common = (
        cases=(scalar_sine_case(), scalar_cosine_case(), scalar_shifted_sine_case(), scalar_expm1_case()),
        operator_builder=triangular_operator_builder(; coupling=coupling),
        operator_label="triangular(coupling=$coupling)",
        outer_center=record.center,
        outer_radius=record.radius,
        centers=record.refinement_centers,
        iterations=2,
        basis_nodes=24,
        rii_nodes=128,
        determinant_nodes=256,
        determinant_capacity=96,
        reduced_nodes=256,
        extractor=:loewner_counted,
        loewner_points=6,
        loewner_radius=1.3,
        residual_normalization=:operator,
        component_scaling=:contour_max,
        selection=:residual,
        skip_empty_expected=false,
        print_charts=false,
    )
    shrink_child = run_dual_local_chart_sweep_analytic(;
        common...,
        radii=(record.radius / 4, 3 * record.radius / 8, 5 * record.radius / 8),
    )
    policy_child = run_dual_local_chart_sweep_analytic(;
        common...,
        radii=record.candidate_radii,
    )
    if print_rows
        println()
        println("Triangular count-error split/shrink probe")
        println("  count-error-only nonnormal charts preserve the parent radius instead of shrinking aggressively")
        @printf(
            "  parent_center=%.6g%+.6gi parent_r=%.3g count_error=%.3e shrink=%d/%d support2=%d/%d policy=%d/%d support2=%d/%d\n",
            real(record.center),
            imag(record.center),
            record.radius,
            record.count_error,
            shrink_child.matched,
            length(shrink_child.expected),
            shrink_child.support2_global_matched,
            length(shrink_child.expected),
            policy_child.matched,
            length(policy_child.expected),
            policy_child.support2_global_matched,
            length(policy_child.expected),
        )
    end
    (
        base=result,
        summary=summary,
        plan=plan,
        parent=record,
        shrink_child=shrink_child,
        policy_child=policy_child,
    )
end

function near_pole_rational_cases(; gap=0.005)
    radius = 1.0 + Float64(gap)
    (
        scalar_rational_case(; root=-0.45, pole=radius, name="rat1"),
        scalar_rational_case(; root=-0.10 + 0.20im, pole=radius * exp(0.15im), name="rat2"),
        scalar_rational_case(; root=0.20 - 0.15im, pole=radius * exp(-0.12im), name="rat3"),
        scalar_rational_case(; root=0.48 + 0.10im, pole=radius * exp(0.05im), name="rat4"),
        scalar_rational_case(; root=0.68 - 0.05im, pole=radius * exp(-0.08im), name="rat5"),
    )
end

function oracle_free_near_pole_rational_cases(; gap=0.01)
    Tuple(
        merge(case, (roots=(center, radius) -> ComplexF64[],))
        for case in near_pole_rational_cases(; gap=gap)
    )
end

function run_near_pole_rational_boundary_diagnostic(;
    gap=0.005,
    residual_tol=1e-8,
    match_atol=1e-6,
    print_rows=true,
)
    cases = near_pole_rational_cases(; gap=gap)
    configs = (
        (component_scaling=:none, residual_normalization=:operator, extractor=:loewner_counted),
        (component_scaling=:none, residual_normalization=:operator, extractor=:ss_counted),
        (component_scaling=:none, residual_normalization=:vector, extractor=:loewner_counted),
        (component_scaling=:contour_max, residual_normalization=:vector, extractor=:ss_counted),
    )
    rows = NamedTuple[]
    for config in configs
        append!(
            rows,
            run_matrix_analytic_case(;
                name="near_pole_rational_gap_$(gap)",
                problem_class=:rational_nep,
                cases=cases,
                radius=1.0,
                iterations=1,
                basis_moments=3,
                basis_nodes=32,
                rii_nodes=128,
                basis_ranktol=1e-10,
                determinant_nodes=512,
                determinant_capacity=32,
                extractor=config.extractor,
                loewner_points=6,
                loewner_radius=1.3,
                reduced_moments=8,
                reduced_nodes=512,
                component_scaling=config.component_scaling,
                residual_normalization=config.residual_normalization,
                residual_tol=residual_tol,
                match_atol=match_atol,
            ),
        )
    end
    success = all(row.success for row in rows)
    if print_rows
        println()
        println("Near-pole rational boundary diagnostic")
        println("  poles sit just outside the target contour; checks whether rational singularities need a special chart policy")
        for row in rows
            @printf(
                "  %-18s %-14s %-15s %-18s matched=%d/%d good=%d spurious=%d max=%.3e near=%.3e %s\n",
                string(row.problem),
                string(row.stage),
                string(row.problem_class),
                row.notes,
                row.matched,
                row.expected,
                row.good,
                row.spurious,
                row.max_residual,
                row.nearest_expected,
                row.success ? "ok" : "check",
            )
        end
    end
    (rows=rows, success=success, gap=gap)
end

function run_residual_laurent_compression_diagnostic(;
    residual_tol=1e-8,
    match_atol=1e-6,
    print_rows=true,
)
    common = (
        cases=(scalar_sine_case(), scalar_cosine_case(), scalar_shifted_sine_case()),
        radius=10.0,
        basis_moments=4,
        basis_nodes=8,
        rii_nodes=128,
        update_moment_count=1,
        iterations=1,
        basis_ranktol=0.5,
        determinant_nodes=512,
        determinant_capacity=64,
        extractor=:loewner_counted,
        loewner_points=6,
        loewner_radius=1.3,
        reduced_moments=12,
        reduced_nodes=512,
        residual_normalization=:vector,
        component_scaling=:none,
        residual_tol=residual_tol,
        match_atol=match_atol,
        verbose=false,
    )
    compressed = run_dual_moment_compressed_rii_analytic_iteration(;
        common...,
        update_mode=:moment_compressed,
    )
    scalar = run_dual_moment_compressed_rii_analytic_iteration(;
        common...,
        update_mode=:scalar_expanded,
    )
    compressed_initial = compressed.summaries[1]
    compressed_updated = compressed.summaries[end]
    scalar_initial = scalar.summaries[1]
    scalar_updated = scalar.summaries[end]
    if print_rows
        println()
        println("Residual Laurent compression diagnostic")
        println("  compares moment-realization update with scalar expanded RII on the same rank-deficient analytic chart")
        @printf(
            "  initial inside=%d good=%d max=%.3e; compressed matched=%d/%d candidates=(%d,%d) basis=(%d,%d) residual_rank=(%d,%d); scalar matched=%d/%d candidates=(%d,%d) basis=(%d,%d) residual_rank=(%d,%d)\n",
            compressed_initial.inside,
            compressed_initial.good,
            compressed_initial.max_residual,
            compressed_updated.matched,
            length(compressed.expected),
            compressed_updated.right_candidate_cols,
            compressed_updated.left_candidate_cols,
            compressed_updated.right_basis_cols,
            compressed_updated.left_basis_cols,
            compressed_updated.right_residual_rank,
            compressed_updated.left_residual_rank,
            scalar_updated.matched,
            length(scalar.expected),
            scalar_updated.right_candidate_cols,
            scalar_updated.left_candidate_cols,
            scalar_updated.right_basis_cols,
            scalar_updated.left_basis_cols,
            scalar_updated.right_residual_rank,
            scalar_updated.left_residual_rank,
        )
    end
    (
        compressed=compressed,
        scalar=scalar,
        compressed_initial=compressed_initial,
        compressed_updated=compressed_updated,
        scalar_initial=scalar_initial,
        scalar_updated=scalar_updated,
        expected=length(compressed.expected),
    )
end

function run_residual_laurent_update_ladder_diagnostic(;
    coupling=10.0,
    outer_radius=20.0,
    spacing=2.4,
    chart_radii=(0.8, 1.2, 1.8, 2.4, 3.0),
    iteration_counts=(0, 1, 2),
    residual_tol=1e-8,
    match_atol=1e-6,
    print_rows=true,
)
    rows = NamedTuple[]
    results = Any[]
    for iterations in iteration_counts
        result = run_dual_grid_chart_cover_triangular_analytic(;
            coupling=coupling,
            outer_radius=outer_radius,
            spacing=spacing,
            chart_radii=chart_radii,
            basis_nodes=32,
            rii_nodes=512,
            iterations=iterations,
            basis_ranktol=1e-8,
            determinant_nodes=1024,
            reduced_nodes=1024,
            residual_normalization=:vector,
            component_scaling=:contour_max,
            residual_tol=residual_tol,
            match_atol=match_atol,
            print_charts=false,
        )
        push!(results, result)
        push!(
            rows,
            (
                iterations=iterations,
                expected=length(result.expected),
                union=length(result.found),
                matched=result.matched,
                retained=length(result.support2_global_found),
                retained_matched=result.support2_global_matched,
            ),
        )
    end
    if print_rows
        println()
        println("Residual Laurent update ladder diagnostic")
        println("  triangular nonnormal chart cover; iterations=0 is reduced extraction only")
        @printf("  %-10s %8s %12s %12s %12s\n", "updates", "union", "matched", "retained", "retained_ok")
        for row in rows
            @printf(
                "  %-10d %8d %5d/%-6d %5d/%-6d %5d/%-6d\n",
                row.iterations,
                row.union,
                row.matched,
                row.expected,
                row.retained,
                row.expected,
                row.retained_matched,
                row.expected,
            )
        end
    end
    (rows=rows, results=results, final=last(rows))
end

function retention_policy_decision(
    summary;
    residual_tol=1e-8,
    layout_agreement=missing,
    extractor_agreement=missing,
)
    actions = Symbol[]
    target_count_ok = summary.target_count_reliable && summary.support2_count_complete
    if target_count_ok
        push!(actions, :retain_support2_global)
    else
        push!(actions, :refine_weak_target_support)
    end
    if summary.weak_inside_clusters > 0
        push!(actions, :add_weak_target_candidate_centers)
    end
    if target_count_ok && !summary.support3_count_complete
        push!(actions, :do_not_raise_support_threshold_without_cover_density)
    end
    summary.target_count_reliable || push!(actions, :tighten_target_count_estimator)
    local_count_warning = summary.count_deficit_records > 0 || summary.count_error_bad > 0
    if local_count_warning
        push!(actions, :treat_local_count_errors_as_chart_warnings)
        push!(actions, :split_or_shrink_count_stressed_charts_before_strict_acceptance)
    end
    if summary.max_record_residual > 10 * residual_tol
        push!(actions, :tighten_or_refine_high_residual_charts)
    end
    if layout_agreement === missing
        if local_count_warning || !target_count_ok
            push!(actions, :request_loewner_layout_agreement)
        end
    elseif !layout_agreement
        push!(actions, :reject_or_split_layout_unstable_candidates)
    else
        push!(actions, :layout_agreement_certified)
    end
    if extractor_agreement === missing
        if local_count_warning || !target_count_ok
            push!(actions, :request_reduced_extractor_agreement)
        end
    elseif !extractor_agreement
        push!(actions, :escalate_reduced_extractor_disagreement)
    else
        push!(actions, :extractor_agreement_certified)
    end

    residual_ok = summary.max_record_residual <= 10 * residual_tol
    support_ok = target_count_ok
    layout_ok = layout_agreement === missing ? !local_count_warning : Bool(layout_agreement)
    extractor_ok = extractor_agreement === missing ? !local_count_warning : Bool(extractor_agreement)
    status = if support_ok && residual_ok && layout_ok && extractor_ok && !local_count_warning
        :accept
    elseif support_ok && residual_ok && layout_ok && extractor_ok
        :accept_with_chart_warnings
    elseif !support_ok
        :refine
    else
        :escalate
    end
    (
        status=status,
        retain_support=2,
        retained=summary.support2_global,
        expected=summary.target_count_estimate,
        validation_expected=summary.expected,
        target_count_error=summary.target_count_error,
        support_ok=support_ok,
        residual_ok=residual_ok,
        local_count_warning=local_count_warning,
        layout_ok=layout_ok,
        extractor_ok=extractor_ok,
        actions=Tuple(actions),
    )
end

function run_three_function_automatic_retention_policy(;
    outer_radius=20.0,
    base_spacing=3.0,
    chart_radii=(1.5, 2.4),
    loewner_radius=1.3,
    loewner_points=6,
    count_error_tol=1e-2,
    residual_tol=1e-8,
    match_atol=1e-6,
    print_rows=true,
)
    base = run_three_function_retention_score_diagnostic(;
        outer_radius=outer_radius,
        base_spacing=base_spacing,
        chart_radii=chart_radii,
        loewner_radius=loewner_radius,
        loewner_points=loewner_points,
        count_error_tol=count_error_tol,
        residual_tol=residual_tol,
        match_atol=match_atol,
        include_layout_agreement=false,
        include_extractor_agreement=false,
        print_rows=false,
    )
    initial_decision = retention_policy_decision(
        base.summary;
        residual_tol=residual_tol,
        layout_agreement=missing,
        extractor_agreement=missing,
    )
    needs_layout = :request_loewner_layout_agreement in initial_decision.actions
    needs_extractor = :request_reduced_extractor_agreement in initial_decision.actions
    layout_agreement = needs_layout ? run_three_function_adaptive_grid_loewner_layout_sweep(;
        radii=(1.15, 1.3, 1.6),
        phase_fractions=(0.0,),
        loewner_points=loewner_points,
        residual_tol=residual_tol,
        match_atol=match_atol,
        print_rows=false,
    ) : nothing
    extractor_agreement = needs_extractor ? run_three_function_adaptive_grid_extractor_agreement(;
        extractors=(:loewner_counted, :ss_counted),
        loewner_radius=loewner_radius,
        loewner_points=loewner_points,
        residual_tol=residual_tol,
        match_atol=match_atol,
        print_rows=false,
    ) : nothing
    final_decision = retention_policy_decision(
        base.summary;
        residual_tol=residual_tol,
        layout_agreement=layout_agreement === nothing ? missing : layout_agreement.summary.success,
        extractor_agreement=extractor_agreement === nothing ? missing : extractor_agreement.summary.success,
    )
    if print_rows
        println()
        println("Three-function automatic retention policy")
        @printf(
            "  initial_status=%s final_status=%s retained=%d/%d residual_ok=%s local_count_warning=%s\n",
            string(initial_decision.status),
            string(final_decision.status),
            final_decision.retained,
            final_decision.expected,
            string(final_decision.residual_ok),
            string(final_decision.local_count_warning),
        )
        println("  initial_actions=", initial_decision.actions)
        println("  final_actions=", final_decision.actions)
    end
    (
        base=base,
        initial_decision=initial_decision,
        final_decision=final_decision,
        layout_agreement=layout_agreement,
        extractor_agreement=extractor_agreement,
    )
end

function find_refinement_row(result, stage, refinement)
    for row in result.rows
        row.stage === stage && row.refinement === refinement && return row
    end
    error("missing refinement row for stage=$stage refinement=$refinement")
end

function run_analytic_block_newton_boundary_diagnostic(; print_rows=true)
    small = run_reduced_analytic_refinement_comparison(;
        cases=(scalar_sine_case(), scalar_cosine_case()),
        radius=4.0,
        extractor=:ss_counted,
        determinant_nodes=512,
        reduced_nodes=256,
        refinement_modes=(:none, :scalar_newton, :block_newton),
        refinement_steps=1,
        refinement_nodes=128,
        print_rows=print_rows,
    )
    large = run_reduced_analytic_refinement_comparison(;
        radius=20.0,
        extractor=:ss_counted,
        determinant_nodes=512,
        reduced_nodes=256,
        refinement_modes=(:none, :scalar_newton, :block_newton),
        refinement_steps=1,
        refinement_nodes=128,
        print_rows=print_rows,
    )
    small_block_initial = find_refinement_row(small, :initial, :block_newton)
    small_none_updated = find_refinement_row(small, :updated, :none)
    small_block_updated = find_refinement_row(small, :updated, :block_newton)
    large_none_updated = find_refinement_row(large, :updated, :none)
    large_scalar_updated = find_refinement_row(large, :updated, :scalar_newton)
    large_block_updated = find_refinement_row(large, :updated, :block_newton)

    if print_rows
        println()
        println("Analytic invariant-pair block-Newton boundary diagnostic")
        println("  true contour-residual block Newton is a reduced local refinement, not a large-chart retention policy")
        @printf(
            "  small radius-4: initial block ratio=%s updated none=%d/%d max=%.3e updated block=%d/%d max=%.3e\n",
            isempty(small_block_initial.newton_ratios) ? "n/a" : @sprintf("%.2e", first(small_block_initial.newton_ratios)),
            small_none_updated.matched,
            small.expected,
            small_none_updated.max_residual,
            small_block_updated.matched,
            small.expected,
            small_block_updated.max_residual,
        )
        @printf(
            "  large radius-20: updated none=%d/%d max=%.3e scalar=%d/%d max=%.3e block=%d/%d max=%.3e block_ratio=%s\n",
            large_none_updated.matched,
            large.expected,
            large_none_updated.max_residual,
            large_scalar_updated.matched,
            large.expected,
            large_scalar_updated.max_residual,
            large_block_updated.matched,
            large.expected,
            large_block_updated.max_residual,
            isempty(large_block_updated.newton_ratios) ? "n/a" : @sprintf("%.2e", first(large_block_updated.newton_ratios)),
        )
    end
    (
        small=small,
        large=large,
        small_block_initial=small_block_initial,
        small_none_updated=small_none_updated,
        small_block_updated=small_block_updated,
        large_none_updated=large_none_updated,
        large_scalar_updated=large_scalar_updated,
        large_block_updated=large_block_updated,
    )
end

function global_loewner_artifact_retention_policy(summary)
    single_layout_safe = summary.bad_single_layouts == 0 && summary.initial_support1_spurious == 0 &&
        summary.updated_support1_spurious == 0
    cross_layout_safe = summary.support2_prunes_artifacts &&
        summary.updated_support2_matched == summary.expected &&
        summary.updated_support2_count == summary.expected
    actions = Symbol[]
    if !single_layout_safe
        push!(actions, :do_not_accept_single_layout_residual_small_values)
        push!(actions, :require_cross_layout_support2)
    end
    if cross_layout_safe
        push!(actions, :retain_cross_layout_support2)
    else
        push!(actions, :split_chart_or_change_reduced_extractor)
    end
    (
        status=cross_layout_safe ? :accept_cross_layout : :escalate,
        single_layout_safe=single_layout_safe,
        cross_layout_safe=cross_layout_safe,
        retained=summary.updated_support2_count,
        expected=summary.expected,
        actions=Tuple(actions),
    )
end

function run_global_loewner_artifact_retention_policy(; print_rows=true)
    diagnostic = run_global_loewner_interior_artifact_diagnostic(; print_rows=false)
    decision = global_loewner_artifact_retention_policy(diagnostic.summary)
    if print_rows
        println()
        println("Global Loewner artifact retention policy")
        @printf(
            "  status=%s single_layout_safe=%s cross_layout_safe=%s retained=%d/%d\n",
            string(decision.status),
            string(decision.single_layout_safe),
            string(decision.cross_layout_safe),
            decision.retained,
            decision.expected,
        )
        println("  actions=", decision.actions)
    end
    (diagnostic=diagnostic, decision=decision)
end

function run_three_function_adaptive_grid_extractor_agreement(;
    extractors=(:loewner_counted, :ss_counted),
    loewner_radius=1.3,
    loewner_points=6,
    min_extractor_support=2,
    residual_tol=1e-8,
    match_atol=1e-6,
    print_rows=true,
)
    rows = NamedTuple[]
    entries = NamedTuple[]
    results = Any[]
    expected = ComplexF64[]
    for (extractor_index, extractor) in pairs(extractors)
        result = run_adaptive_grid_loewner_refinement(;
            label="Three-function adaptive $(extractor) refinement",
            cases=(scalar_sine_case(), scalar_cosine_case(), scalar_shifted_sine_case()),
            outer_radius=20.0,
            base_spacing=3.0,
            target_support=2,
            refinement_rounds=2,
            loewner_radius=loewner_radius,
            loewner_points=loewner_points,
            chart_radii=(1.5, 2.4),
            extractor=extractor,
            residual_tol=residual_tol,
            match_atol=match_atol,
            print_rows=false,
        )
        push!(results, result)
        expected = ComplexF64.(result.refined.expected)
        final = result.rows[end]
        push!(
            rows,
            (
                extractor=extractor,
                centers=final.centers,
                expected=final.expected,
                support2_global=final.support2_global,
                support2_global_matched=final.support2_global_matched,
                success=final.success_support2_global,
            ),
        )
        extractor_id = ComplexF64(extractor_index, 0.0)
        append!(
            entries,
            [
                (
                    value=ComplexF64(value),
                    residual=0.0,
                    right_residual=0.0,
                    left_residual=0.0,
                    center=extractor_id,
                    radius=Float64(extractor_index),
                    extractor=extractor,
                )
                for value in result.refined.support2_global_found
            ],
        )
    end
    clusters = loewner_layout_clusters(entries; atol=match_atol)
    supported = loewner_supported_cluster_values(clusters; min_support=min_extractor_support)
    matched = match_expected_count(supported, expected; atol=match_atol)
    summary = (
        extractors=length(rows),
        expected=length(expected),
        support=min_extractor_support,
        supported=length(supported),
        matched=matched,
        success=matched == length(expected) && length(supported) == length(expected),
    )
    if print_rows
        println()
        println("Three-function adaptive reduced-extractor agreement")
        println("  reruns target-limited adaptive charting across reduced extractors")
        @printf("  %-18s %8s %10s %14s %8s\n", "extractor", "centers", "global2", "global2_ok", "status")
        for row in rows
            @printf(
                "  %-18s %8d %8d %4d/%-7d %s\n",
                string(row.extractor),
                row.centers,
                row.support2_global,
                row.support2_global_matched,
                row.expected,
                row.success ? "ok" : "check",
            )
        end
        @printf(
            "  cross-extractor support>=%d count=%d matched=%d/%d status=%s\n",
            summary.support,
            summary.supported,
            summary.matched,
            summary.expected,
            summary.success ? "ok" : "check",
        )
    end
    (
        extractors=extractors,
        rows=rows,
        results=results,
        clusters=clusters,
        supported_values=supported,
        summary=summary,
    )
end

function run_three_function_adaptive_grid_loewner_layout_sweep(;
    radii=(1.15, 1.3, 1.6),
    phase_fractions=(0.0,),
    loewner_points=6,
    min_layout_support=2,
    residual_tol=1e-8,
    match_atol=1e-6,
    print_rows=true,
)
    rows = NamedTuple[]
    entries = NamedTuple[]
    results = Any[]
    expected = ComplexF64[]
    for rho in radii, fraction in phase_fractions
        phase = 2pi * fraction / loewner_points
        result = run_three_function_adaptive_grid_loewner_refinement(;
            loewner_radius=rho,
            loewner_phase=phase,
            loewner_points=loewner_points,
            residual_tol=residual_tol,
            match_atol=match_atol,
            print_rows=false,
        )
        push!(results, result)
        expected = ComplexF64.(result.refined.expected)
        final = result.rows[end]
        push!(
            rows,
            (
                rho=Float64(rho),
                phase=Float64(phase),
                centers=final.centers,
                expected=final.expected,
                support2_global=final.support2_global,
                support2_global_matched=final.support2_global_matched,
                success=final.success_support2_global,
            ),
        )
        layout_id = ComplexF64(rho, phase)
        append!(
            entries,
            [
                (
                    value=ComplexF64(value),
                    residual=0.0,
                    right_residual=0.0,
                    left_residual=0.0,
                    center=layout_id,
                    radius=Float64(rho),
                    phase=Float64(phase),
                )
                for value in result.refined.support2_global_found
            ],
        )
    end
    clusters = loewner_layout_clusters(entries; atol=match_atol)
    supported = loewner_supported_cluster_values(clusters; min_support=min_layout_support)
    matched = match_expected_count(supported, expected; atol=match_atol)
    summary = (
        layouts=length(rows),
        expected=length(expected),
        support=min_layout_support,
        supported=length(supported),
        matched=matched,
        success=matched == length(expected) && length(supported) == length(expected),
    )
    if print_rows
        println()
        println("Three-function adaptive Loewner layout sweep")
        println("  reruns target-limited adaptive charting across Loewner interpolation layouts")
        @printf("  %-6s %-8s %8s %12s %12s %8s\n", "rho", "phase", "centers", "global2", "matched", "status")
        for row in rows
            @printf(
                "  %-6.2f %-8.3f %8d %5d/%-6d %5d/%-6d %s\n",
                row.rho,
                row.phase,
                row.centers,
                row.support2_global,
                row.expected,
                row.support2_global_matched,
                row.expected,
                row.success ? "ok" : "check",
            )
        end
        @printf(
            "  layout_support>=%d supported=%d matched=%d/%d %s\n",
            min_layout_support,
            summary.supported,
            summary.matched,
            summary.expected,
            summary.success ? "ok" : "check",
        )
    end
    (rows=rows, clusters=clusters, supported=supported, summary=summary, results=results)
end

function run_matrix_local_chart_case(;
    name="triangular_local_charts",
    problem_class=:analytic_nonnormal,
    dimension=4,
    outer_radius=4.0,
    radii=(2.4,),
    iterations=2,
    basis_ranktol=1e-8,
    basis_nodes=24,
    rii_nodes=128,
    determinant_nodes=256,
    reduced_nodes=256,
    residual_normalization=:operator,
    component_scaling=:contour_max,
    match_atol=1e-6,
)
    result = run_dual_local_chart_sweep_analytic(;
        outer_radius=outer_radius,
        operator_builder=triangular_operator_builder(; coupling=10.0),
        operator_label="triangular(coupling=10)",
        radii=radii,
        iterations=iterations,
        basis_ranktol=basis_ranktol,
        basis_nodes=basis_nodes,
        rii_nodes=rii_nodes,
        determinant_nodes=determinant_nodes,
        reduced_nodes=reduced_nodes,
        residual_normalization=residual_normalization,
        component_scaling=component_scaling,
        print_charts=false,
    )
    [
        experiment_matrix_support_row(
            name,
            problem_class,
            :support2_chart_merge,
            result;
            match_atol=match_atol,
            dimension=dimension,
            notes=@sprintf("union=%d support2=%d", length(result.found), length(result.support2_found)),
        ),
    ]
end

function scalar_rational_case(; root, pole, name=nothing)
    root = ComplexF64(root)
    pole = ComplexF64(pole)
    label = name === nothing ? "rational_$(root)_pole_$(pole)" : String(name)
    (
        name=label,
        f=z -> (z - root) / (z - pole),
        df=z -> (root - pole) / (z - pole)^2,
        fmat=S -> begin
            Ired = Matrix{ComplexF64}(I, size(S, 1), size(S, 2))
            (S .- root .* Ired) / (S .- pole .* Ired)
        end,
        roots=(center, radius) -> scalar_roots_in_contour(ComplexF64[root], center, radius),
    )
end

function scalar_exponential_shift_case(; alpha, tau=1.0, name=nothing)
    alpha = ComplexF64(alpha)
    tau = Float64(tau)
    base = log(alpha) / tau
    period = 2pi / tau
    label = name === nothing ? "exp_shift_$(alpha)_tau_$(tau)" : String(name)
    (
        name=label,
        f=z -> exp(tau * z) - alpha,
        df=z -> tau * exp(tau * z),
        fmat=S -> exp(tau * S) .- alpha .* Matrix{ComplexF64}(I, size(S, 1), size(S, 2)),
        roots=(center, radius) -> begin
            lower = (imag(center) - radius - imag(base)) / period
            upper = (imag(center) + radius - imag(base)) / period
            candidates = ComplexF64[base + im * period * k for k in floor(Int, lower)-2:ceil(Int, upper)+2]
            scalar_roots_in_contour(candidates, center, radius)
        end,
    )
end

function exponential_many_root_cases()
    (
        scalar_exponential_shift_case(; alpha=1.0, name="exp1"),
        scalar_exponential_shift_case(; alpha=0.7, name="exp07"),
        scalar_exponential_shift_case(; alpha=1.4, name="exp14"),
        scalar_exponential_shift_case(; alpha=1.0 + 0.35im, name="expc1"),
        scalar_exponential_shift_case(; alpha=0.9 - 0.25im, name="expc2"),
        scalar_exponential_shift_case(; alpha=1.2 + 0.15im, name="expc3"),
    )
end

function near_multiple_polynomial_problem()
    n = 6
    center = 0.0 + 0.0im
    radius = 1.0
    δ = 2e-4
    inside_roots = [
        ComplexF64[-0.55 - δ, -0.55 + δ, 0.15 + 0.08im],
        ComplexF64[-0.35 - δ, -0.35 + δ, 0.25 - 0.06im],
        ComplexF64[-0.15 - δ, -0.15 + δ, 0.35 + 0.04im],
        ComplexF64[0.05 - δ, 0.05 + δ, 0.45 - 0.03im],
        ComplexF64[0.25 - δ, 0.25 + δ, 0.55 + 0.02im],
        ComplexF64[0.45 - δ, 0.45 + δ, 0.65 - 0.01im],
    ]
    outside_roots = [
        ComplexF64[2.0 + 0.1im * i, 2.8 + 0.1im * i] for i in 1:n
    ]
    roots_by_direction = [vcat(inside_roots[i], outside_roots[i]) for i in 1:n]
    scalar_coeffs = [polynomial_coefficients_from_roots(roots) for roots in roots_by_direction]
    degree = length(roots_by_direction[1])

    V = ComplexF64[
        1.0 0.90 0.80 0.70 0.60 0.50
        0.0 1.00 0.85 0.75 0.65 0.55
        0.0 0.10 1.00 0.80 0.70 0.60
        0.0 0.00 0.20 1.00 0.75 0.65
        0.0 0.00 0.00 0.30 1.00 0.80
        0.0 0.00 0.00 0.00 0.40 1.00
    ]
    Vinv = inv(V)
    coeffs = Matrix{ComplexF64}[]
    for j in 1:(degree + 1)
        push!(coeffs, V * Diagonal([scalar_coeffs[i][j] for i in 1:n]) * Vinv)
    end
    expected = sort(vcat(inside_roots...); by=z -> (real(z), imag(z)))
    coeffs, center, radius, n, expected
end

function regular_quadratic_polynomial_problem()
    n = 8
    center = 0.0 + 0.0im
    radius = 1.25
    inside_roots = ComplexF64[
        -0.95 + 0.05im,
        -0.70 - 0.05im,
        -0.42 + 0.10im,
        -0.15 - 0.10im,
        0.12 + 0.08im,
        0.38 - 0.08im,
        0.66 + 0.04im,
        0.92 - 0.04im,
    ]
    outside_roots = ComplexF64[2.5 + 0.1im * i for i in 1:n]
    A0 = Diagonal(inside_roots .* outside_roots)
    A1 = Diagonal(-(inside_roots .+ outside_roots))
    A2 = Matrix{ComplexF64}(I, n, n)
    V = ComplexF64[exp(0.07im * i * j) / (1 + abs(i - j)) for i in 1:n, j in 1:n]
    Vinv = inv(V)
    coeffs = [V * Matrix(A0) * Vinv, V * Matrix(A1) * Vinv, A2]
    coeffs, center, radius, n, inside_roots
end

function run_moment_rii_experiment_matrix(; print_rows=true)
    rows = NamedTuple[]
    append!(
        rows,
        run_matrix_polynomial_case(;
            name="regular_quadratic_n8",
            problem_class=:regular_polynomial,
            make_problem=regular_quadratic_polynomial_problem,
            basis_moments=2,
            basis_nodes=24,
            rii_nodes=48,
            update_moment_count=1,
            basis_ranktol=1e-10,
            residual_tol=1e-9,
        ),
    )
    append!(
        rows,
        run_matrix_polynomial_case(;
            name="deficient_quadratic",
            problem_class=:deficient_polynomial,
            make_problem=deficient_quadratic_problem,
            basis_moments=2,
            basis_nodes=32,
            rii_nodes=64,
            update_moment_count=1,
            basis_ranktol=1e-9,
            residual_tol=1e-8,
        ),
    )
    append!(
        rows,
        run_matrix_polynomial_case(;
            name="many_root_nonnormal_poly",
            problem_class=:many_root_polynomial,
            make_problem=many_eigenvalue_nonnormal_polynomial_problem,
            basis_moments=5,
            basis_nodes=48,
            rii_nodes=96,
            update_moment_count=1,
            basis_ranktol=1e-10,
            residual_tol=1e-7,
        ),
    )
    append!(
        rows,
        run_matrix_polynomial_case(;
            name="near_multiple_cluster_poly",
            problem_class=:clustered_polynomial,
            make_problem=near_multiple_polynomial_problem,
            basis_moments=5,
            basis_nodes=64,
            rii_nodes=96,
            update_moment_count=1,
            basis_ranktol=1e-10,
            residual_tol=1e-7,
            match_atol=1e-5,
        ),
    )
    append!(
        rows,
        run_matrix_analytic_case(;
            name="many_root_scalar_analytic",
            problem_class=:many_root_analytic,
            cases=(scalar_sine_case(), scalar_cosine_case(), scalar_shifted_sine_case()),
            radius=20.0,
            iterations=1,
            basis_moments=4,
            basis_nodes=8,
            rii_nodes=128,
            basis_ranktol=0.5,
            determinant_nodes=512,
            reduced_moments=16,
            reduced_nodes=512,
            component_scaling=:none,
        ),
    )
    append!(
        rows,
        run_matrix_analytic_case(;
            name="near_pole_rational",
            problem_class=:rational_nep,
            cases=(
                scalar_rational_case(; root=-0.45, pole=1.18, name="rat1"),
                scalar_rational_case(; root=-0.10 + 0.20im, pole=1.22 + 0.15im, name="rat2"),
                scalar_rational_case(; root=0.20 - 0.15im, pole=1.25 - 0.10im, name="rat3"),
                scalar_rational_case(; root=0.48 + 0.10im, pole=1.28 + 0.05im, name="rat4"),
                scalar_rational_case(; root=0.68 - 0.05im, pole=1.30 - 0.08im, name="rat5"),
            ),
            radius=1.0,
            iterations=1,
            basis_moments=3,
            basis_nodes=24,
            rii_nodes=96,
            basis_ranktol=1e-10,
            determinant_nodes=512,
            determinant_capacity=32,
            reduced_moments=8,
            reduced_nodes=512,
            component_scaling=:contour_max,
        ),
    )
    append!(
        rows,
        run_matrix_analytic_case(;
            name="exponential_many_root",
            problem_class=:exponential_nep,
            cases=exponential_many_root_cases(),
            radius=10.0,
            iterations=1,
            basis_moments=5,
            basis_nodes=24,
            rii_nodes=128,
            basis_ranktol=1e-8,
            determinant_nodes=768,
            determinant_capacity=96,
            extractor=:loewner_counted,
            loewner_points=6,
            loewner_radius=1.3,
            reduced_moments=16,
            reduced_nodes=768,
            component_scaling=:none,
            residual_normalization=:vector,
        ),
    )
    append!(
        rows,
        run_matrix_analytic_case(;
            name="exponential_ss_counted",
            problem_class=:extractor_sensitivity,
            cases=exponential_many_root_cases(),
            radius=10.0,
            iterations=1,
            basis_moments=5,
            basis_nodes=24,
            rii_nodes=128,
            basis_ranktol=1e-8,
            determinant_nodes=768,
            determinant_capacity=96,
            extractor=:ss_counted,
            reduced_moments=16,
            reduced_nodes=768,
            component_scaling=:none,
            residual_normalization=:vector,
        ),
    )
    append!(
        rows,
        run_matrix_analytic_case(;
            name="exponential_contour_scaled",
            problem_class=:scaling_sensitivity,
            cases=exponential_many_root_cases(),
            radius=10.0,
            iterations=0,
            basis_moments=5,
            basis_nodes=24,
            rii_nodes=128,
            basis_ranktol=1e-8,
            determinant_nodes=768,
            determinant_capacity=96,
            extractor=:loewner_counted,
            loewner_points=6,
            loewner_radius=1.3,
            reduced_moments=16,
            reduced_nodes=768,
            component_scaling=:contour_max,
            residual_normalization=:vector,
        ),
    )
    append!(
        rows,
        run_matrix_analytic_case(;
            name="multiple_root_sine2",
            problem_class=:multiple_root_analytic,
            cases=(scalar_squared_sine_case(),),
            radius=10.0,
            iterations=1,
            basis_moments=8,
            basis_nodes=48,
            rii_nodes=128,
            basis_ranktol=1e-10,
            determinant_nodes=512,
            determinant_capacity=64,
            reduced_moments=16,
            reduced_nodes=512,
            residual_normalization=:vector,
        ),
    )
    append!(rows, run_matrix_local_chart_case())
    print_rows && print_experiment_matrix(rows)
    rows
end
