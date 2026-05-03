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
    rii_nodes=128,
    basis_ranktol=1e-8,
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
            basis_ranktol=basis_ranktol,
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
)
    final = result.refined
    expected = final.expected
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
    (
        expected=length(expected),
        support1_global=length(support1_global),
        support1_global_matched=match_expected_count(support1_global, expected; atol=match_atol),
        support2_global=length(support2_global),
        support2_global_matched=match_expected_count(support2_global, expected; atol=match_atol),
        support3_global=length(support3_global),
        support3_global_matched=match_expected_count(support3_global, expected; atol=match_atol),
        weak_inside_clusters=count(cluster -> cluster.support < 2, inside_clusters),
        inside_cluster_count=length(inside_clusters),
        usable_records=length(usable_records),
        good_records=length(good_records),
        count_error_bad=count(record -> record.count_error > count_error_tol, good_records),
        count_deficit_records=count(record -> record.good < record.count_estimate, good_records),
        max_count_error=isempty(good_records) ? Inf : maximum(record.count_error for record in good_records),
        max_record_residual=isempty(good_records) ? Inf : maximum(record.max_residual for record in good_records),
        exact_support2_global=length(support2_global) == length(expected) &&
            match_expected_count(support2_global, expected; atol=match_atol) == length(expected),
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
    summary = adaptive_retention_score_summary(
        result;
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
            "  expected=%d support1_global=%d/%d support2_global=%d/%d support3_global=%d/%d weak_inside=%d\n",
            summary.expected,
            summary.support1_global_matched,
            summary.support1_global,
            summary.support2_global_matched,
            summary.support2_global,
            summary.support3_global_matched,
            summary.support3_global,
            summary.weak_inside_clusters,
        )
        @printf(
            "  local records usable=%d good=%d count_error_bad=%d count_deficit=%d max_count_error=%.3e max_residual=%.3e status=%s\n",
            summary.usable_records,
            summary.good_records,
            summary.count_error_bad,
            summary.count_deficit_records,
            summary.max_count_error,
            summary.max_record_residual,
            summary.exact_support2_global ? "ok" : "check",
        )
        println("  evidence=", evidence)
    end
    (
        result=result,
        summary=summary,
        evidence=evidence,
        layout_agreement=layout_agreement,
        extractor_agreement=extractor_agreement,
    )
end

function retention_policy_decision(
    summary;
    residual_tol=1e-8,
    layout_agreement=missing,
    extractor_agreement=missing,
)
    actions = Symbol[]
    if summary.exact_support2_global
        push!(actions, :retain_support2_global)
    else
        push!(actions, :refine_weak_target_support)
    end
    local_count_warning = summary.count_deficit_records > 0 || summary.count_error_bad > 0
    if local_count_warning
        push!(actions, :treat_local_count_errors_as_chart_warnings)
    end
    if summary.max_record_residual > 10 * residual_tol
        push!(actions, :tighten_or_refine_high_residual_charts)
    end
    if layout_agreement === missing
        if local_count_warning || !summary.exact_support2_global
            push!(actions, :request_loewner_layout_agreement)
        end
    elseif !layout_agreement
        push!(actions, :reject_or_split_layout_unstable_candidates)
    else
        push!(actions, :layout_agreement_certified)
    end
    if extractor_agreement === missing
        if local_count_warning || !summary.exact_support2_global
            push!(actions, :request_reduced_extractor_agreement)
        end
    elseif !extractor_agreement
        push!(actions, :escalate_reduced_extractor_disagreement)
    else
        push!(actions, :extractor_agreement_certified)
    end

    residual_ok = summary.max_record_residual <= 10 * residual_tol
    support_ok = summary.exact_support2_global
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
        expected=summary.expected,
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
