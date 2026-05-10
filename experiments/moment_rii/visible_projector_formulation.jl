# Visible-projector formulation for the former "packet" diagnostics.
#
# The working conclusion is deliberately conservative: "packet" is useful as a
# spectral-cluster word, but the algorithmic object we control is a visible
# model-space projector plus a defect/correction budget.

function visible_projector_research_axes()
    (
        (
            axis=:projector_first,
            object=:external_visible_projector,
            claim=:control_projected_defect_not_hankel_coordinates,
            evidence=:linear_riesz_and_dual_oblique_riesz,
        ),
        (
            axis=:polynomial_lower_rung,
            object=:companion_consistent_polynomial_realization,
            claim=:polynomial_moments_must_match_companion_linearization,
            evidence=:polynomial_bridge,
        ),
        (
            axis=:invariant_pair_residual_geometry,
            object=:finite_realization_residual,
            claim=:moments_represent_an_invariant_pair_state_before_scalar_roots,
            evidence=:block_newton_and_residual_laurent_rows,
        ),
        (
            axis=:dual_formulation,
            object=:trial_test_visible_spaces,
            claim=:nonnormal_visibility_is_two_sided,
            evidence=:B_oblique_left_right_riesz_projector,
        ),
        (
            axis=:failure_layer_diagnostics,
            object=:layered_visible_defect_report,
            claim=:steering_depends_on_failed_layer_not_single_canonical_chart,
            evidence=:torture_failure_layer_reports,
        ),
        (
            axis=:feshbach_grushin_effective_operator,
            object=:model_space_plus_complement_elimination,
            claim=:use_as_theory_analogy_for_visible_defect_contract_not_as_new_api,
            evidence=:lean_visible_projector_handoff,
        ),
    )
end

function classify_visible_projector_failure_layers()
    reports = moment_rii_failure_layer_reports()
    Tuple(
        (
            id=report.id,
            layer=report.layer,
            diagnostic=report.diagnostic,
            steering=report.steering,
            fundamental=report.fundamental,
        ) for report in reports
    )
end

function polynomial_invariant_pair_visible_summary(polynomial)
    rows = polynomial.polynomial_rows
    stages = Tuple(row.stage for row in rows)
    successful = all(row.success for row in rows)
    max_residual = maximum(row.max_residual for row in rows)
    (
        stages=stages,
        all_successful=successful,
        max_residual=max_residual,
        companion_matched=polynomial.companion.matched,
        companion_expected=polynomial.companion.expected,
        interpretation=:finite_realization_residual_is_the_polynomial_visible_defect,
    )
end

function visible_projector_effective_operator_diagnostic(;
    center=1.0 + 0.0im,
    radius=0.62,
    z=center + 0.31im,
    perturbation=1e-3,
    seed=20260510,
)
    Random.seed!(seed)
    n = 9
    λ = ComplexF64[-1.2, -0.6, 0.25, 0.72, 1.02, 1.33, 1.78, 2.3, 3.0]
    V = ComplexF64[exp(0.11im * i * j) / (1 + abs(i - j)) for i in 1:n, j in 1:n]
    A = V * Diagonal(λ) * inv(V)
    P, X, Y, values = linear_standard_packet_projector(A, center, radius)
    Iₙ = Matrix{ComplexF64}(I, n, n)
    T = z .* Iₙ .- A
    Q = Iₙ - P
    exact_visible_to_complement = norm(P * T * Q)
    exact_complement_to_visible = norm(Q * T * P)
    exact_schur_correction = norm((P * T * Q) * pinv(Q * T * Q) * (Q * T * P))

    Xperturbed = X .+ perturbation .* randn(ComplexF64, size(X))
    Pperturbed = oblique_packet_projector(Xperturbed, Y; ranktol=1e-12)
    Qperturbed = Iₙ - Pperturbed
    perturbed_visible_to_complement = norm(Pperturbed * T * Qperturbed)
    perturbed_complement_to_visible = norm(Qperturbed * T * Pperturbed)
    perturbed_schur_correction = norm((Pperturbed * T * Qperturbed) * pinv(Qperturbed * T * Qperturbed) * (Qperturbed * T * Pperturbed))

    (
        z=z,
        selected=length(values),
        exact_coupling=max(exact_visible_to_complement, exact_complement_to_visible),
        exact_schur_correction=exact_schur_correction,
        perturbed_coupling=max(perturbed_visible_to_complement, perturbed_complement_to_visible),
        perturbed_schur_correction=perturbed_schur_correction,
        interpretation=:effective_operator_coupling_measures_visible_projector_defect,
    )
end

function run_visible_projector_formulation_diagnostic(;
    include_polynomial=true,
    print_rows=true,
)
    handoff = lean_theory_handoff_summary(; print_rows=false)
    linear = linear_packet_projector_diagnostic(; print_rows=false)
    linear_leakage = run_linear_visible_projector_leakage_diagnostic(; print_rows=false)
    axes = visible_projector_research_axes()
    vocabulary = visible_projector_vocabulary()
    failure_layers = classify_visible_projector_failure_layers()
    polynomial = include_polynomial ? run_polynomial_family_bridge_diagnostic(; print_rows=false) : nothing
    polynomial_summary = polynomial === nothing ? nothing : polynomial_invariant_pair_visible_summary(polynomial)
    dual_summary = (
        projector=:B_oblique_left_right_riesz_projector,
        right_filter_gap=linear.dual.right_filter_gap,
        left_filter_gap=linear.dual.left_filter_gap,
        monitor_gap=linear.dual.monitor_gap,
    )
    projector_summary = (
        standard=:riesz_spectral_projector,
        standard_filter_gap=linear.standard.filter_gap,
        standard_monitor_gap=linear.standard.monitor_gap,
        dual=dual_summary.projector,
        dual_filter_gap=max(dual_summary.right_filter_gap, dual_summary.left_filter_gap),
        dual_monitor_gap=dual_summary.monitor_gap,
    )
    effective_operator = (
        analogy=:feshbach_grushin_effective_operator,
        model_space=:visible_projector_range,
        complement=:invisible_or_eliminated_defect,
        reduced_object=:effective_residual_or_reduced_operator,
        diagnostic=visible_projector_effective_operator_diagnostic(),
        status=:theory_analogy_not_canonical_packet_api,
    )

    result = (
        conclusion=:visible_projector_not_canonical_packet_coordinates,
        vocabulary=vocabulary,
        axes=axes,
        lean_contract=handoff.contract,
        projector=projector_summary,
        linear_leakage=linear_leakage,
        polynomial=polynomial_summary,
        dual=dual_summary,
        failure_layers=failure_layers,
        effective_operator=effective_operator,
    )

    if print_rows
        println()
        println("Visible-projector formulation diagnostic")
        println("  legacy word: $(vocabulary.legacy_name); preferred algorithmic object: $(vocabulary.preferred_algorithmic_name)")
        println("  contract: $(result.lean_contract)")
        @printf(
            "  linear projector gaps: standard=%.3e dual=%.3e; one-sided leakage floor=%.3e\n",
            projector_summary.standard_filter_gap,
            projector_summary.dual_filter_gap,
            linear_leakage.one_sided_minimum,
        )
        if polynomial_summary !== nothing
            @printf(
                "  polynomial lower rung: companion=%d/%d stages=%s max_residual=%.3e\n",
                polynomial_summary.companion_matched,
                polynomial_summary.companion_expected,
                string(polynomial_summary.stages),
                polynomial_summary.max_residual,
            )
        end
        println("  failure layers: ", join(string.(unique(layer.layer for layer in failure_layers)), ", "))
        @printf(
            "  effective-operator angle: exact_coupling=%.3e perturbed_coupling=%.3e\n",
            effective_operator.diagnostic.exact_coupling,
            effective_operator.diagnostic.perturbed_coupling,
        )
    end

    result
end
