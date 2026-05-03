# Experiment-layer chart policy objects for count-driven moment-NLFEAST.
#
# These are still research interfaces. They keep policy/numerics decisions out
# of individual diagnostic runners so tests can pin algorithmic choices instead
# of incidental keyword bundles.

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

function moment_basis_config(numerics::CountDrivenNumericsConfig; seed=nothing)
    MomentBasisConfig(;
        moments=numerics.basis_moments,
        nodes=numerics.basis_nodes,
        ranktol=numerics.basis_ranktol,
        seed=seed,
        biorthogonalize=numerics.biorthogonalize,
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

function count_stressed_chart_refinement(record, candidate_centers, child_centers; match_atol=1e-6)
    stress = record.good < record.count_estimate ? :count_deficit : :count_error
    refinement_centers = sorted_unique_values(
        vcat(candidate_centers, ComplexF64[record.center], child_centers);
        atol=match_atol,
    )
    if stress === :count_deficit
        candidate_radii = (record.radius / 4, 3 * record.radius / 8, 5 * record.radius / 8)
        action = :shrink_around_residual_candidates
        rationale = :missing_local_count
    else
        candidate_radii = (record.radius / 2, 2 * record.radius / 3, record.radius)
        action = :preserve_parent_radius_for_count_error
        rationale = :unstable_local_count
    end
    (
        stress=stress,
        refinement_centers=refinement_centers,
        candidate_radii=candidate_radii,
        action=action,
        rationale=rationale,
    )
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
