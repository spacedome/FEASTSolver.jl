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
