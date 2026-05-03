# Experiment-layer pipeline objects for moment-NLFEAST prototypes.
#
# These are deliberately not public FEASTSolver APIs. They make the research
# stages explicit while the algorithm is still being validated in-tree.

Base.@kwdef struct ContourChart
    center::ComplexF64 = 0.0 + 0.0im
    radius::Float64 = 1.0
    coordinate::Symbol = :scaled
    component_scaling::Symbol = :none
    component_scaling_nodes::Int = 64
end

ContourChart(center, radius; kwargs...) = ContourChart(; center=ComplexF64(center), radius=Float64(radius), kwargs...)

function circular_rule(chart::ContourChart, nodes)
    circular_rule(chart.center, chart.radius, nodes)
end

Base.@kwdef struct TrialSpaces
    X::Matrix{ComplexF64}
    Y::Matrix{ComplexF64}
    right_singulars::Vector{Float64} = Float64[]
    left_singulars::Vector{Float64} = Float64[]
    source::Symbol = :unknown
end

Base.@kwdef struct MomentBasisConfig
    moments::Int = 4
    nodes::Int = 8
    ranktol::Float64 = 0.5
    seed::Union{Nothing, Int} = nothing
    biorthogonalize::Bool = false
end

function common_square_trial_spaces(trial::TrialSpaces)
    common = min(size(trial.X, 2), size(trial.Y, 2))
    TrialSpaces(
        X=trial.X[:, 1:common],
        Y=trial.Y[:, 1:common],
        right_singulars=trial.right_singulars,
        left_singulars=trial.left_singulars,
        source=trial.source,
    )
end

function trial_space_summary(trial::TrialSpaces)
    right_ratio = isempty(trial.right_singulars) ? NaN : trial.right_singulars[end] / trial.right_singulars[1]
    left_ratio = isempty(trial.left_singulars) ? NaN : trial.left_singulars[end] / trial.left_singulars[1]
    (right_cols=size(trial.X, 2), left_cols=size(trial.Y, 2), right_sigma=right_ratio, left_sigma=left_ratio)
end

Base.@kwdef struct ReducedExtractorConfig
    extractor::Symbol = :ss_counted
    determinant_nodes::Int = 2048
    determinant_capacity::Int = 96
    reduced_moments::Int = 8
    reduced_nodes::Int = 256
    reduced_ranktol::Float64 = 1e-10
    reduced_maxrank::Int = typemax(Int)
    reduced_ss_mode::Symbol = :similarity
    loewner_points::Int = 4
    loewner_radius::Float64 = 1.6
    loewner_phase::Float64 = 0.0
    residual_normalization::Symbol = :operator
    refinement::Symbol = :none
    refinement_steps::Int = 4
    refinement_nodes::Int = 256
end

Base.@kwdef struct ResidualUpdateConfig
    moment_count::Int = 1
    rii_nodes::Int = 256
    residual_ranktol::Float64 = 1e-10
    compression_ranktol::Float64 = 1e-10
    mode::Symbol = :moment_compressed
    biorthogonalize::Bool = false
end

function analytic_context(cases, chart::ContourChart, operator_builder)
    component_scales = analytic_component_scales(
        cases,
        chart.center,
        chart.radius;
        mode=chart.component_scaling,
        nodes=chart.component_scaling_nodes,
    )
    Tmatrix, Tderivative, Tsolve, Tadjoint_solve, expected_roots =
        operator_builder(cases; component_scales=component_scales)
    (
        Tmatrix=Tmatrix,
        Tderivative=Tderivative,
        Tsolve=Tsolve,
        Tadjoint_solve=Tadjoint_solve,
        expected=expected_roots(chart.center, chart.radius),
        n=length(cases),
        component_scales=component_scales,
    )
end

function initial_dual_trial_spaces(
    ctx,
    chart::ContourChart;
    basis_moments=4,
    basis_nodes=8,
    basis_ranktol=0.5,
    seed=nothing,
)
    seed === nothing || Random.seed!(seed)
    z_nodes, z_weights = circular_rule(chart, basis_nodes)
    Xprobe = rand(ComplexF64, ctx.n, ctx.n)
    Wprobe = rand(ComplexF64, ctx.n, ctx.n)
    right_moments = initial_moments_generic_scaled(ctx.Tsolve, Xprobe, z_nodes, z_weights, chart.center, chart.radius, basis_moments)
    left_moments = initial_adjoint_moments_generic_scaled(ctx.Tadjoint_solve, Wprobe, z_nodes, z_weights, chart.center, chart.radius, basis_moments)
    Xbasis, right_singulars = moment_block_basis(right_moments, basis_moments; ranktol=basis_ranktol)
    Ybasis, left_singulars = moment_block_basis(left_moments, basis_moments; ranktol=basis_ranktol)
    common_square_trial_spaces(TrialSpaces(
        X=Xbasis,
        Y=Ybasis,
        right_singulars=right_singulars,
        left_singulars=left_singulars,
        source=:initial_contour_moments,
    ))
end

function initial_dual_trial_spaces(ctx, chart::ContourChart, config::MomentBasisConfig)
    trial = initial_dual_trial_spaces(
        ctx,
        chart;
        basis_moments=config.moments,
        basis_nodes=config.nodes,
        basis_ranktol=config.ranktol,
        seed=config.seed,
    )
    if config.biorthogonalize
        Xbi, Ybi, cross_singulars = biorthogonalize_bases(trial.X, trial.Y)
        return TrialSpaces(
            X=Xbi,
            Y=Ybi,
            right_singulars=Float64.(cross_singulars),
            left_singulars=Float64.(cross_singulars),
            source=:biorthogonalized_initial_contour_moments,
        )
    end
    trial
end

function extract_reduced_nep(ctx, trial::TrialSpaces, chart::ContourChart, config::ReducedExtractorConfig)
    reduced_analytic_extraction(
        ctx.Tmatrix,
        ctx.Tderivative,
        trial.X,
        trial.Y,
        chart.center,
        chart.radius;
        extractor=config.extractor,
        determinant_nodes=config.determinant_nodes,
        determinant_capacity=config.determinant_capacity,
        reduced_moments=config.reduced_moments,
        reduced_nodes=config.reduced_nodes,
        reduced_ranktol=config.reduced_ranktol,
        reduced_maxrank=config.reduced_maxrank,
        reduced_ss_mode=config.reduced_ss_mode,
        loewner_points=config.loewner_points,
        loewner_radius=config.loewner_radius,
        loewner_phase=config.loewner_phase,
        residual_normalization=config.residual_normalization,
        refinement=config.refinement,
        refinement_steps=config.refinement_steps,
        refinement_nodes=config.refinement_nodes,
    )
end

function residual_laurent_update(ctx, trial::TrialSpaces, extraction, chart::ContourChart, config::ResidualUpdateConfig)
    config.mode === :moment_compressed || error("unsupported residual update mode: $(config.mode)")
    rii_z_nodes, rii_z_weights = circular_rule(chart, config.rii_nodes)
    Xnew, Ynew, stats = moment_compressed_dual_rii_bases_generic(
        ctx.Tsolve,
        ctx.Tadjoint_solve,
        ctx.Tmatrix,
        trial.X,
        trial.Y,
        extraction,
        rii_z_nodes,
        rii_z_weights,
        chart.center,
        chart.radius;
        moment_count=config.moment_count,
        residual_ranktol=config.residual_ranktol,
        compression_ranktol=config.compression_ranktol,
    )
    updated = common_square_trial_spaces(TrialSpaces(
        X=Xnew,
        Y=Ynew,
        right_singulars=Float64.(stats.right_singulars),
        left_singulars=Float64.(stats.left_singulars),
        source=:residual_laurent_update,
    ))
    if config.biorthogonalize
        Xbi, Ybi, cross_singulars = biorthogonalize_bases(updated.X, updated.Y)
        updated = TrialSpaces(
            X=Xbi,
            Y=Ybi,
            right_singulars=Float64.(cross_singulars),
            left_singulars=Float64.(cross_singulars),
            source=:biorthogonalized_residual_laurent_update,
        )
    end
    updated, stats
end

function good_extraction_values(extraction; residual_tol)
    extraction === nothing && return ComplexF64[]
    good = extraction.inside .& (extraction.residuals .<= residual_tol)
    ComplexF64.(extraction.values[good])
end

function good_extraction_entries(extraction, chart_center, chart_radius; residual_tol)
    extraction === nothing && return NamedTuple[]
    good = extraction.inside .& (extraction.residuals .<= residual_tol)
    [
        (
            value=ComplexF64(extraction.values[j]),
            residual=Float64(extraction.residuals[j]),
            right_residual=Float64(extraction.right_residuals[j]),
            left_residual=Float64(extraction.left_residuals[j]),
            center=ComplexF64(chart_center),
            radius=Float64(chart_radius),
        )
        for j in eachindex(extraction.values) if good[j]
    ]
end

function chart_entry_clusters(entries; atol=1e-6)
    clusters = Vector{Vector{Any}}()
    for entry in entries
        index = findfirst(cluster -> minimum(abs.(entry.value .- [item.value for item in cluster])) <= atol, clusters)
        if index === nothing
            push!(clusters, Any[entry])
        else
            push!(clusters[index], entry)
        end
    end
    map(clusters) do cluster
        best_index = argmin([entry.residual for entry in cluster])
        centers = unique_values([entry.center for entry in cluster]; atol=atol)
        (
            value=cluster[best_index].value,
            support=length(centers),
            observations=length(cluster),
            best_residual=cluster[best_index].residual,
        )
    end
end

function supported_cluster_values(clusters; min_support=2)
    ComplexF64[cluster.value for cluster in clusters if cluster.support >= min_support]
end

function globally_supported_cluster_values(clusters, center, radius; min_support=2, boundary_margin=0.0)
    ComplexF64[
        cluster.value for cluster in clusters
        if cluster.support >= min_support && abs(cluster.value - center) <= radius + boundary_margin
    ]
end

function support_sweep_counts(result; max_support=5, atol=1e-6)
    [
        (
            support=support,
            count=length(supported_cluster_values(result.support_clusters; min_support=support)),
            matched=match_expected_count(
                supported_cluster_values(result.support_clusters; min_support=support),
                result.expected;
                atol=atol,
            ),
        )
        for support in 1:max_support
    ]
end
