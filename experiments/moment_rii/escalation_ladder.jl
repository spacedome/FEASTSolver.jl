# Escalation-ladder entry points.
#
# These functions collect the current "what to try next" rules into executable
# examples.  They are intentionally policy-level: the heavy numerical cases
# still live in `torture_matrix.jl`, while this file states how those cases
# inform automatic steering.

function moment_rii_escalation_ladder_policy()
    (
        (:contour_count, :increase_quadrature_nodes, "first fix unreliable argument-principle counts"),
        (:moment_order, :increase_positive_moments, "needed when algebraic multiplicity exceeds represented moments"),
        (:local_overlap, :increase_chart_overlap, "preferred for clustered simple roots before shrinking charts"),
        (:local_radius, :shrink_local_charts, "only after overlap is adequate and value-level resolution is required"),
        (:packet_defect, :refine_extraction_or_cleanup, "packet-visible residual defects are algorithm-layer failures"),
        (:target_packet, :do_not_move_outer_contour_automatically, "changing the outer contour changes the problem"),
    )
end

function print_moment_rii_escalation_policy(policy=moment_rii_escalation_ladder_policy(); io=stdout)
    println(io, "Moment-RII escalation policy")
    for (layer, action, reason) in policy
        println(io, "  $(layer): $(action) -- $(reason)")
    end
    nothing
end

diagnostic_get(diagnostic, name::Symbol, default) =
    hasproperty(diagnostic, name) ? getproperty(diagnostic, name) : default

function moment_rii_escalation_decision(diagnostic; count_error_tol=1e-2, visible_tol=1e-8)
    if diagnostic_get(diagnostic, :target_packet_changed, false) ||
       diagnostic_get(diagnostic, :near_pole_or_branch_cut, false) ||
       diagnostic_get(diagnostic, :contour_pathology, false)
        return (
            layer=:target_packet,
            action=:declare_contour_pathology,
            update_stage=:stop_or_request_outer_contour_change,
            reason=:outer_contour_changes_problem_or_crosses_singularity,
            automatic_parameter_change=false,
        )
    end

    count_reliable = diagnostic_get(diagnostic, :count_reliable, true)
    count_error = diagnostic_get(diagnostic, :count_error, 0.0)
    if !count_reliable || count_error > count_error_tol
        return (
            layer=:contour_count,
            action=:increase_quadrature_nodes,
            update_stage=:recompute_count_before_solver_changes,
            reason=:argument_principle_count_not_reliable,
            automatic_parameter_change=true,
        )
    end

    moment_adequate = diagnostic_get(diagnostic, :moment_adequate, true)
    if !moment_adequate
        return (
            layer=:moment_order,
            action=:increase_positive_moments,
            update_stage=:rebuild_moment_realization,
            reason=:represented_moments_do_not_cover_algebraic_multiplicity,
            automatic_parameter_change=true,
        )
    end

    overlap_ok = diagnostic_get(diagnostic, :local_overlap_ok, true)
    if !overlap_ok
        return (
            layer=:local_overlap,
            action=:increase_chart_overlap,
            update_stage=:refine_local_chart_cover,
            reason=:clustered_roots_need_more_chart_support_before_radius_changes,
            automatic_parameter_change=true,
        )
    end

    visible_defect = diagnostic_get(diagnostic, :visible_defect, 0.0)
    extraction_agreement = diagnostic_get(diagnostic, :extraction_agreement, true)
    if visible_defect > visible_tol
        action = extraction_agreement ? :refine_extraction_or_cleanup : :change_extraction
        return (
            layer=:packet_defect,
            action=action,
            update_stage=:repair_visible_projector_defect,
            reason=:packet_visible_residual_defect,
            automatic_parameter_change=true,
        )
    end

    if !extraction_agreement
        return (
            layer=:extraction,
            action=:change_extraction,
            update_stage=:compare_reduced_extraction_models,
            reason=:extractors_disagree_without_visible_defect_certificate,
            automatic_parameter_change=true,
        )
    end

    (
        layer=:accepted,
        action=:accept,
        update_stage=:accept,
        reason=:diagnostics_clean,
        automatic_parameter_change=false,
    )
end

function run_moment_rii_escalation_ladder(; include_multiplicity_smoke=false, print_rows=true)
    policy = moment_rii_escalation_ladder_policy()
    clustered = run_clustered_simple_roots_resolution_ladder(; print_rows=false)
    decisions = (
        count=moment_rii_escalation_decision((count_reliable=false, count_error=0.2)),
        moments=moment_rii_escalation_decision((count_reliable=true, moment_adequate=false)),
        extraction=moment_rii_escalation_decision((visible_defect=1e-3, extraction_agreement=false)),
        contour=moment_rii_escalation_decision((near_pole_or_branch_cut=true,)),
        clean=moment_rii_escalation_decision((visible_defect=1e-12, extraction_agreement=true)),
    )
    multiplicity = include_multiplicity_smoke ?
        run_high_multiplicity_moment_order_sweep(;
            powers=(2,),
            moment_factors=(1,),
            outer_radius=4.0,
            print_rows=false,
        ) :
        nothing

    result = (
        conclusion=:escalate_by_failure_layer_not_by_one_canonical_chart,
        policy=policy,
        clustered=clustered,
        decisions=decisions,
        multiplicity=multiplicity,
        packet_stage=clustered.packet_stage,
        resolved_stage=clustered.resolved_stage,
        multiplicity_stage=include_multiplicity_smoke ? first(multiplicity.rows) : nothing,
    )

    if print_rows
        println()
        println("Moment-RII escalation ladder")
        println("  packet information is a visible-defect diagnostic; steering depends on the failed layer")
        print_moment_rii_escalation_policy(policy)
        println()
        println("  executable decisions:")
        for name in propertynames(decisions)
            decision = getproperty(decisions, name)
            println("    $(name): $(decision.action) ($(decision.layer))")
        end
        println()
        println("  clustered simple roots:")
        for row in clustered.rows
            status = row.value_resolved ? "value-resolved" : (row.packet_complete ? "packet-complete" : "undersampled")
            @printf(
                "    %-24s retained=%d algebraic=%d count=%d status=%s\n",
                string(row.stage),
                row.retained,
                row.algebraic_retained,
                row.count,
                status,
            )
        end
        if multiplicity !== nothing
            row = first(multiplicity.rows)
            @printf(
                "  multiplicity smoke: power=%d moments=%d status=%s recommendation=%s\n",
                row.power,
                row.moments,
                string(row.status),
                string(row.recommendation),
            )
        end
    end

    result
end
