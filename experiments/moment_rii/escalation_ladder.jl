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

function run_moment_rii_escalation_ladder(; include_multiplicity_smoke=false, print_rows=true)
    policy = moment_rii_escalation_ladder_policy()
    clustered = run_clustered_simple_roots_resolution_ladder(; print_rows=false)
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

