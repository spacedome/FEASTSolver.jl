# Executable, literate entry points for the current moment-NLFEAST findings.
#
# These are intentionally small and narrative. The long exploratory drivers
# remain available, but new readers should start here or in `torture_matrix.jl`.

function showcase_linear_feast_packet_geometry(; print_rows=true)
    print_rows && println("\nShowcase: linear dual RII reduces to FEAST contour filtering")
    result = run_linear_dual_rii_reduction_diagnostic(; print_rows=print_rows)
    (
        conclusion=:packet_projector_is_riesz_oblique_projector_in_linear_dual_feast,
        max_projection_gap=result.max_projection_gap,
        rows=result.rows,
    )
end

function showcase_polynomial_lower_rung(; print_rows=true)
    print_rows && println("\nShowcase: polynomial-native moments agree with companion FEAST")
    result = run_near_multiple_polynomial_companion_torture(; print_rows=print_rows)
    (
        conclusion=:polynomial_moment_extraction_matches_companion_linearization,
        accepted=result.accepted,
        companion=result.companion,
        rows=result.rows,
    )
end

function showcase_adversarial_packets(; print_rows=true)
    print_rows && println("\nShowcase: multiplicity, weak support, and packet-level resolution")
    ids = (
        :defective_triangular_multiplicity,
        :meromorphic_pole_ladder,
        :clustered_simple_roots,
        :near_branch_fixed_sheet,
    )
    cases = Tuple(run_moment_rii_torture_case(id; print_rows=false) for id in ids)
    if print_rows
        @printf("  %-36s %-10s %-8s %s\n", "case", "status", "passed", "metrics")
        for case in cases
            @printf("  %-36s %-10s %-8s %s\n", string(case.id), string(case.status), string(case.passed), string(case.metrics))
        end
    end
    (
        conclusion=:packet_completion_is_not_always_value_level_resolution,
        cases=cases,
    )
end

function showcase_clustered_root_resolution_ladder(; print_rows=true)
    result = run_clustered_simple_roots_resolution_ladder(; print_rows=print_rows)
    (
        conclusion=:overlap_before_shrink_for_value_level_resolution,
        rows=result.rows,
        resolved_stage=result.resolved_stage,
        packet_stage=result.packet_stage,
    )
end

function showcase_schrodinger_packet_monitor(; print_rows=true)
    print_rows && println("\nShowcase: packet-visible defect monitor on Schrodinger/DD")
    result = run_fused_schrodinger_dd_packet_policy_diagnostic(; print_rows=print_rows)
    (
        conclusion=:packet_visible_defect_is_a_diagnostic_monitor_not_a_canonical_chart,
        selected_action=result.selected_action,
        selected_nodes=result.selected_nodes,
        rows=result.rows,
    )
end

function run_moment_rii_showcases(; print_rows=true)
    (
        linear=showcase_linear_feast_packet_geometry(; print_rows=print_rows),
        polynomial=showcase_polynomial_lower_rung(; print_rows=print_rows),
        adversarial=showcase_adversarial_packets(; print_rows=print_rows),
        clustered_resolution=showcase_clustered_root_resolution_ladder(; print_rows=print_rows),
    )
end
