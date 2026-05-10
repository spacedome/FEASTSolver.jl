# Compact, executable-facing summary of the adjacent Lean theory attempt.
#
# The Lean repository is useful as a contract language for local diagnostics.
# It does not currently prove global convergence of moment-NLFEAST, and it
# argues against treating the Hankel packet coordinates as canonical geometry.

function lean_theory_handoff_summary(; print_rows=true)
    summary = (
        source="/home/julien/Code/LEAN_FEAST",
        primary_boundary=:(Corr_Rlin_equals_Ppacket_plus_E),
        contract="Corr ∘ Rlin = P_packet + E",
        interpretation=:visible_defect_contract,
        useful_for=(
            :packet_visible_failure_diagnostics,
            :local_contraction_logs,
            :residual_budget_splitting,
            :polynomial_invariant_pair_lower_rung,
        ),
        not_yet=(
            :global_convergence_proof,
            :canonical_hankel_chart,
            :floating_point_correctness_proof,
        ),
        linear_lower_rung=(
            standard=:riesz_spectral_projector,
            dual=:B_oblique_left_right_riesz_projector,
            packet=:diagnostic_visible_projection,
        ),
        next_theory_target=:external_visible_projector_plus_range_local_defect_propagator,
    )

    if print_rows
        println()
        println("Lean handoff summary")
        println("  contract: ", summary.contract)
        println("  interpretation: packet-visible defect diagnostic, not canonical Hankel coordinates")
        println("  linear rung: standard Riesz projector; dual B-oblique Riesz projector")
        println("  next target: external visible projector + range-local defect propagator")
    end
    summary
end

