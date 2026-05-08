const MOMENT_RII_TORTURE_REQUIRED_FAILURE_CLASSES = (
    :high_count_low_dimension,
    :nonnormal_weak_support,
    :near_pole_meromorphic,
    :algebraic_multiplicity,
    :residual_laurent_update,
    :sparse_realistic_nep,
    :schrodinger_domain_decomposition,
    :branch_cut,
    :dense_spectral_region,
)

const MOMENT_RII_TORTURE_MATRIX = (
    (
        id=:low_dim_many_roots_delay,
        failure_class=:high_count_low_dimension,
        matrix_dimension=:low,
        spectral_difficulty=:many_roots,
        status=:covered,
        executable=true,
        smoke=false,
        runner=:run_multi_delay_count_driven_adaptive_refinement,
        evidence="slow test: count-driven refinement handles oracle-free nonnormal delay",
        expected_behavior="Oracle-free full-operator contour count drives local chart refinement until all counted roots are retained.",
        failure_mode="Moment extraction must not require one active vector per root in a small matrix NEP.",
    ),
    (
        id=:dense_multi_delay_weak_support,
        failure_class=:nonnormal_weak_support,
        matrix_dimension=:low,
        spectral_difficulty=:weak_left_right_support,
        status=:covered,
        executable=true,
        smoke=false,
        runner=:run_dense_multi_delay_weak_support_stress,
        evidence="slow test: count-driven refinement handles dense multi-delay weak support",
        expected_behavior="Weak-support charts are diagnosed and refined without accepting residual-small exterior candidates.",
        failure_mode="Nonnormal reduced extraction can produce residual-small values with poor two-sided support.",
    ),
    (
        id=:near_pole_rational,
        failure_class=:near_pole_meromorphic,
        matrix_dimension=:low,
        spectral_difficulty=:near_exterior_pole,
        status=:covered,
        executable=true,
        smoke=false,
        runner=:run_near_pole_rational_count_driven_adaptive_refinement,
        evidence="slow tests: oracle-free near-pole rational; near-pole count diagnostic rejects too-close poles",
        expected_behavior="Safe exterior poles are handled by count-driven refinement; too-close poles produce an unreliable-count diagnostic.",
        failure_mode="Meromorphic singularities near the contour can poison argument-principle counts and contour samples.",
    ),
    (
        id=:duplicate_delay_multiplicity,
        failure_class=:algebraic_multiplicity,
        matrix_dimension=:low,
        spectral_difficulty=:repeated_roots,
        status=:covered,
        executable=true,
        smoke=false,
        runner=:run_duplicate_delay_count_driven_adaptive_refinement,
        evidence="slow test: count-driven refinement handles oracle-free multiplicity",
        expected_behavior="Local contour counts assign multiplicity to retained unique roots so algebraic count closes.",
        failure_mode="A geometric root list can be incomplete even when every visible root has small residual.",
    ),
    (
        id=:squared_sine_multiplicity,
        failure_class=:algebraic_multiplicity,
        matrix_dimension=:scalar,
        spectral_difficulty=:high_root_multiplicity,
        status=:covered,
        executable=true,
        smoke=false,
        runner=:run_squared_sine_count_driven_adaptive_refinement,
        evidence="slow test: count-driven refinement handles repeated analytic roots",
        expected_behavior="Multiplicity probes close the algebraic count without duplicating scalar values.",
        failure_mode="Repeated analytic roots stress the distinction between unique values and algebraic count.",
    ),
    (
        id=:residual_laurent_correction_space,
        failure_class=:residual_laurent_update,
        matrix_dimension=:low,
        spectral_difficulty=:weak_initial_chart,
        status=:covered,
        executable=true,
        smoke=true,
        runner=:run_residual_laurent_correction_space_diagnostic,
        evidence="slow test: residual Laurent enrichment contains Ritz corrections",
        expected_behavior="New Ritz-vector components outside the old spaces lie in the residual-Laurent enrichment spaces.",
        failure_mode="The generalized moment update must repair physical spaces, not iterate expanded Hankel columns.",
    ),
    (
        id=:sparse_schrodinger_gallery,
        failure_class=:sparse_realistic_nep,
        matrix_dimension=:medium_sparse,
        spectral_difficulty=:realistic_sparse_polynomial_nep,
        status=:covered,
        executable=true,
        smoke=true,
        runner=:run_sparse_schrodinger_moment_gallery_smoke,
        evidence="slow test: sparse Schrodinger gallery count and residual repair",
        expected_behavior="The sparse moving-boundary Schrodinger gallery count is reliable and residual-Laurent repair accepts the target packet.",
        failure_mode="Sparse materialization/factorization paths must preserve the same contour and update geometry as dense controls.",
    ),
    (
        id=:schrodinger_dd_packet,
        failure_class=:schrodinger_domain_decomposition,
        matrix_dimension=:compressed_sparse_interface,
        spectral_difficulty=:packet_visible_defect,
        status=:covered,
        executable=true,
        smoke=true,
        runner=:run_fused_schrodinger_dd_packet_policy_diagnostic,
        evidence="slow test: fused Schrodinger DD packet policy selects convergence path",
        expected_behavior="Packet diagnostics distinguish contour-resolution defects from local cleanup defects.",
        failure_mode="A domain-decomposition Schur-complement NEP can look converged in reduced coordinates while the physical packet defect remains visible.",
    ),
    (
        id=:schrodinger_dd_fixed_target_guard,
        failure_class=:schrodinger_domain_decomposition,
        matrix_dimension=:compressed_sparse_interface,
        spectral_difficulty=:packet_changes_under_chart,
        status=:diagnostic_boundary,
        executable=true,
        smoke=false,
        runner=:run_fused_schrodinger_dd_packet_chart_ladder_diagnostic,
        evidence="slow test: fused Schrodinger DD chart ladder guards target packet",
        expected_behavior="Changing radius is recorded as changing the target packet, not as an automatic convergence fix.",
        failure_mode="Automatic contour changes can silently solve a different spectral selection problem.",
    ),
    (
        id=:branch_cut_operator,
        failure_class=:branch_cut,
        matrix_dimension=:unspecified,
        spectral_difficulty=:sheet_selection,
        status=:documented_gap,
        executable=false,
        smoke=false,
        runner=:none,
        evidence="documented gap",
        expected_behavior="No torture test should be accepted until the branch, sheet, and contour analytic domain are part of the model.",
        failure_mode="Operators with branch cuts violate the single-valued analytic assumptions unless the sheet model is explicit.",
    ),
    (
        id=:dense_spectral_region,
        failure_class=:dense_spectral_region,
        matrix_dimension=:problem_dependent,
        spectral_difficulty=:near_continuum_or_cluster,
        status=:degenerate_boundary,
        executable=false,
        smoke=false,
        runner=:none,
        evidence="documented boundary",
        expected_behavior="Treat as a diagnostic boundary unless the contour separates a finite packet with a stable count.",
        failure_mode="A dense spectral region can make the target packet ill-conditioned or semantically unstable.",
    ),
)

function moment_rii_torture_required_failure_classes()
    MOMENT_RII_TORTURE_REQUIRED_FAILURE_CLASSES
end

function moment_rii_torture_matrix(; include_documented_gaps=true)
    if include_documented_gaps
        return MOMENT_RII_TORTURE_MATRIX
    end
    Tuple(row for row in MOMENT_RII_TORTURE_MATRIX if row.executable)
end

function moment_rii_torture_entry(id::Symbol)
    for row in MOMENT_RII_TORTURE_MATRIX
        row.id === id && return row
    end
    error("unknown moment-RII torture case: $id")
end

function moment_rii_torture_smoke_cases()
    Tuple(row.id for row in MOMENT_RII_TORTURE_MATRIX if row.smoke)
end

function moment_rii_torture_coverage_summary(; include_documented_gaps=true)
    matrix = moment_rii_torture_matrix(; include_documented_gaps=include_documented_gaps)
    required = moment_rii_torture_required_failure_classes()
    covered = NamedTuple[]
    for failure_class in required
        rows = filter(row -> row.failure_class === failure_class, matrix)
        executable = any(row -> row.executable, rows)
        documented = !isempty(rows)
        push!(covered, (
            failure_class=failure_class,
            documented=documented,
            executable=executable,
            cases=Tuple(row.id for row in rows),
            statuses=Tuple(unique(row.status for row in rows)),
        ))
    end
    Tuple(covered)
end

function print_moment_rii_torture_matrix(; io=stdout, include_documented_gaps=true)
    println(io, "Moment-RII torture matrix")
    for row in moment_rii_torture_matrix(; include_documented_gaps=include_documented_gaps)
        println(io, "  $(row.id) [$(row.failure_class), $(row.status)]")
        println(io, "    runner: $(row.runner)")
        println(io, "    failure: $(row.failure_mode)")
        println(io, "    expected: $(row.expected_behavior)")
    end
    nothing
end

function run_moment_rii_torture_case(id::Symbol; print_rows=false)
    row = moment_rii_torture_entry(id)
    row.executable || return (
        id=row.id,
        status=row.status,
        passed=false,
        skipped=true,
        reason=row.failure_mode,
        metrics=NamedTuple(),
    )

    if row.id === :residual_laurent_correction_space
        result = run_residual_laurent_correction_space_diagnostic(; print_rows=print_rows)
        passed = result.updated.matched == result.expected &&
            result.updated.spurious_good == 0 &&
            result.max_gap <= 1e-12
        return (
            id=row.id,
            status=row.status,
            passed=passed,
            skipped=false,
            metrics=(
                expected=result.expected,
                initial_matched=result.initial.matched,
                updated_matched=result.updated.matched,
                max_gap=result.max_gap,
            ),
        )
    elseif row.id === :sparse_schrodinger_gallery
        result = run_sparse_schrodinger_moment_gallery_smoke(; residual_tol=1e-8, print_rows=print_rows)
        passed = result.target_count_reliable &&
            result.updated.good == result.target_count &&
            result.updated.max_inside_residual <= 1e-8
        return (
            id=row.id,
            status=row.status,
            passed=passed,
            skipped=false,
            metrics=(
                target_count=result.target_count,
                initial_good=result.initial.good,
                updated_good=result.updated.good,
                max_inside_residual=result.updated.max_inside_residual,
            ),
        )
    elseif row.id === :schrodinger_dd_packet
        result = run_fused_schrodinger_dd_packet_policy_diagnostic(; print_rows=print_rows)
        passed = result.expected == 13 &&
            result.selected_action === :accept &&
            last(result.rows).refined_matched == result.expected
        return (
            id=row.id,
            status=row.status,
            passed=passed,
            skipped=false,
            metrics=(
                expected=result.expected,
                selected_nodes=result.selected_nodes,
                selected_action=result.selected_action,
                final_matched=last(result.rows).refined_matched,
            ),
        )
    end

    error("no lightweight torture runner is implemented for $(row.id); use $(row.runner) directly")
end

function run_moment_rii_torture_smoke(; cases=moment_rii_torture_smoke_cases(), print_rows=false)
    Tuple(run_moment_rii_torture_case(id; print_rows=print_rows) for id in cases)
end
