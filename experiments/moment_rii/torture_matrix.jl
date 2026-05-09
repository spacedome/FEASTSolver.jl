const MOMENT_RII_TORTURE_REQUIRED_FAILURE_CLASSES = (
    :high_count_low_dimension,
    :nonnormal_weak_support,
    :near_pole_meromorphic,
    :singularity_accumulation,
    :algebraic_multiplicity,
    :residual_laurent_update,
    :sparse_realistic_nep,
    :schrodinger_domain_decomposition,
    :branch_cut,
    :dense_spectral_region,
    :defective_matrix_nep,
    :clustered_simple_roots,
    :polynomial_companion_control,
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
        id=:meromorphic_pole_ladder,
        failure_class=:singularity_accumulation,
        matrix_dimension=:low,
        spectral_difficulty=:finite_approximation_to_pole_accumulation,
        status=:covered,
        executable=true,
        smoke=false,
        runner=:run_meromorphic_pole_ladder_torture,
        evidence="torture diagnostic: meromorphic pole ladder; weak-support local count escalation",
        expected_behavior="A finite rational ladder with many roots near exterior poles reports a reliable global count and accepts weak-support clusters only after local algebraic count certification.",
        failure_mode="Meromorphic zero packets can accumulate near poles or contour boundaries, so cheap support-only retention can miss real clusters.",
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
        id=:quartic_sine_multiplicity,
        failure_class=:algebraic_multiplicity,
        matrix_dimension=:scalar,
        spectral_difficulty=:high_root_multiplicity,
        status=:covered,
        executable=true,
        smoke=false,
        runner=:run_high_multiplicity_sine_torture,
        evidence="torture diagnostic: quartic sine multiplicity",
        expected_behavior="Positive moments plus clustered local multiplicity probes recover the algebraic count while filtering raw spurious candidates.",
        failure_mode="High multiplicity requires enough positive moments and multiplicity-aware retained-cluster probes; raw support candidates are not the accepted set.",
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
        expected_behavior="No cross-cut torture test should be accepted until the branch, sheet, and contour analytic domain are part of the model.",
        failure_mode="Operators with branch cuts violate the single-valued analytic assumptions unless the sheet model is explicit.",
    ),
    (
        id=:branch_cut_fixed_sheet,
        failure_class=:branch_cut,
        matrix_dimension=:low,
        spectral_difficulty=:fixed_sheet_analytic_contour,
        status=:diagnostic_boundary,
        executable=true,
        smoke=false,
        runner=:run_branch_cut_fixed_sheet_torture,
        evidence="torture diagnostic: branch cut on a declared sheet",
        expected_behavior="A branch-function operator is acceptable only when the chosen sheet is analytic on and inside the contour.",
        failure_mode="Branch functions are not invalid by syntax; they are invalid when the sheet/domain data make the contour transfer multivalued.",
    ),
    (
        id=:near_branch_fixed_sheet,
        failure_class=:branch_cut,
        matrix_dimension=:low,
        spectral_difficulty=:near_branch_point_conditioning,
        status=:covered,
        executable=true,
        smoke=false,
        runner=:run_near_branch_fixed_sheet_torture,
        evidence="torture diagnostic: branch point just outside contour on declared sheet",
        expected_behavior="A declared analytic sheet remains solvable even when the branch point is close enough to stress contour conditioning.",
        failure_mode="Near branch points can look like near-pole contour stress, but the correct diagnostic is sheet/domain validity plus count reliability.",
    ),
    (
        id=:defective_triangular_multiplicity,
        failure_class=:defective_matrix_nep,
        matrix_dimension=:low,
        spectral_difficulty=:coalesced_eigenvectors,
        status=:covered,
        executable=true,
        smoke=false,
        runner=:run_defective_triangular_multiplicity_torture,
        evidence="torture diagnostic: triangular repeated-sine Jordan-like NEP",
        expected_behavior="Local count certification closes algebraic multiplicity for coalesced matrix eigenvectors, not only scalar repeated roots.",
        failure_mode="Matrix-valued multiplicity can collapse eigenvectors; scalar repeated-root tests do not exercise the left/right geometry.",
    ),
    (
        id=:clustered_simple_roots,
        failure_class=:clustered_simple_roots,
        matrix_dimension=:low,
        spectral_difficulty=:below_chart_resolution,
        status=:diagnostic_boundary,
        executable=true,
        smoke=false,
        runner=:run_clustered_simple_roots_torture,
        evidence="torture diagnostic: simple roots closer than local chart resolution",
        expected_behavior="Local count certification reports algebraic packet completion while documenting that nearby simple roots may be merged into packets.",
        failure_mode="A contour solver can certify the packet count without resolving each simple root as a separate geometric value unless the chart radius is reduced.",
    ),
    (
        id=:near_multiple_polynomial_companion,
        failure_class=:polynomial_companion_control,
        matrix_dimension=:low,
        spectral_difficulty=:clustered_polynomial_roots,
        status=:covered,
        executable=true,
        smoke=false,
        runner=:run_near_multiple_polynomial_companion_torture,
        evidence="torture diagnostic: polynomial-native extraction agrees with companion FEAST",
        expected_behavior="Polynomial-native moment extraction agrees with a companion linearization on clustered polynomial roots.",
        failure_mode="Polynomial NEPs are the lower rung between linear FEAST and fully general analytic NEPs; disagreement here would invalidate the higher-rung story.",
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

const MOMENT_RII_FAILURE_LAYER_REPORTS = (
    (
        id=:near_pole_rational,
        layer=:contour_count_quadrature,
        diagnostic=:argument_principle_count_error,
        steering=:increase_count_nodes_or_move_contour_away_from_singularity,
        fundamental=false,
        competing_solver_note="Not FEAST-specific: rational and contour solvers both become ill-conditioned when the target contour is too close to poles; non-contour solvers may avoid this contour but must still represent the nearby singularity.",
    ),
    (
        id=:meromorphic_pole_ladder,
        layer=:local_chart_support_retention,
        diagnostic=:reliable_count_weak_support_local_completion,
        steering=:probe_weak_support_clusters_or_refine_local_charts,
        fundamental=false,
        competing_solver_note="NLEIGS/Beyn may recover some instances with different interpolation or moments, but the same near-singularity packet is ill-conditioned; the support-only failure is fixed here by local count certification of weak clusters.",
    ),
    (
        id=:quartic_sine_multiplicity,
        layer=:multiplicity_moment_retention,
        diagnostic=:reliable_count_clustered_multiplicity_completion,
        steering=:increase_moments_to_local_multiplicity_then_filter_by_cluster_counts,
        fundamental=false,
        competing_solver_note="Derivative/Jordan or deflation solvers may represent multiplicity differently, but the contour-moment path is sufficient here once retained candidates are clustered before local multiplicity counts.",
    ),
    (
        id=:branch_cut_operator,
        layer=:analytic_model_validity,
        diagnostic=:missing_sheet_and_domain_data,
        steering=:declare_branch_sheet_or_reformulate_operator_before_solving,
        fundamental=true,
        competing_solver_note="No black-box eigenvalue solver can make a multivalued operator single-valued without branch/domain data; this is a problem-definition issue.",
    ),
    (
        id=:near_branch_fixed_sheet,
        layer=:contour_conditioning,
        diagnostic=:reliable_count_on_declared_sheet,
        steering=:increase_count_nodes_if_branch_point_approaches_contour,
        fundamental=false,
        competing_solver_note="A non-contour solver may not expose the branch geometry explicitly, but it still needs a consistent analytic sheet.",
    ),
    (
        id=:defective_triangular_multiplicity,
        layer=:matrix_multiplicity_geometry,
        diagnostic=:reliable_count_coalesced_eigenvectors,
        steering=:use_local_multiplicity_counts_after_two_sided_extraction,
        fundamental=false,
        competing_solver_note="Jordan-aware methods represent this directly; the contour-moment path certifies algebraic completion without explicitly forming Jordan chains.",
    ),
    (
        id=:clustered_simple_roots,
        layer=:local_resolution_limit,
        diagnostic=:count_complete_geometric_values_merged,
        steering=:reduce_local_chart_radius_or_accept_packet_level_solution,
        fundamental=false,
        competing_solver_note="Root-polishing methods can separate simple roots after good initial guesses; contour methods should report whether they solved a packet or resolved individual values.",
    ),
    (
        id=:near_multiple_polynomial_companion,
        layer=:polynomial_linearization_consistency,
        diagnostic=:native_polynomial_matches_companion_linearization,
        steering=:investigate_before_extending_to_general_nep_if_mismatch,
        fundamental=false,
        competing_solver_note="Companion linearization is a strong reference for polynomial NEPs, but it increases dimension; native moments should agree before claiming nonlinear generality.",
    ),
    (
        id=:dense_spectral_region,
        layer=:target_packet_definition,
        diagnostic=:unstable_or_nonisolated_packet,
        steering=:choose_a_stable_finite_packet_or_report_degenerate_target,
        fundamental=true,
        competing_solver_note="Other solvers may return values in the region, but correctness is not well-defined unless the target set is isolated or a different objective is specified.",
    ),
    (
        id=:schrodinger_dd_packet,
        layer=:packet_visibility,
        diagnostic=:packet_visible_defect,
        steering=:increase_contour_resolution_or_refine_extraction_without_changing_target_packet,
        fundamental=false,
        competing_solver_note="A direct large linearization can avoid the Schur packet diagnostic but pays with a much larger problem; the diagnostic is useful because it preserves the reduced formulation.",
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

function moment_rii_failure_layer_report(id::Symbol)
    for report in MOMENT_RII_FAILURE_LAYER_REPORTS
        report.id === id && return report
    end
    error("no failure-layer report is registered for $id")
end

function moment_rii_failure_layer_reports()
    MOMENT_RII_FAILURE_LAYER_REPORTS
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

function run_near_pole_count_reliability_sweep(;
    gap=0.005,
    nodes_values=(512, 2048, 8192),
    count_error_tol=1e-6,
    print_rows=true,
)
    rows = NamedTuple[]
    for nodes in nodes_values
        count = full_operator_count_estimate(
            oracle_free_near_pole_rational_cases(; gap=gap);
            outer_radius=1.0,
            operator_builder=triangular_operator_builder(; coupling=5.0),
            component_scaling=:contour_max,
            determinant_nodes=nodes,
            determinant_capacity=64,
        )
        push!(
            rows,
            (
                gap=Float64(gap),
                nodes=Int(nodes),
                count=count.count_estimate,
                count_error=count.count_error,
                reliable=count.count_error <= count_error_tol,
            ),
        )
    end
    if print_rows
        println()
        println("Near-pole count reliability sweep")
        println("  isolates argument-principle quadrature resolution near exterior poles")
        @printf("  %8s %8s %8s %12s %s\n", "gap", "nodes", "count", "count_error", "status")
        for row in rows
            @printf(
                "  %8.4f %8d %8d %12.3e %s\n",
                row.gap,
                row.nodes,
                row.count,
                row.count_error,
                row.reliable ? "reliable" : "underresolved",
            )
        end
    end
    first_reliable_nodes = nothing
    for row in rows
        if row.reliable
            first_reliable_nodes = row.nodes
            break
        end
    end
    (
        rows=Tuple(rows),
        first_reliable_nodes=first_reliable_nodes,
        layer=:contour_count_quadrature,
        steering=:increase_count_nodes_or_move_contour_away_from_singularity,
    )
end

function scalar_product_rational_case(; roots, poles, name=nothing)
    roots = ComplexF64.(roots)
    poles = ComplexF64.(poles)
    label = name === nothing ? "product_rational" : String(name)
    function f(z)
        prod(z - root for root in roots; init=one(ComplexF64)) /
            prod(z - pole for pole in poles; init=one(ComplexF64))
    end
    function df(z)
        value = f(z)
        root_sum = sum(inv(z - root) for root in roots; init=zero(ComplexF64))
        pole_sum = sum(inv(z - pole) for pole in poles; init=zero(ComplexF64))
        value * (root_sum - pole_sum)
    end
    function fmat(S)
        Ired = Matrix{ComplexF64}(I, size(S, 1), size(S, 2))
        numerator = Matrix{ComplexF64}(I, size(S, 1), size(S, 2))
        denominator = Matrix{ComplexF64}(I, size(S, 1), size(S, 2))
        for root in roots
            numerator = numerator * (S .- root .* Ired)
        end
        for pole in poles
            denominator = denominator * (S .- pole .* Ired)
        end
        numerator / denominator
    end
    (
        name=label,
        f=f,
        df=df,
        fmat=fmat,
        roots=(center, radius) -> scalar_roots_in_contour(roots, center, radius),
    )
end

function scalar_powered_sine_case(; shift=0.0 + 0.0im, power=4, name=nothing)
    shift = ComplexF64(shift)
    power = Int(power)
    label = name === nothing ? "sine_power_$power" : String(name)
    (
        name=label,
        f=z -> sin(z - shift)^power,
        df=z -> power * sin(z - shift)^(power - 1) * cos(z - shift),
        fmat=S -> begin
            Ired = Matrix{ComplexF64}(I, size(S, 1), size(S, 2))
            sin(S .- shift .* Ired)^power
        end,
        roots=(center, radius) -> begin
            lower = (real(center) - radius - real(shift)) / pi
            upper = (real(center) + radius - real(shift)) / pi
            candidates = ComplexF64[shift + pi * k for k in floor(Int, lower)-2:ceil(Int, upper)+2]
            scalar_roots_in_contour(candidates, center, radius)
        end,
    )
end

function scalar_branch_sqrt_case(; branch_point=-2.0 + 0.0im, beta, name=nothing)
    branch_point = ComplexF64(branch_point)
    beta = ComplexF64(beta)
    root = branch_point + beta^2
    label = name === nothing ? "sqrt_sheet_$(beta)" : String(name)
    (
        name=label,
        f=z -> sqrt(z - branch_point) - beta,
        df=z -> inv(2 * sqrt(z - branch_point)),
        fmat=S -> begin
            error("matrix square-root branch case is only used through scalar contour samples")
        end,
        roots=(center, radius) -> scalar_roots_in_contour(ComplexF64[root], center, radius),
    )
end

function meromorphic_pole_ladder_cases(; gap=0.035)
    angles = (0.05, 0.21, -0.17)
    Tuple(
        scalar_product_rational_case(;
            name="pole_ladder_$j",
            roots=[
                0.18 * cis(angle),
                0.44 * cis(angle + 0.35),
                0.68 * cis(angle - 0.25),
                0.84 * cis(angle + 0.10),
            ],
            poles=[
                (1.0 + gap) * cis(angle),
                (1.0 + 2gap) * cis(angle + 0.22),
                (1.0 + 3gap) * cis(angle - 0.18),
            ],
        )
        for (j, angle) in pairs(angles)
    )
end

function clustered_simple_root_cases()
    (
        scalar_product_rational_case(;
            name="clustered_simple_1",
            roots=[-0.55, -0.52, 0.15 + 0.02im, 0.18 + 0.02im],
            poles=[],
        ),
        scalar_product_rational_case(;
            name="clustered_simple_2",
            roots=[-0.10 - 0.02im, -0.07 - 0.02im, 0.42 + 0.03im, 0.45 + 0.03im],
            poles=[],
        ),
        scalar_product_rational_case(;
            name="clustered_simple_3",
            roots=[0.65 - 0.04im, 0.68 - 0.04im, -0.35 + 0.04im, -0.32 + 0.04im],
            poles=[],
        ),
    )
end

function run_meromorphic_pole_ladder_torture(;
    gap=0.035,
    outer_radius=1.0,
    base_spacing=0.34,
    chart_radii=(0.24, 0.42),
    count_error_tol=1e-5,
    residual_tol=1e-8,
    match_atol=1e-6,
    print_rows=true,
)
    result = run_count_driven_adaptive_grid_refinement(;
        label="Meromorphic pole-ladder torture",
        cases=meromorphic_pole_ladder_cases(; gap=gap),
        outer_radius=outer_radius,
        operator_builder=dense_multi_delay_operator_builder(; coupling=0.08),
        operator_label="dense rational pole ladder",
        base_spacing=base_spacing,
        chart_radii=chart_radii,
        max_refinement_rounds=4,
        iterations=2,
        basis_moments=10,
        basis_nodes=80,
        rii_nodes=192,
        determinant_nodes=1024,
        determinant_capacity=128,
        reduced_moments=20,
        reduced_nodes=1024,
        residual_normalization=:operator,
        component_scaling=:contour_max,
        count_error_tol=count_error_tol,
        residual_tol=residual_tol,
        match_atol=match_atol,
        print_rows=print_rows,
    )
    diagnostic = count_driven_chart_diagnostic_summary(
        result;
        outer_radius=outer_radius,
        match_atol=match_atol,
        count_error_tol=count_error_tol,
    )
    (
        result=result,
        diagnostic=diagnostic,
        rows=result.rows,
        count=result.count,
        stop_reason=result.stop_reason,
        algebraic_retained_count=result.algebraic_retained_count,
        accepted=result.stop_reason in (:target_count_complete, :target_algebraic_count_complete),
        count_reliable=result.count.count_error <= count_error_tol,
    )
end

function run_defective_triangular_multiplicity_torture(;
    outer_radius=7.0,
    residual_tol=1e-8,
    match_atol=1e-6,
    print_rows=true,
)
    result = run_count_driven_adaptive_grid_refinement(;
        label="Defective triangular multiplicity torture",
        cases=(scalar_sine_case(), scalar_sine_case(), scalar_sine_case()),
        outer_radius=outer_radius,
        operator_builder=triangular_operator_builder(; coupling=4.0),
        operator_label="triangular repeated sine",
        base_spacing=2.0,
        chart_radii=(1.3, 2.2),
        max_refinement_rounds=2,
        iterations=2,
        basis_moments=6,
        basis_nodes=48,
        rii_nodes=128,
        determinant_nodes=1024,
        determinant_capacity=64,
        reduced_moments=12,
        reduced_nodes=512,
        residual_normalization=:operator,
        component_scaling=:contour_max,
        residual_tol=residual_tol,
        match_atol=match_atol,
        print_rows=print_rows,
    )
    (
        result=result,
        rows=result.rows,
        count=result.count,
        stop_reason=result.stop_reason,
        multiplicities=result.multiplicities,
        algebraic_retained_count=result.algebraic_retained_count,
        accepted=result.stop_reason === :target_algebraic_count_complete &&
            result.algebraic_retained_count == result.count.count_estimate,
    )
end

function run_clustered_simple_roots_torture(;
    outer_radius=1.0,
    residual_tol=1e-8,
    match_atol=1e-6,
    print_rows=true,
)
    result = run_count_driven_adaptive_grid_refinement(;
        label="Clustered simple roots torture",
        cases=clustered_simple_root_cases(),
        outer_radius=outer_radius,
        operator_builder=triangular_operator_builder(; coupling=0.8),
        operator_label="triangular clustered simple roots",
        base_spacing=0.34,
        chart_radii=(0.18, 0.32),
        max_refinement_rounds=3,
        iterations=2,
        basis_moments=8,
        basis_nodes=64,
        rii_nodes=128,
        determinant_nodes=1024,
        determinant_capacity=64,
        reduced_moments=16,
        reduced_nodes=512,
        residual_normalization=:operator,
        component_scaling=:contour_max,
        residual_tol=residual_tol,
        match_atol=match_atol,
        print_rows=print_rows,
    )
    (
        result=result,
        rows=result.rows,
        count=result.count,
        stop_reason=result.stop_reason,
        multiplicities=result.multiplicities,
        algebraic_retained_count=result.algebraic_retained_count,
        accepted=result.stop_reason === :target_algebraic_count_complete &&
            result.algebraic_retained_count == result.count.count_estimate,
        packet_level_solution=last(result.rows).retained < result.count.count_estimate,
    )
end

function meromorphic_pole_ladder_boundary_status(result)
    final = last(result.rows)
    if !result.count_reliable
        return :unreliable_count
    elseif result.accepted &&
            (final.retained == result.count.count_estimate ||
             result.algebraic_retained_count == result.count.count_estimate)
        return :accepted
    elseif result.stop_reason === :count_multiplicity_or_unresolved_defect &&
            final.retained < result.count.count_estimate
        return :incomplete_retention
    else
        return :unclassified_boundary
    end
end

function run_meromorphic_pole_ladder_boundary_sweep(;
    gaps=(0.12, 0.07, 0.035),
    print_rows=true,
)
    rows = NamedTuple[]
    results = Any[]
    for gap in gaps
        result = run_meromorphic_pole_ladder_torture(; gap=gap, print_rows=false)
        push!(results, result)
        final = last(result.rows)
        push!(
            rows,
            (
                gap=Float64(gap),
                status=meromorphic_pole_ladder_boundary_status(result),
                stop_reason=result.stop_reason,
                count=result.count.count_estimate,
                count_error=result.count.count_error,
                retained=final.retained,
                reliable=result.count_reliable,
                accepted=result.accepted,
            ),
        )
    end
    if print_rows
        println()
        println("Meromorphic pole-ladder boundary sweep")
        println("  finite rational approximation to roots accumulating near exterior poles")
        @printf("  %8s %12s %9s %9s %12s %s\n", "gap", "count_error", "retained", "count", "status", "stop")
        for row in rows
            @printf(
                "  %8.3f %12.3e %9d %9d %12s %s\n",
                row.gap,
                row.count_error,
                row.retained,
                row.count,
                string(row.status),
                string(row.stop_reason),
            )
        end
    end
    transition_gap = Inf
    for row in rows
        if row.status !== :accepted
            transition_gap = row.gap
            break
        end
    end
    (
        rows=Tuple(rows),
        results=Tuple(results),
        transition_gap=transition_gap,
    )
end

function run_high_multiplicity_sine_torture(;
    power=4,
    outer_radius=7.0,
    base_spacing=2.0,
    chart_radii=(1.3, 2.2),
    residual_tol=1e-8,
    match_atol=1e-6,
    print_rows=true,
)
    result = run_count_driven_adaptive_grid_refinement(;
        label="High-multiplicity sine torture",
        cases=(scalar_powered_sine_case(; power=power),),
        outer_radius=outer_radius,
        base_spacing=base_spacing,
        chart_radii=chart_radii,
        basis_moments=12,
        basis_nodes=96,
        determinant_nodes=1024,
        determinant_capacity=128,
        reduced_moments=24,
        reduced_nodes=1024,
        component_scaling=:contour_max,
        residual_tol=residual_tol,
        match_atol=match_atol,
        print_rows=print_rows,
    )
    (
        result=result,
        rows=result.rows,
        count=result.count,
        stop_reason=result.stop_reason,
        multiplicities=result.multiplicities,
        expected_power=power,
        accepted=result.stop_reason === :target_algebraic_count_complete &&
            !isempty(result.multiplicities) &&
            all(item.multiplicity == power for item in result.multiplicities),
    )
end

function high_multiplicity_sine_boundary_status(result)
    final = last(result.rows)
    expected_unique = result.count.count_estimate ÷ result.expected_power
    filtered_unique = length(result.multiplicities)
    if result.accepted && filtered_unique == expected_unique
        return :accepted_unique_multiplicity
    elseif result.stop_reason === :target_algebraic_count_complete && final.retained > expected_unique
        return :algebraic_count_with_spurious_retention
    elseif result.stop_reason === :max_rounds && final.retained > result.count.count_estimate
        return :runaway_overretention
    elseif result.count.count_error > 1e-8
        return :unreliable_count
    else
        return :unclassified_boundary
    end
end

function run_high_multiplicity_sine_boundary_sweep(;
    powers=(2, 3, 4),
    outer_radius=4.0,
    print_rows=true,
)
    rows = NamedTuple[]
    results = Any[]
    for power in powers
        result = run_high_multiplicity_sine_torture(;
            power=power,
            outer_radius=outer_radius,
            print_rows=false,
        )
        push!(results, result)
        final = last(result.rows)
        expected_unique = result.count.count_estimate ÷ result.expected_power
        push!(
            rows,
            (
                power=Int(power),
                status=high_multiplicity_sine_boundary_status(result),
                stop_reason=result.stop_reason,
                count=result.count.count_estimate,
                count_error=result.count.count_error,
                expected_unique=expected_unique,
                retained=final.retained,
                filtered_retained=length(result.multiplicities),
                algebraic_retained=result.result.algebraic_retained_count,
                multiplicities=Tuple(item.multiplicity for item in result.multiplicities),
                accepted=result.accepted,
            ),
        )
    end
    if print_rows
        println()
        println("High-multiplicity sine boundary sweep")
        println("  separates low multiplicity success from spurious over-retention and runaway refinement")
        @printf("  %6s %12s %9s %9s %9s %12s %s\n", "power", "count_error", "unique", "retained", "count", "status", "stop")
        for row in rows
            @printf(
                "  %6d %12.3e %9d %9d %9d %12s %s\n",
                row.power,
                row.count_error,
                row.expected_unique,
                row.retained,
                row.count,
                string(row.status),
                string(row.stop_reason),
            )
        end
    end
    first_failure_power = typemax(Int)
    for row in rows
        if row.status !== :accepted_unique_multiplicity
            first_failure_power = row.power
            break
        end
    end
    (
        rows=Tuple(rows),
        results=Tuple(results),
        first_failure_power=first_failure_power,
    )
end

function high_multiplicity_moment_adequacy_recommendation(row)
    if row.moments < row.power
        return :increase_moments_to_algebraic_multiplicity
    elseif row.status === :accepted_unique_multiplicity
        return :accept
    elseif row.status in (:algebraic_count_with_spurious_retention, :runaway_overretention)
        return :moments_are_not_the_limiting_layer_use_deflation_or_retention
    elseif row.status === :unreliable_count
        return :fix_contour_count_before_moment_policy
    else
        return :inspect_extraction_layer
    end
end

function run_high_multiplicity_moment_order_sweep(;
    powers=(3, 4),
    moment_factors=(1, 2, 4),
    outer_radius=4.0,
    print_rows=true,
)
    rows = NamedTuple[]
    results = Any[]
    for power in powers
        for factor in moment_factors
            moments = Int(power * factor)
            result = run_count_driven_adaptive_grid_refinement(;
                label="High-multiplicity moment-order adequacy",
                cases=(scalar_powered_sine_case(; power=power),),
                outer_radius=outer_radius,
                base_spacing=2.0,
                chart_radii=(1.3, 2.2),
                max_refinement_rounds=3,
                basis_moments=moments,
                basis_nodes=max(64, 8moments),
                determinant_nodes=1024,
                determinant_capacity=128,
                reduced_moments=max(2moments, 8),
                reduced_nodes=1024,
                component_scaling=:contour_max,
                residual_tol=1e-8,
                match_atol=1e-6,
                print_rows=false,
            )
            wrapped = (
                result=result,
                rows=result.rows,
                count=result.count,
                stop_reason=result.stop_reason,
                multiplicities=result.multiplicities,
                expected_power=power,
                accepted=result.stop_reason === :target_algebraic_count_complete &&
                    !isempty(result.multiplicities) &&
                    all(item.multiplicity == power for item in result.multiplicities),
            )
            push!(results, wrapped)
            final = last(result.rows)
            expected_unique = result.count.count_estimate ÷ power
            status = high_multiplicity_sine_boundary_status(wrapped)
            row = (
                power=Int(power),
                moments=moments,
                status=status,
                stop_reason=result.stop_reason,
                count=result.count.count_estimate,
                count_error=result.count.count_error,
                expected_unique=expected_unique,
                retained=final.retained,
                filtered_retained=length(result.multiplicities),
                algebraic_retained=result.algebraic_retained_count,
                moment_adequate=moments >= power,
                multiplicities=Tuple(item.multiplicity for item in result.multiplicities),
            )
            push!(
                rows,
                merge(row, (recommendation=high_multiplicity_moment_adequacy_recommendation(row),)),
            )
        end
    end
    if print_rows
        println()
        println("High-multiplicity moment-order adequacy sweep")
        println("  tests whether adding positive moments changes the multiplicity failure layer")
        @printf("  %6s %8s %9s %9s %9s %12s %s\n", "power", "moments", "unique", "retained", "count", "status", "recommendation")
        for row in rows
            @printf(
                "  %6d %8d %9d %9d %9d %12s %s\n",
                row.power,
                row.moments,
                row.expected_unique,
                row.retained,
                row.count,
                string(row.status),
                string(row.recommendation),
            )
        end
    end
    (
        rows=Tuple(rows),
        results=Tuple(results),
        all_moment_adequate=all(row.moment_adequate for row in rows),
        unchanged_by_moment_escalation=all(
            length(unique(row.status for row in rows if row.power == power)) == 1
            for power in powers
        ),
    )
end

function run_branch_cut_fixed_sheet_torture(;
    outer_radius=1.0,
    base_spacing=0.42,
    chart_radii=(0.3, 0.48),
    residual_tol=1e-8,
    match_atol=1e-6,
    print_rows=true,
)
    cases = (
        scalar_branch_sqrt_case(; branch_point=-2.0, beta=sqrt(1.55), name="sqrt1"),
        scalar_branch_sqrt_case(; branch_point=-2.4, beta=sqrt(2.05 + 0.15im), name="sqrt2"),
    )
    result = run_count_driven_adaptive_grid_refinement(;
        label="Fixed-sheet branch-cut torture",
        cases=cases,
        outer_radius=outer_radius,
        operator_builder=triangular_operator_builder(; coupling=0.1),
        operator_label="triangular fixed-sheet sqrt",
        base_spacing=base_spacing,
        chart_radii=chart_radii,
        max_refinement_rounds=3,
        iterations=2,
        basis_moments=6,
        basis_nodes=48,
        determinant_nodes=512,
        determinant_capacity=32,
        reduced_moments=12,
        reduced_nodes=512,
        residual_normalization=:operator,
        component_scaling=:contour_max,
        residual_tol=residual_tol,
        match_atol=match_atol,
        print_rows=print_rows,
    )
    (
        result=result,
        rows=result.rows,
        count=result.count,
        stop_reason=result.stop_reason,
        accepted=result.stop_reason === :target_count_complete,
        sheet_domain="principal sqrt, branch points/cuts outside the contour disk",
    )
end

function run_near_branch_fixed_sheet_torture(;
    outer_radius=1.0,
    count_error_tol=1e-5,
    residual_tol=1e-8,
    match_atol=1e-6,
    print_rows=true,
)
    result = run_count_driven_adaptive_grid_refinement(;
        label="Near-branch fixed-sheet torture",
        cases=(
            scalar_branch_sqrt_case(; branch_point=-1.03, beta=sqrt(1.25), name="near_sqrt1"),
            scalar_branch_sqrt_case(; branch_point=-1.06, beta=sqrt(1.45 + 0.08im), name="near_sqrt2"),
            scalar_branch_sqrt_case(; branch_point=-1.09, beta=sqrt(1.65 - 0.06im), name="near_sqrt3"),
        ),
        outer_radius=outer_radius,
        operator_builder=similarity_analytic_tools,
        operator_label="near branch fixed sheet",
        base_spacing=0.34,
        chart_radii=(0.24, 0.42),
        max_refinement_rounds=3,
        iterations=2,
        basis_moments=6,
        basis_nodes=64,
        rii_nodes=160,
        determinant_nodes=2048,
        determinant_capacity=64,
        reduced_moments=12,
        reduced_nodes=1024,
        residual_normalization=:operator,
        component_scaling=:contour_max,
        count_error_tol=count_error_tol,
        residual_tol=residual_tol,
        match_atol=match_atol,
        print_rows=print_rows,
    )
    (
        result=result,
        rows=result.rows,
        count=result.count,
        stop_reason=result.stop_reason,
        accepted=result.stop_reason === :target_count_complete &&
            result.count.count_error <= count_error_tol,
        sheet_domain="principal sqrt, branch points just outside the contour disk",
    )
end

function run_near_multiple_polynomial_companion_torture(;
    residual_tol=1e-7,
    match_atol=1e-5,
    print_rows=true,
)
    result = run_polynomial_family_bridge_diagnostic(;
        name="near_multiple_cluster_poly",
        make_problem=near_multiple_polynomial_problem,
        nodes=48,
        iterations=4,
        basis_moments=5,
        basis_nodes=64,
        rii_nodes=128,
        update_moment_count=2,
        residual_tol=residual_tol,
        match_atol=match_atol,
        print_rows=print_rows,
    )
    (
        result=result,
        companion=result.companion,
        rows=result.polynomial_rows,
        accepted=result.companion.matched == result.companion.expected &&
            all(row -> row.matched == result.companion.expected && row.spurious == 0, result.polynomial_rows),
    )
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
    elseif row.id === :meromorphic_pole_ladder
        result = run_meromorphic_pole_ladder_torture(; print_rows=print_rows)
        passed = result.count_reliable &&
            result.stop_reason in (:target_count_complete, :target_algebraic_count_complete) &&
            result.algebraic_retained_count == result.count.count_estimate
        return (
            id=row.id,
            status=row.status,
            passed=passed,
            skipped=false,
            metrics=(
                count=result.count.count_estimate,
                count_error=result.count.count_error,
                stop_reason=result.stop_reason,
                retained=last(result.rows).retained,
                algebraic_retained=result.algebraic_retained_count,
            ),
        )
    elseif row.id === :quartic_sine_multiplicity
        result = run_high_multiplicity_sine_torture(; print_rows=print_rows)
        expected_unique = result.count.count_estimate ÷ result.expected_power
        passed = result.accepted &&
            result.count.count_error <= 1e-8 &&
            length(result.multiplicities) == expected_unique &&
            all(item.multiplicity == result.expected_power for item in result.multiplicities)
        return (
            id=row.id,
            status=row.status,
            passed=passed,
            skipped=false,
            metrics=(
                count=result.count.count_estimate,
                raw_retained=last(result.rows).retained,
                filtered_retained=length(result.multiplicities),
                multiplicities=Tuple(item.multiplicity for item in result.multiplicities),
                stop_reason=result.stop_reason,
            ),
        )
    elseif row.id === :defective_triangular_multiplicity
        result = run_defective_triangular_multiplicity_torture(; print_rows=print_rows)
        passed = result.accepted &&
            all(item.multiplicity == 3 for item in result.multiplicities)
        return (
            id=row.id,
            status=row.status,
            passed=passed,
            skipped=false,
            metrics=(
                count=result.count.count_estimate,
                algebraic_retained=result.algebraic_retained_count,
                retained=last(result.rows).retained,
                multiplicities=Tuple(item.multiplicity for item in result.multiplicities),
                stop_reason=result.stop_reason,
            ),
        )
    elseif row.id === :clustered_simple_roots
        result = run_clustered_simple_roots_torture(; print_rows=print_rows)
        passed = result.accepted &&
            result.packet_level_solution &&
            all(item.multiplicity == 2 for item in result.multiplicities)
        return (
            id=row.id,
            status=row.status,
            passed=passed,
            skipped=false,
            metrics=(
                count=result.count.count_estimate,
                algebraic_retained=result.algebraic_retained_count,
                retained=last(result.rows).retained,
                packets=length(result.multiplicities),
                multiplicities=Tuple(item.multiplicity for item in result.multiplicities),
                stop_reason=result.stop_reason,
            ),
        )
    elseif row.id === :branch_cut_fixed_sheet
        result = run_branch_cut_fixed_sheet_torture(; print_rows=print_rows)
        passed = result.accepted
        return (
            id=row.id,
            status=row.status,
            passed=passed,
            skipped=false,
            metrics=(
                count=result.count.count_estimate,
                retained=last(result.rows).retained,
                stop_reason=result.stop_reason,
            ),
        )
    elseif row.id === :near_branch_fixed_sheet
        result = run_near_branch_fixed_sheet_torture(; print_rows=print_rows)
        passed = result.accepted &&
            result.count.count_estimate == 3 &&
            last(result.rows).retained == result.count.count_estimate
        return (
            id=row.id,
            status=row.status,
            passed=passed,
            skipped=false,
            metrics=(
                count=result.count.count_estimate,
                count_error=result.count.count_error,
                retained=last(result.rows).retained,
                stop_reason=result.stop_reason,
                sheet_domain=result.sheet_domain,
            ),
        )
    elseif row.id === :near_multiple_polynomial_companion
        result = run_near_multiple_polynomial_companion_torture(; print_rows=print_rows)
        passed = result.accepted &&
            result.companion.expected == 18 &&
            all(row -> row.success, result.rows)
        return (
            id=row.id,
            status=row.status,
            passed=passed,
            skipped=false,
            metrics=(
                expected=result.companion.expected,
                companion_matched=result.companion.matched,
                native_matched=Tuple(row.matched for row in result.rows),
                max_native_residual=maximum(row.max_residual for row in result.rows),
                companion_size=result.companion.companion_size,
            ),
        )
    end

    error("no lightweight torture runner is implemented for $(row.id); use $(row.runner) directly")
end

function run_moment_rii_torture_smoke(; cases=moment_rii_torture_smoke_cases(), print_rows=false)
    Tuple(run_moment_rii_torture_case(id; print_rows=print_rows) for id in cases)
end
