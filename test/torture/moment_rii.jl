@testitem "torture moment RII: failure matrix covers hard NEP classes" tags=[:torture, :moment_rii] begin
    include(joinpath(@__DIR__, "..", "..", "experiments", "moment_rii", "run.jl"))

    matrix = moment_rii_torture_matrix()
    required = moment_rii_torture_required_failure_classes()

    @test length(matrix) >= length(required)
    for failure_class in required
        @test any(row -> row.failure_class === failure_class, matrix)
    end

    for row in matrix
        @test row.id isa Symbol
        @test row.failure_class isa Symbol
        @test row.status isa Symbol
        @test !isempty(row.evidence)
        @test !isempty(row.expected_behavior)
        @test !isempty(row.failure_mode)
        @test row.runner isa Symbol
    end

    covered = filter(row -> row.status === :covered, matrix)
    gaps = filter(row -> !row.executable, matrix)
    smoke = moment_rii_torture_smoke_cases()
    summary = moment_rii_torture_coverage_summary()

    @test length(covered) >= 7
    @test :residual_laurent_correction_space in smoke
    @test :sparse_schrodinger_gallery in smoke
    @test :schrodinger_dd_packet in smoke
    @test all(id -> moment_rii_torture_entry(id).executable, smoke)
    @test any(row -> row.id === :branch_cut_operator && row.status === :documented_gap, gaps)
    @test any(row -> row.id === :dense_spectral_region && row.status === :degenerate_boundary, gaps)
    @test all(row -> row.documented, summary)
    @test moment_rii_torture_entry(:branch_cut_operator).executable == false
    @test any(row -> row.failure_class === :dense_spectral_region && !row.executable, summary)
    @test moment_rii_torture_entry(:meromorphic_pole_ladder).status === :covered
    @test moment_rii_torture_entry(:quartic_sine_multiplicity).status === :covered
    @test moment_rii_torture_entry(:branch_cut_fixed_sheet).status === :diagnostic_boundary
    @test moment_rii_torture_entry(:defective_triangular_multiplicity).status === :covered
    @test moment_rii_torture_entry(:clustered_simple_roots).status === :diagnostic_boundary
    @test moment_rii_torture_entry(:near_multiple_polynomial_companion).status === :covered

    reports = moment_rii_failure_layer_reports()
    @test any(report -> report.id === :near_pole_rational && report.layer === :contour_count_quadrature, reports)
    @test any(report -> report.id === :meromorphic_pole_ladder && report.layer === :local_chart_support_retention, reports)
    @test any(report -> report.id === :quartic_sine_multiplicity && report.layer === :multiplicity_moment_retention, reports)
    @test any(report -> report.id === :defective_triangular_multiplicity && report.layer === :matrix_multiplicity_geometry, reports)
    @test any(report -> report.id === :clustered_simple_roots && report.layer === :local_resolution_limit, reports)
    @test any(report -> report.id === :branch_cut_operator && report.fundamental, reports)
end

@testitem "torture moment RII: generated adversarial NEP profiles" tags=[:slow, :torture, :moment_rii, :moment_heavy] begin
    include(joinpath(@__DIR__, "..", "..", "experiments", "moment_rii", "run.jl"))

    branch = run_moment_rii_torture_case(:branch_cut_fixed_sheet; print_rows=false)
    @test branch.passed
    @test branch.metrics.count == 2
    @test branch.metrics.retained == branch.metrics.count
    @test branch.metrics.stop_reason === :target_count_complete

    pole_ladder = run_moment_rii_torture_case(:meromorphic_pole_ladder; print_rows=false)
    @test pole_ladder.passed
    @test pole_ladder.status === :covered
    @test pole_ladder.metrics.count == 12
    @test pole_ladder.metrics.retained < pole_ladder.metrics.count
    @test pole_ladder.metrics.algebraic_retained == pole_ladder.metrics.count
    @test pole_ladder.metrics.stop_reason === :target_algebraic_count_complete

    multiplicity = run_moment_rii_torture_case(:quartic_sine_multiplicity; print_rows=false)
    @test multiplicity.passed
    @test multiplicity.status === :covered
    @test multiplicity.metrics.count == 20
    @test multiplicity.metrics.raw_retained <= multiplicity.metrics.filtered_retained
    @test multiplicity.metrics.filtered_retained == 5
    @test all(==(4), multiplicity.metrics.multiplicities)
    @test multiplicity.metrics.stop_reason === :target_algebraic_count_complete
end

@testitem "torture moment RII: expanded adversarial NEP profiles" tags=[:slow, :torture, :moment_rii, :moment_heavy] begin
    include(joinpath(@__DIR__, "..", "..", "experiments", "moment_rii", "run.jl"))

    defective = run_moment_rii_torture_case(:defective_triangular_multiplicity; print_rows=false)
    @test defective.passed
    @test defective.metrics.count == 15
    @test defective.metrics.algebraic_retained == defective.metrics.count
    @test all(==(3), defective.metrics.multiplicities)
    @test defective.metrics.stop_reason === :target_algebraic_count_complete

    clustered = run_moment_rii_torture_case(:clustered_simple_roots; print_rows=false)
    @test clustered.passed
    @test clustered.status === :diagnostic_boundary
    @test clustered.metrics.count == 12
    @test clustered.metrics.algebraic_retained == clustered.metrics.count
    @test clustered.metrics.packets == 6
    @test all(==(2), clustered.metrics.multiplicities)

    near_branch = run_moment_rii_torture_case(:near_branch_fixed_sheet; print_rows=false)
    @test near_branch.passed
    @test near_branch.metrics.count == 3
    @test near_branch.metrics.retained == 3
    @test near_branch.metrics.count_error <= 1e-8
    @test near_branch.metrics.stop_reason === :target_count_complete

    polynomial = run_moment_rii_torture_case(:near_multiple_polynomial_companion; print_rows=false)
    @test polynomial.passed
    @test polynomial.metrics.expected == 18
    @test polynomial.metrics.companion_matched == 18
    @test polynomial.metrics.native_matched == (18, 18, 18)
    @test polynomial.metrics.companion_size == 30
    @test polynomial.metrics.max_native_residual <= 1e-8
end

@testitem "torture moment RII: failure boundary sweeps characterize transitions" tags=[:slow, :torture, :moment_rii, :moment_heavy] begin
    include(joinpath(@__DIR__, "..", "..", "experiments", "moment_rii", "run.jl"))

    pole = run_meromorphic_pole_ladder_boundary_sweep(; print_rows=false)
    @test Tuple(row.status for row in pole.rows) == (:accepted, :accepted, :accepted)
    @test Tuple(row.retained for row in pole.rows) == (12, 11, 9)
    @test isinf(pole.transition_gap)
    @test all(row.count == 12 for row in pole.rows)
    @test all(row.count_error <= 1e-8 for row in pole.rows)

    multiplicity = run_high_multiplicity_sine_boundary_sweep(; print_rows=false)
    @test Tuple(row.status for row in multiplicity.rows) == (
        :accepted_unique_multiplicity,
        :accepted_unique_multiplicity,
        :accepted_unique_multiplicity,
    )
    @test Tuple(row.power for row in multiplicity.rows) == (2, 3, 4)
    @test Tuple(row.count for row in multiplicity.rows) == (6, 9, 12)
    @test multiplicity.rows[1].retained == multiplicity.rows[1].expected_unique
    @test multiplicity.rows[2].filtered_retained == multiplicity.rows[2].expected_unique
    @test multiplicity.rows[3].filtered_retained == multiplicity.rows[3].expected_unique
    @test multiplicity.rows[3].retained <= multiplicity.rows[3].filtered_retained
    @test multiplicity.first_failure_power == typemax(Int)
end

@testitem "torture moment RII: multiplicity moment escalation identifies next layer" tags=[:slow, :torture, :moment_rii, :moment_heavy] begin
    include(joinpath(@__DIR__, "..", "..", "experiments", "moment_rii", "run.jl"))

    sweep = run_high_multiplicity_moment_order_sweep(; print_rows=false)

    @test sweep.all_moment_adequate
    @test sweep.unchanged_by_moment_escalation
    @test all(row.count_error <= 1e-8 for row in sweep.rows)
    @test all(row -> row.recommendation === :accept, sweep.rows)

    p3 = filter(row -> row.power == 3, sweep.rows)
    p4 = filter(row -> row.power == 4, sweep.rows)
    @test Tuple(row.moments for row in p3) == (3, 6, 12)
    @test Tuple(row.status for row in p3) == (
        :accepted_unique_multiplicity,
        :accepted_unique_multiplicity,
        :accepted_unique_multiplicity,
    )
    @test Tuple(row.moments for row in p4) == (4, 8, 16)
    @test Tuple(row.status for row in p4) == (
        :accepted_unique_multiplicity,
        :accepted_unique_multiplicity,
        :accepted_unique_multiplicity,
    )
end

@testitem "torture moment RII: near-pole count diagnostic identifies quadrature layer" tags=[:slow, :torture, :moment_rii, :moment_heavy] begin
    include(joinpath(@__DIR__, "..", "..", "experiments", "moment_rii", "run.jl"))

    sweep = run_near_pole_count_reliability_sweep(; print_rows=false)

    @test sweep.layer === :contour_count_quadrature
    @test sweep.steering === :increase_count_nodes_or_move_contour_away_from_singularity
    @test Tuple(row.nodes for row in sweep.rows) == (512, 2048, 8192)
    @test Tuple(row.count for row in sweep.rows) == (5, 5, 5)
    @test sweep.rows[1].count_error > 1e-1
    @test sweep.rows[2].count_error > 1e-6
    @test sweep.rows[3].count_error <= 1e-6
    @test sweep.first_reliable_nodes == 8192

    report = moment_rii_failure_layer_report(:near_pole_rational)
    @test report.layer === :contour_count_quadrature
    @test report.diagnostic === :argument_principle_count_error
    @test !report.fundamental
end
