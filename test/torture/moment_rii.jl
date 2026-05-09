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
    @test moment_rii_torture_entry(:meromorphic_pole_ladder).status === :known_failure_boundary
    @test moment_rii_torture_entry(:quartic_sine_multiplicity).status === :known_failure_boundary
    @test moment_rii_torture_entry(:branch_cut_fixed_sheet).status === :diagnostic_boundary

    reports = moment_rii_failure_layer_reports()
    @test any(report -> report.id === :near_pole_rational && report.layer === :contour_count_quadrature, reports)
    @test any(report -> report.id === :meromorphic_pole_ladder && report.layer === :local_chart_support_retention, reports)
    @test any(report -> report.id === :quartic_sine_multiplicity && report.layer === :multiplicity_deflation_retention, reports)
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
    @test pole_ladder.status === :known_failure_boundary
    @test pole_ladder.metrics.count == 12
    @test pole_ladder.metrics.retained < pole_ladder.metrics.count
    @test pole_ladder.metrics.stop_reason === :count_multiplicity_or_unresolved_defect

    multiplicity = run_moment_rii_torture_case(:quartic_sine_multiplicity; print_rows=false)
    @test multiplicity.passed
    @test multiplicity.status === :known_failure_boundary
    @test multiplicity.metrics.count == 20
    @test multiplicity.metrics.unique_retained > multiplicity.metrics.count
    @test multiplicity.metrics.stop_reason === :max_rounds
end

@testitem "torture moment RII: failure boundary sweeps characterize transitions" tags=[:slow, :torture, :moment_rii, :moment_heavy] begin
    include(joinpath(@__DIR__, "..", "..", "experiments", "moment_rii", "run.jl"))

    pole = run_meromorphic_pole_ladder_boundary_sweep(; print_rows=false)
    @test Tuple(row.status for row in pole.rows) == (:accepted, :accepted, :incomplete_retention)
    @test Tuple(row.retained for row in pole.rows) == (12, 12, 10)
    @test pole.transition_gap == 0.035
    @test all(row.count == 12 for row in pole.rows)
    @test all(row.count_error <= 1e-8 for row in pole.rows)

    multiplicity = run_high_multiplicity_sine_boundary_sweep(; print_rows=false)
    @test Tuple(row.status for row in multiplicity.rows) == (
        :accepted_unique_multiplicity,
        :algebraic_count_with_spurious_retention,
        :runaway_overretention,
    )
    @test Tuple(row.power for row in multiplicity.rows) == (2, 3, 4)
    @test Tuple(row.count for row in multiplicity.rows) == (6, 9, 12)
    @test multiplicity.rows[1].retained == multiplicity.rows[1].expected_unique
    @test multiplicity.rows[2].retained > multiplicity.rows[2].expected_unique
    @test multiplicity.rows[3].retained > multiplicity.rows[3].count
    @test multiplicity.first_failure_power == 3
end

@testitem "torture moment RII: multiplicity moment escalation identifies next layer" tags=[:slow, :torture, :moment_rii, :moment_heavy] begin
    include(joinpath(@__DIR__, "..", "..", "experiments", "moment_rii", "run.jl"))

    sweep = run_high_multiplicity_moment_order_sweep(; print_rows=false)

    @test sweep.all_moment_adequate
    @test sweep.unchanged_by_moment_escalation
    @test all(row.count_error <= 1e-8 for row in sweep.rows)
    @test all(row -> row.recommendation === :moments_are_not_the_limiting_layer_use_deflation_or_retention, sweep.rows)

    p3 = filter(row -> row.power == 3, sweep.rows)
    p4 = filter(row -> row.power == 4, sweep.rows)
    @test Tuple(row.moments for row in p3) == (3, 6, 12)
    @test Tuple(row.status for row in p3) == (
        :algebraic_count_with_spurious_retention,
        :algebraic_count_with_spurious_retention,
        :algebraic_count_with_spurious_retention,
    )
    @test Tuple(row.moments for row in p4) == (4, 8, 16)
    @test Tuple(row.status for row in p4) == (
        :runaway_overretention,
        :runaway_overretention,
        :runaway_overretention,
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
