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
