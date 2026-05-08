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
    @test any(row -> row.failure_class === :branch_cut && !row.executable, summary)
    @test any(row -> row.failure_class === :dense_spectral_region && !row.executable, summary)
end
