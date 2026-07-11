@testset "left/right matching minimizes the bottleneck" begin
    right = ComplexF64[0.0, 0.21]
    left = ComplexF64[0.1, -0.2]
    assignment, errors = FusedNLFEAST.bottleneck_match(right, left)

    @test assignment == [2, 1]
    @test maximum(errors) ≈ 0.2
end

@testset "continuation matching selects a candidate subset" begin
    references = ComplexF64[0.0, 1.0]
    candidates = ComplexF64[1.1, 8.0, 0.2]
    assignment, errors = FusedNLFEAST.bottleneck_subset_match(references, candidates)
    @test assignment == [3, 1]
    @test isapprox(errors, [0.2, 0.1]; atol=10eps(Float64))
end
