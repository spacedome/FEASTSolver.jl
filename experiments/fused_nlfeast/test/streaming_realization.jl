@testset "streaming TSQR matches explicit Hankel realization" begin
    rng = MersenneTwister(20260710)
    output = randn(rng, ComplexF64, 40, 4)
    state = ComplexF64[
        -0.6 0.2 0.0 0.0
        0.0 -0.1 0.1 0.0
        0.0 0.0 0.3 0.15
        0.0 0.0 0.0 0.7
    ]
    input = randn(rng, ComplexF64, 4, 2)
    moments = Matrix{ComplexF64}[output * state^power * input for power in 0:5]

    explicit = FusedNLFEAST.hankel_realization(
        moments,
        nothing,
        3;
        ranktol=1e-12,
        maxrank=4,
    )
    streaming = streaming_hankel_realization(
        moments,
        3;
        ranktol=1e-12,
        maxrank=4,
    )
    _, errors = FusedNLFEAST.bottleneck_match(
        eigvals(explicit.state),
        eigvals(streaming.state),
    )

    @test streaming.rank == explicit.rank == 4
    @test streaming.singular_values ≈ explicit.singular_values rtol=1e-12
    @test maximum(errors) <= 1e-12
    @test realization_subspace_gap(streaming.output, explicit.output) <= 1e-12
end
