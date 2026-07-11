@testset "direct cached Loewner realization" begin
    rng = MersenneTwister(731)
    poles = ComplexF64[-0.6 + 0.1im, 0.35 - 0.2im, 0.7 + 0.05im]
    state = Diagonal(poles)
    output = randn(rng, ComplexF64, 5, 3)
    input = randn(rng, ComplexF64, 3, 2)
    observer = randn(rng, ComplexF64, 5, 2)
    left_points = ComplexF64[1.3 * cis(2pi * j / 5) for j in 0:4]
    right_points = ComplexF64[1.3 * cis(2pi * (j + 0.5) / 5) for j in 0:4]
    full_sample(point) = output * ((point .* I - state) \ input)
    full_left = full_sample.(left_points)
    full_right = full_sample.(right_points)
    projected_left = [adjoint(observer) * sample for sample in full_left]
    projected_right = [adjoint(observer) * sample for sample in full_right]
    realization = loewner_realization(
        left_points,
        projected_left,
        right_points,
        projected_right,
        full_right;
        ranktol=1e-12,
        fixed_rank=3,
    )
    @test realization.rank == 3
    @test multiset_distance(eigvals(realization.state), poles) <= 1e-11
    @test realization.singular_values[4] <= 1e-11 * realization.singular_values[1]
    extracted = realization.output * eigen(realization.state).vectors
    @test realization_subspace_gap(extracted, output) <= 1e-11
end


@testset "Loewner realization from cached contour solves" begin
    rng = MersenneTwister(732)
    poles = ComplexF64[-0.55 + 0.12im, 0.25 - 0.18im, 0.68 + 0.04im]
    similarity = randn(rng, ComplexF64, 3, 3)
    A = similarity * Diagonal(poles) / similarity
    case = linear_case(A)
    chart = CircularChart(0.0 + 0.0im, 1.0, 64)
    probe = randn(rng, ComplexF64, 3, 2)
    observer = randn(rng, ComplexF64, 3, 2)
    cache = ContourSampleCache(chart, case.right_solve, case.left_solve)
    add_right_probe!(cache, :initial, probe)
    point_count = 4
    left_points = ComplexF64[1.35 * cis(2pi * j / point_count) for j in 0:(point_count - 1)]
    right_points = ComplexF64[
        1.35 * cis(2pi * (j + 0.5) / point_count) for j in 0:(point_count - 1)
    ]
    full_left = rational_probe_samples(cache, :initial, :right, left_points)
    full_right = rational_probe_samples(cache, :initial, :right, right_points)
    realization = loewner_realization(
        left_points,
        [adjoint(observer) * sample for sample in full_left],
        right_points,
        [adjoint(observer) * sample for sample in full_right],
        full_right;
        ranktol=1e-12,
        fixed_rank=3,
    )
    values = chart.center .+ chart.radius .* eigvals(realization.state)
    @test multiset_distance(values, poles) <= 1e-10
    @test realization.rank == 3
end
