@testset "rectangular contour chart" begin
    chart = RectangularChart(0.2 - 0.1im, 2.0, 1.0, 128)
    @test sum(chart.weights) ≈ 0.0 atol=1e-14
    @test FusedNLFEAST.in_chart(chart, 2.1 - 0.1im)
    @test !FusedNLFEAST.in_chart(chart, 2.3 - 0.1im)
    @test isapprox(chart_inward_score(chart, 2.0 + 0.8im), 0.9)
    @test isapprox(chart_boundary_margin(chart, 2.0 + 0.8im), 0.1)

    roots = ComplexF64[-1.0, 0.5 + 0.3im, 3.0]
    estimate = argument_principle_count(
        z -> sum(inv(z - root) for root in roots),
        chart;
        integrality_tolerance=1e-8,
        refinement_tolerance=1e-8,
        resolution_certified=true,
    )
    @test estimate.count == 2
    @test estimate.reliable
end

@testset "chart node cover bounds the continuous boundary" begin
    circle = CircularChart(0.0, 1.0, 16)
    @test chart_node_cover_radius(circle) ≈ 2sin(pi / 32)

    rectangle = RectangularChart(0.0, 2.0, 1.0, 32)
    samples = ComplexF64[]
    for x in range(-2.0, 2.0; length=1001)
        push!(samples, x - im, x + im)
    end
    for y in range(-1.0, 1.0; length=1001)
        push!(samples, -2 + im * y, 2 + im * y)
    end
    measured = maximum(
        minimum(abs(sample - node) for node in rectangle.nodes) / rectangle.radius
        for sample in samples
    )
    @test measured <= chart_node_cover_radius(rectangle) + 1e-12
end
