@testset "argument-principle count reliability" begin
    eigenvalues = ComplexF64[-0.6 + 0.1im, 0.25 - 0.2im, 1.8]
    logarithmic_derivative = z -> sum(inv(z - value) for value in eigenvalues)
    linear_count = argument_principle_count(
        logarithmic_derivative,
        CircularChart(0.0 + 0.0im, 1.0, 64);
        integrality_tolerance=1e-8,
        refinement_tolerance=1e-8,
        resolution_certified=true,
    )
    @test linear_count.count == 2
    @test linear_count.reliable

    separated_sine = argument_principle_count(
        cot,
        CircularChart(0.0 + 0.0im, 7.5pi, 120),
        resolution_certified=true,
    )
    @test separated_sine.count == 15
    @test separated_sine.reliable

    near_boundary_sine = argument_principle_count(
        cot,
        CircularChart(0.0 + 0.0im, 25.0, 60),
    )
    @test !near_boundary_sine.reliable
    @test near_boundary_sine.refinement_error >= 0.1

    symmetric_alias = argument_principle_count(
        z -> cos(z) / sin(z),
        CircularChart(pi / 2 + 1e-5, pi / 2, 16);
        integrality_tolerance=1e-7,
        refinement_tolerance=1e-7,
    )
    @test symmetric_alias.stable
    @test !symmetric_alias.reliable
end

@testset "meromorphic count restores certified pole multiplicity" begin
    pole = 1.3
    chart = CircularChart(0.0, 4.0, 128)
    scalar = meromorphic_eigenvalue_count(
        z -> cot(z) - inv(z - pole),
        chart;
        pole_multiplicity=1,
        pole_multiplicity_certified=true,
        resolution_certified=true,
        integrality_tolerance=1e-8,
        refinement_tolerance=1e-8,
    )
    @test scalar.index.count == 2
    @test scalar.count == 3
    @test scalar.reliable

    coincident = meromorphic_eigenvalue_count(
        _ -> zero(ComplexF64),
        CircularChart(0.0, 1.0, 32);
        pole_multiplicity=1,
        pole_multiplicity_certified=true,
        resolution_certified=true,
        integrality_tolerance=1e-12,
        refinement_tolerance=1e-12,
    )
    @test coincident.index.count == 0
    @test coincident.count == 1
    @test coincident.reliable

    uncertified = meromorphic_eigenvalue_count(
        z -> cot(z) - inv(z - pole),
        chart;
        pole_multiplicity=1,
        resolution_certified=true,
    )
    @test uncertified.count == 3
    @test !uncertified.reliable
end


@testset "determinant winding count" begin
    roots = ComplexF64[-0.4, 0.2, 1.8]
    T = z -> Diagonal(z .- roots)
    chart = CircularChart(0.0, 1.0, 32)
    estimate = determinant_winding_count(T, chart; phase_resolution_certified=true)

    @test estimate.count == 2
    @test estimate.stable
    @test estimate.reliable
    @test estimate.refinement_error <= 1e-12
end

@testset "derivative bound certifies phase resolution and boundary invertibility" begin
    linear = z -> fill(ComplexF64(z - 0.2), 1, 1)
    separated = determinant_winding_count(
        linear,
        CircularChart(0.0, 1.0, 32);
        derivative_bound=(start, stop) -> 1.0,
    )
    @test separated.count == 1
    @test separated.reliable

    boundary = determinant_winding_count(
        z -> fill(ComplexF64(z), 1, 1),
        RectangularChart(0.5, 0.5, 1.0, 32);
        derivative_bound=(start, stop) -> 1.0,
    )
    @test !boundary.reliable

    resolved_power = determinant_winding_count(
        z -> fill(ComplexF64(z^150), 1, 1),
        CircularChart(0.0, 1.0, 1024);
        derivative_bound=(start, stop) -> 150.0,
    )
    @test resolved_power.count == 150
    @test resolved_power.reliable
end

@testset "determinant winding does not self-certify phase resolution" begin
    chart = CircularChart(0.0, 1.0, 64)
    aliased = determinant_winding_count(z -> fill(ComplexF64(z^150), 1, 1), chart)

    @test aliased.count == 22
    @test !aliased.reliable
end

@testset "residual convergence does not certify completeness" begin
    case = scalar_sine_case()
    result = fused_nlfeast(
        case.T,
        case.right_solve,
        case.left_solve,
        CircularChart(0.0 + 0.0im, 50.0, 124),
        ones(ComplexF64, 1, 1),
        ones(ComplexF64, 1, 1);
        config=FusedConfig(
            moment_count=31,
            iterations=0,
            ranktol=1e-10,
            residual_tol=1e-10,
            target_count=31,
        ),
        divided_overlap=case.divided_overlap,
    )
    @test result.residual_converged
    @test !result.count_certified
    @test !result.converged
    @test result.termination_reason === :count_mismatch
    @test length(result.extraction.values) < 31
end
