@testitem "nonlinear FEAST: linear pencil behaves like standard FEAST" setup=[FEASTTestSetup] begin
    using FEASTSolver
    using LinearAlgebra
    using .FEASTTestSetup: initial_subspace, assert_eigenvalues_found, assert_converged

    n = 12
    A = Matrix(Diagonal(1.0:n))
    T(z) = z * Matrix{Float64}(I, n, n) - A
    expected = complex.(1.0:4.0)
    stats = DenseFeastStats()

    λ, _, res = nlfeast!(
        T,
        initial_subspace(n, 4, 401),
        8,
        10;
        c=2.5,
        r=1.6,
        ϵ=1e-12,
        store=true,
        stats=stats,
    )

    inside = in_contour(λ, 2.5, 1.6)
    assert_eigenvalues_found(λ[inside], expected; atol=1e-10)
    assert_converged(res[inside]; atol=1e-10)
    @test stats.iterations == length(stats.iteration_log)
    @test stats.iteration_log[end].variant == :nonlinear
end

@testitem "experimental moment RII: projected SS Hankel recovers deficient quadratic" setup=[FEASTTestSetup] begin
    using FEASTSolver
    using LinearAlgebra
    using MatrixMarket
    using .FEASTTestSetup: initial_subspace, assert_converged

    A0 = Matrix{ComplexF64}(Matrix(mmread(joinpath(@__DIR__, "..", "..", "data", "quadraticM0.mtx"))))
    A1 = Matrix{ComplexF64}(Matrix(mmread(joinpath(@__DIR__, "..", "..", "data", "quadraticM1.mtx"))))
    coeffs = [A0 - 0.02 * A1, 0.1 * A1, A1]
    c, r = 0.0 + 0.0im, 0.25
    reference, _, reference_res = companion(coeffs)
    expected = reference[in_contour(reference, c, r) .& (reference_res .< 1e-7)]

    result = FEASTSolver.nlfeast_moment_rii!(
        coeffs,
        initial_subspace(size(A0, 1), 3, 961),
        16,
        5;
        c=c,
        r=r,
        moments=2,
        ranktol=1e-9,
        residual_tol=1e-8,
        left_probe=initial_subspace(size(A0, 1), 3, 962),
    )

    inside = in_contour(result.values, c, r)
    actual = result.values[inside]
    @test length(actual) == length(expected)
    @test maximum(λ -> minimum(abs.(λ .- expected)), actual) <= 1e-8
    @test maximum(λ -> minimum(abs.(λ .- actual)), expected) <= 1e-8
    assert_converged(result.residuals[inside]; atol=1e-8)
    @test result.history[end].rank == length(expected)
end

@testitem "nonlinear FEAST: sparse linear polynomial uses sparse path" setup=[FEASTTestSetup] begin
    using FEASTSolver
    using LinearAlgebra
    using SparseArrays
    using .FEASTTestSetup: initial_subspace, assert_eigenvalues_found, assert_converged

    n = 12
    T = feast_gallery(
        "polynomial",
        [spdiagm(0 => complex.(-1.0:-1.0:-n)), sparse(I, n, n)],
    )
    expected = complex.(1.0:4.0)

    for (store, seed) in ((true, 411), (false, 412))
        stats = DenseFeastStats()
        λ, _, res = nlfeast!(
            T,
            initial_subspace(n, 4, seed),
            8,
            10;
            c=2.5,
            r=1.6,
            ϵ=1e-12,
            store=store,
            stats=stats,
        )

        inside = in_contour(λ, 2.5, 1.6)
        assert_eigenvalues_found(λ[inside], expected; atol=1e-10)
        assert_converged(res[inside]; atol=1e-10)
        @test stats.iteration_log[end].variant == :sparse_nonlinear
        @test stats.stored_factor_count == (store ? 8 : 0)
        @test stats.stored_factor_bytes >= 0
    end
end

@testitem "nonlinear FEAST: custom contour on linear pencil" setup=[FEASTTestSetup] begin
    using FEASTSolver
    using LinearAlgebra
    using .FEASTTestSetup: initial_subspace, assert_eigenvalues_found, assert_converged

    n = 12
    A = Matrix(Diagonal(1.0:n))
    T(z) = z * Matrix{Float64}(I, n, n) - A
    c, r = 2.5 + 0.0im, 1.6
    base = circular_contour_trapezoidal(c, r, 8)
    contour = CustomContour(
        contour_nodes(base),
        contour_weights(base);
        inside=z -> abs(z - c) <= r,
    )
    expected = complex.(1.0:4.0)

    λ, _, res = nlfeast!(
        T,
        initial_subspace(n, 4, 402),
        contour,
        10;
        ϵ=1e-12,
        store=true,
    )

    inside = in_contour(λ, contour)
    assert_eigenvalues_found(λ[inside], expected; atol=1e-10)
    assert_converged(res[inside]; atol=1e-10)
end

@testitem "nonlinear FEAST: quadratic polynomial with exact roots" setup=[FEASTTestSetup] begin
    using FEASTSolver
    using LinearAlgebra
    using .FEASTTestSetup: initial_subspace, assert_eigenvalues_found, assert_converged

    roots1 = ComplexF64[
        0.6 + 0.05im,
        0.75,
        0.9 - 0.03im,
        2.0,
        2.3 + 0.1im,
        2.6 - 0.1im,
    ]
    roots2 = ComplexF64[-1.0, -1.2 + 0.2im, -1.4 - 0.1im, 3.0, 3.2, 3.4]
    D1 = Diagonal(roots1 .+ roots2)
    D0 = Diagonal(roots1 .* roots2)
    V = Matrix{ComplexF64}(I, length(roots1), length(roots1))
    V[1, 2] = 0.4
    V[2, 3] = -0.2im
    Vinv = inv(V)
    T(z) = V * (z^2 * I - z * D1 + D0) * Vinv
    c, r = 0.75 + 0.0im, 0.25
    expected = roots1[in_contour(roots1, c, r)]

    for (seed, store) in ((777, true), (778, false))
        λ, _, res = nlfeast!(
            T,
            initial_subspace(length(roots1), length(expected) + 1, seed),
            16,
            20;
            c=c,
            r=r,
            ϵ=1e-10,
            store=store,
        )

        inside = in_contour(λ, c, r)
        assert_eigenvalues_found(λ[inside], expected; atol=1e-9)
        assert_converged(res[inside]; atol=1e-9)
    end
end

@testitem "nonlinear FEAST: butterfly polynomial reference problem" setup=[FEASTTestSetup] begin
    using FEASTSolver
    using .FEASTTestSetup: initial_subspace, assert_eigenvalues_found, assert_converged, butterfly_polynomial_matrices

    A = butterfly_polynomial_matrices()
    T = feast_gallery("polynomial", A)
    c, r = 1.0 + 1.0im, 0.5
    reference, _, reference_res = companion(A)
    expected = reference[in_contour(reference, c, r) .& (reference_res .< 1e-8)]

    λ, _, res = nlfeast!(
        T,
        initial_subspace(size(A[1], 1), length(expected) + 4, 901),
        64,
        3;
        c=c,
        r=r,
        ϵ=1e-8,
        store=true,
        spurious=5e-3,
    )

    inside = in_contour(λ, c, r)
    assert_eigenvalues_found(λ[inside], expected; atol=1e-8)
    assert_converged(res[inside]; atol=1e-8)
end

@testitem "experimental moment RII: deficient quadratic needs lifted pair state" setup=[FEASTTestSetup] begin
    using FEASTSolver
    using LinearAlgebra
    using MatrixMarket
    using .FEASTTestSetup: initial_subspace, assert_converged

    A0 = Matrix{ComplexF64}(Matrix(mmread(joinpath(@__DIR__, "..", "..", "data", "quadraticM0.mtx"))))
    A1 = Matrix{ComplexF64}(Matrix(mmread(joinpath(@__DIR__, "..", "..", "data", "quadraticM1.mtx"))))
    coeffs = [A0 - 0.02 * A1, 0.1 * A1, A1]
    c, r = 0.0 + 0.0im, 0.25
    reference, _, reference_res = companion(coeffs)
    expected = reference[in_contour(reference, c, r) .& (reference_res .< 1e-7)]

    result = FEASTSolver.nlfeast_moment_rii!(
        coeffs,
        initial_subspace(size(A0, 1), 3, 951),
        8,
        8;
        c=c,
        r=r,
        moments=2,
        ranktol=1e-9,
        residual_tol=1e-8,
    )

    inside = in_contour(result.values, c, r)
    actual = result.values[inside]
    @test length(actual) == length(expected)
    @test maximum(λ -> minimum(abs.(λ .- expected)), actual) <= 1e-8
    @test maximum(λ -> minimum(abs.(λ .- actual)), expected) <= 1e-8
    assert_converged(result.residuals[inside]; atol=1e-8)
    @test result.history[end].rank == length(expected)
end

@testitem "experimental moment RII: local chart support diagnostics retain triangular roots" tags=[:slow] begin
    include(joinpath(@__DIR__, "..", "..", "experiments", "moment_rii", "run.jl"))

    result = run_dual_local_chart_sweep_analytic(;
        outer_radius=4.0,
        operator_builder=triangular_operator_builder(; coupling=10.0),
        operator_label="triangular(coupling=10)",
        radii=(2.4,),
        iterations=2,
        basis_ranktol=1e-8,
        basis_nodes=24,
        rii_nodes=128,
        determinant_nodes=256,
        reduced_nodes=256,
        residual_normalization=:operator,
        component_scaling=:contour_max,
        print_charts=false,
    )

    @test result.matched == length(result.expected)
    @test result.support2_matched == length(result.expected)
    @test length(result.support2_found) <= length(result.found)
    @test length(result.found) > length(result.expected)
end
