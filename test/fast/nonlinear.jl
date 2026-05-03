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
    @test result.support2_global_matched == length(result.expected)
    @test length(result.support2_global_found) == length(result.expected)
    @test length(result.support2_found) <= length(result.found)
    @test length(result.found) > length(result.expected)
end

@testitem "experimental moment RII: Loewner layout support prunes interpolation artifacts" tags=[:slow] begin
    include(joinpath(@__DIR__, "..", "..", "experiments", "moment_rii", "run.jl"))

    result = run_loewner_interpolation_sweep(;
        radii=(1.15, 1.3, 1.6),
        phase_fractions=(0.0, 0.25),
        loewner_points=6,
        print_rows=false,
    )

    @test result.initial_support[1].spurious > 0
    @test result.initial_support[2].matched == 18
    @test result.initial_support[2].spurious == 0
    @test result.updated_support[2].matched == 18
    @test result.updated_support[2].spurious == 0
end

@testitem "experimental moment RII: global Loewner layout support removes in-target artifacts" tags=[:slow] begin
    include(joinpath(@__DIR__, "..", "..", "experiments", "moment_rii", "run.jl"))

    result = run_global_loewner_interior_artifact_diagnostic(; print_rows=false)
    summary = result.summary

    @test summary.layouts == 6
    @test summary.bad_single_layouts > 0
    @test summary.worst_nearest_expected > 1e-6
    @test summary.worst_bad_residual < 1e-8
    @test summary.initial_support1_spurious > 0
    @test summary.initial_support2_spurious == 0
    @test summary.updated_support2_matched == summary.expected
    @test summary.updated_support2_count == summary.expected
    @test summary.updated_support2_spurious == 0
    @test summary.support2_prunes_artifacts
end

@testitem "experimental moment RII: global Loewner artifact policy requires cross-layout support" tags=[:slow] begin
    include(joinpath(@__DIR__, "..", "..", "experiments", "moment_rii", "run.jl"))

    result = run_global_loewner_artifact_retention_policy(; print_rows=false)
    decision = result.decision

    @test decision.status == :accept_cross_layout
    @test !decision.single_layout_safe
    @test decision.cross_layout_safe
    @test decision.retained == decision.expected
    @test :do_not_accept_single_layout_residual_small_values in decision.actions
    @test :require_cross_layout_support2 in decision.actions
    @test :retain_cross_layout_support2 in decision.actions
end

@testitem "experimental moment RII: local Loewner layouts retain exponential roots" tags=[:slow] begin
    include(joinpath(@__DIR__, "..", "..", "experiments", "moment_rii", "run.jl"))

    rows = run_exponential_local_chart_loewner_layout_sweep(;
        loewner_radii=(1.3,),
        phase_fractions=(0.0, 0.25),
        loewner_points=6,
        chart_radii=(1.2,),
        print_rows=false,
    )

    @test length(rows) == 2
    @test all(row.union_matched == row.expected for row in rows)
    @test all(row.support2_matched == row.expected for row in rows)
    @test all(row.support2 == row.expected for row in rows)
end

@testitem "experimental moment RII: grid Loewner charts recover exponential roots" tags=[:slow] begin
    include(joinpath(@__DIR__, "..", "..", "experiments", "moment_rii", "run.jl"))

    rows = run_exponential_grid_chart_loewner_spacing_sweep(;
        spacings=(1.8,),
        loewner_radius=1.3,
        loewner_points=6,
        chart_radii=(1.2, 2.0),
        print_rows=false,
    )

    @test length(rows) == 1
    @test rows[1].union_matched == rows[1].expected
    @test rows[1].support2_matched == rows[1].expected
    @test rows[1].support2 == rows[1].expected
    @test rows[1].centers > rows[1].expected
end

@testitem "experimental moment RII: adaptive grid Loewner refinement repairs weak support" tags=[:slow] begin
    include(joinpath(@__DIR__, "..", "..", "experiments", "moment_rii", "run.jl"))

    result = run_exponential_adaptive_grid_loewner_refinement(;
        base_spacing=2.4,
        loewner_radius=1.3,
        loewner_points=6,
        chart_radii=(1.2, 2.0),
        print_rows=false,
    )
    base = result.rows[1]
    refined = result.rows[end]

    @test base.union_matched == base.expected
    @test base.support2_matched < base.expected
    @test refined.support2_matched == refined.expected
    @test refined.support2 == refined.expected
    @test refined.support2_global_matched == refined.expected
    @test refined.support2_global == refined.expected
    @test refined.centers < 97
end

@testitem "experimental moment RII: adaptive grid refinement exposes triangular spurious boundary" tags=[:slow] begin
    include(joinpath(@__DIR__, "..", "..", "experiments", "moment_rii", "run.jl"))

    result = run_triangular_adaptive_grid_loewner_refinement(;
        coupling=10.0,
        outer_radius=6.0,
        base_spacing=2.4,
        chart_radii=(1.2, 1.8),
        refinement_rounds=2,
        print_rows=false,
    )
    base = result.rows[1]
    refined = result.rows[end]

    @test base.union_matched == base.expected
    @test base.support2_matched < base.expected
    @test refined.support2_matched == refined.expected
    @test refined.support2 > refined.expected
    @test refined.support2_global_matched == refined.expected
    @test refined.support2_global == refined.expected
    @test refined.centers < 57
end

@testitem "experimental moment RII: target-limited refinement repairs radius-20 analytic support" tags=[:slow] begin
    include(joinpath(@__DIR__, "..", "..", "experiments", "moment_rii", "run.jl"))

    result = run_three_function_adaptive_grid_loewner_refinement(;
        outer_radius=20.0,
        base_spacing=3.0,
        chart_radii=(1.5, 2.4),
        refinement_rounds=2,
        loewner_radius=1.3,
        loewner_points=6,
        print_rows=false,
    )
    base = result.rows[1]
    first_refined = result.rows[2]
    refined = result.rows[end]

    @test base.union_matched < base.expected
    @test first_refined.union_matched == first_refined.expected
    @test first_refined.support2_global_matched < first_refined.expected
    @test refined.support2_global_matched == refined.expected
    @test refined.support2_global == refined.expected
    @test refined.support2 > refined.expected
    @test refined.centers < 161
end

@testitem "experimental moment RII: adaptive radius-20 analytic solve is Loewner-layout stable" tags=[:slow] begin
    include(joinpath(@__DIR__, "..", "..", "experiments", "moment_rii", "run.jl"))

    result = run_three_function_adaptive_grid_loewner_layout_sweep(;
        radii=(1.15, 1.3, 1.6),
        phase_fractions=(0.0,),
        loewner_points=6,
        print_rows=false,
    )

    @test all(row.success for row in result.rows)
    @test result.summary.layouts == 3
    @test result.summary.support == 2
    @test result.summary.matched == result.summary.expected
    @test result.summary.supported == result.summary.expected
    @test result.summary.success
end

@testitem "experimental moment RII: adaptive retention score distinguishes support and local count stress" tags=[:slow] begin
    include(joinpath(@__DIR__, "..", "..", "experiments", "moment_rii", "run.jl"))

    result = run_three_function_retention_score_diagnostic(; print_rows=false)
    summary = result.summary

    @test summary.exact_support2_global
    @test summary.support1_global_matched == summary.expected
    @test summary.support2_global_matched == summary.expected
    @test summary.support2_global == summary.expected
    @test summary.support3_global < summary.expected
    @test summary.weak_inside_clusters == 0
    @test summary.count_deficit_records > 0
    @test summary.max_count_error > 1e-2
    @test summary.max_record_residual <= 1e-7
    @test result.evidence.support_and_target
    @test result.evidence.local_count_warning
    @test result.evidence.residual_ok
end

@testitem "experimental moment RII: automatic retention policy escalates count warnings to agreement checks" tags=[:slow] begin
    include(joinpath(@__DIR__, "..", "..", "experiments", "moment_rii", "run.jl"))

    result = run_three_function_automatic_retention_policy(; print_rows=false)
    initial = result.initial_decision
    final = result.final_decision

    @test initial.status == :escalate
    @test :request_loewner_layout_agreement in initial.actions
    @test :request_reduced_extractor_agreement in initial.actions
    @test final.status == :accept_with_chart_warnings
    @test final.retained == final.expected
    @test final.support_ok
    @test final.residual_ok
    @test final.local_count_warning
    @test final.layout_ok
    @test final.extractor_ok
    @test :layout_agreement_certified in final.actions
    @test :extractor_agreement_certified in final.actions
end

@testitem "experimental moment RII: adaptive radius-20 analytic solve agrees across reduced extractors" tags=[:slow] begin
    include(joinpath(@__DIR__, "..", "..", "experiments", "moment_rii", "run.jl"))

    result = run_three_function_adaptive_grid_extractor_agreement(;
        extractors=(:loewner_counted, :ss_counted),
        loewner_radius=1.3,
        loewner_points=6,
        print_rows=false,
    )

    @test all(row.success for row in result.rows)
    @test result.summary.extractors == 2
    @test result.summary.support == 2
    @test result.summary.matched == result.summary.expected
    @test result.summary.supported == result.summary.expected
    @test result.summary.success
end
