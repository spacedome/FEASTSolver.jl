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

@testitem "experimental moment RII: retention policy exposes acceptance contract" begin
    include(joinpath(@__DIR__, "..", "..", "experiments", "moment_rii", "run.jl"))

    base = (
        target_count_reliable=true,
        support2_count_complete=true,
        support3_count_complete=false,
        weak_inside_clusters=0,
        count_deficit_records=0,
        count_error_bad=0,
        max_record_residual=1e-12,
        support2_global=3,
        target_count_estimate=3,
        expected=3,
        target_count_error=0.0,
    )

    accepted = retention_policy_decision(base)
    @test accepted.status === :accept
    @test accepted.support_ok
    @test accepted.residual_ok
    @test :retain_support2_global in accepted.actions
    @test :do_not_raise_support_threshold_without_cover_density in accepted.actions

    warned_summary = merge(base, (count_error_bad=1,))
    warned = retention_policy_decision(warned_summary; layout_agreement=true, extractor_agreement=true)
    @test warned.status === :accept_with_chart_warnings
    @test warned.local_count_warning
    @test warned.layout_ok
    @test warned.extractor_ok
    @test :split_or_shrink_count_stressed_charts_before_strict_acceptance in warned.actions

    incomplete_summary = merge(base, (support2_count_complete=false, support2_global=2))
    incomplete = retention_policy_decision(incomplete_summary)
    @test incomplete.status === :refine
    @test !incomplete.support_ok
    @test :refine_weak_target_support in incomplete.actions

    high_residual_summary = merge(base, (max_record_residual=1e-4,))
    high_residual = retention_policy_decision(high_residual_summary; residual_tol=1e-8)
    @test high_residual.status === :escalate
    @test !high_residual.residual_ok
    @test :tighten_or_refine_high_residual_charts in high_residual.actions
end

@testitem "experimental moment RII: numerics config lowers to pipeline configs" begin
    include(joinpath(@__DIR__, "..", "..", "experiments", "moment_rii", "run.jl"))

    numerics = CountDrivenNumericsConfig(;
        iterations=2,
        basis_moments=5,
        basis_nodes=24,
        update_moment_count=3,
        rii_nodes=96,
        basis_ranktol=1e-7,
        residual_ranktol=1e-8,
        compression_ranktol=1e-9,
        determinant_nodes=384,
        determinant_capacity=21,
        extractor=:ss_counted,
        reduced_moments=7,
        reduced_nodes=192,
        reduced_ranktol=1e-11,
        reduced_refinement=:block_newton,
        refinement_steps=3,
        loewner_radius=1.4,
        loewner_phase=0.25,
        loewner_points=5,
        residual_normalization=:vector,
        update_mode=:moment_compressed,
        biorthogonalize=true,
    )

    configs = moment_pipeline_configs(numerics; seed=1201)
    @test configs isa MomentPipelineConfig
    @test configs.basis isa MomentBasisConfig
    @test configs.extractor isa ReducedExtractorConfig
    @test configs.update isa ResidualUpdateConfig
    @test configs.basis.moments == 5
    @test configs.basis.nodes == 24
    @test configs.basis.ranktol == 1e-7
    @test configs.basis.seed == 1201
    @test configs.basis.biorthogonalize
    @test configs.extractor.extractor === :ss_counted
    @test configs.extractor.determinant_nodes == 384
    @test configs.extractor.determinant_capacity == 21
    @test configs.extractor.reduced_moments == 7
    @test configs.extractor.reduced_nodes == 192
    @test configs.extractor.reduced_ranktol == 1e-11
    @test configs.extractor.refinement === :block_newton
    @test configs.extractor.refinement_steps == 3
    @test configs.extractor.loewner_radius == 1.4
    @test configs.extractor.loewner_phase == 0.25
    @test configs.extractor.loewner_points == 5
    @test configs.extractor.residual_normalization === :vector
    @test configs.update.moment_count == 3
    @test configs.update.rii_nodes == 96
    @test configs.update.residual_ranktol == 1e-8
    @test configs.update.compression_ranktol == 1e-9
    @test configs.update.mode === :moment_compressed
    @test configs.update.biorthogonalize
end

@testitem "experimental moment RII: pipeline configs match loose chart keywords" tags=[:slow] begin
    include(joinpath(@__DIR__, "..", "..", "experiments", "moment_rii", "run.jl"))

    cases = (scalar_rational_case(; root=0.15 + 0.05im, pole=2.5, name="contract"),)
    numerics = CountDrivenNumericsConfig(;
        iterations=1,
        basis_moments=1,
        basis_nodes=8,
        update_moment_count=1,
        rii_nodes=32,
        basis_ranktol=1e-10,
        residual_ranktol=1e-10,
        compression_ranktol=1e-10,
        determinant_nodes=128,
        determinant_capacity=8,
        extractor=:loewner_counted,
        reduced_moments=4,
        reduced_nodes=128,
        reduced_ranktol=1e-10,
        loewner_points=4,
        residual_normalization=:operator,
    )
    configs = moment_pipeline_configs(numerics; seed=1441)
    common = (
        cases=cases,
        center=0.0 + 0.0im,
        radius=0.5,
        residual_tol=1e-8,
        match_atol=1e-8,
        verbose=false,
    )

    loose = run_dual_moment_compressed_rii_analytic_iteration(;
        common...,
        iterations=numerics.iterations,
        basis_moments=numerics.basis_moments,
        basis_nodes=numerics.basis_nodes,
        update_moment_count=numerics.update_moment_count,
        rii_nodes=numerics.rii_nodes,
        basis_ranktol=numerics.basis_ranktol,
        residual_ranktol=numerics.residual_ranktol,
        compression_ranktol=numerics.compression_ranktol,
        determinant_nodes=numerics.determinant_nodes,
        determinant_capacity=numerics.determinant_capacity,
        extractor=numerics.extractor,
        reduced_moments=numerics.reduced_moments,
        reduced_nodes=numerics.reduced_nodes,
        reduced_ranktol=numerics.reduced_ranktol,
        loewner_points=numerics.loewner_points,
        residual_normalization=numerics.residual_normalization,
        basis_config=configs.basis,
    )
    configured = run_dual_moment_compressed_rii_analytic_iteration(;
        common...,
        iterations=numerics.iterations,
        pipeline_config=configs,
    )

    loose_good = good_extraction_values(loose.extraction; residual_tol=1e-8)
    configured_good = good_extraction_values(configured.extraction; residual_tol=1e-8)
    @test length(loose_good) == 1
    @test length(configured_good) == 1
    @test abs(loose_good[1] - configured_good[1]) <= 1e-10
    @test loose.summaries[end].matched == configured.summaries[end].matched
    @test loose.summaries[end].good == configured.summaries[end].good
end

@testitem "experimental moment RII: linear SS-FEAST reduces to FEAST residual correction" tags=[:slow] begin
    include(joinpath(@__DIR__, "..", "..", "experiments", "moment_rii", "run.jl"))

    result = run_linear_ss_feast_control(
        "many_eigenvalue_diagonal",
        many_eigenvalue_diagonal_matrix;
        nodes=32,
        iterations=4,
        moment_count=3,
        ranktol=1e-11,
        residual_tol=1e-10,
        target_count=true,
    )

    @test result.expected == 10
    @test result.probe_cols == 4
    @test result.feast_returned < result.expected
    @test result.wide_feast_converged == result.expected
    @test result.initial.rank == result.expected
    @test result.initial.converged_inside == 0
    @test result.final.rank == result.expected
    @test result.final.converged_inside == result.expected
    @test result.final.max_inside_residual <= 1e-10
    @test result.final.pair_residual < result.initial.pair_residual
end

@testitem "experimental moment RII: dual linear RII reduces to FEAST contour filter" tags=[:slow] begin
    include(joinpath(@__DIR__, "..", "..", "experiments", "moment_rii", "run.jl"))

    result = run_linear_dual_rii_reduction_diagnostic(; print_rows=false)

    @test length(result.rows) == 2
    @test result.max_projection_gap <= 1e-12
    for row in result.rows
        @test row.extracted > 0
        @test row.right_projection_gap <= 1e-12
        @test row.left_projection_gap <= 1e-12
        @test row.right_relative_error <= 1e-12
        @test row.left_relative_error <= 1e-12
        @test row.right_residual_rank == row.extracted
        @test row.left_residual_rank == row.extracted
    end
end

@testitem "experimental moment RII: polynomial bridge agrees with companion FEAST" tags=[:slow] begin
    include(joinpath(@__DIR__, "..", "..", "experiments", "moment_rii", "run.jl"))

    result = run_polynomial_family_bridge_diagnostic(; print_rows=false)
    companion = result.companion
    rows = result.polynomial_rows
    distances = result.companion_row_distances

    @test companion.expected == 20
    @test companion.companion_size == 32
    @test companion.matched == companion.expected
    @test companion.good >= companion.expected
    @test companion.nearest_expected <= 1e-8
    @test companion.max_poly_residual <= 1e-8
    @test length(rows) == 3
    @test length(distances) == length(rows)
    @test [row.stage for row in rows] == [:initial_extraction, :block_newton_cleanup, :laurent_update]
    @test [distance.stage for distance in distances] == [row.stage for row in rows]
    @test all(row.expected == companion.expected for row in rows)
    @test all(row.matched == companion.expected for row in rows)
    @test all(row.spurious == 0 for row in rows)
    @test all(row.max_residual <= 1e-8 for row in rows)
    @test all(distance.nearest_companion <= 1e-6 for distance in distances)
end

@testitem "experimental moment RII: dual reduced extraction rejects one-sided false Ritz data" tags=[:slow] begin
    include(joinpath(@__DIR__, "..", "..", "experiments", "moment_rii", "run.jl"))

    result = run_dual_reduced_polynomial_control(;
        name="dual_sensitive",
        make_problem=dual_sensitive_polynomial_problem,
        residual_tol=1e-6,
        match_atol=1e-3,
        print_rows=false,
    )

    dual = result.rows[findfirst(row -> row.mode === :dual && row.refinement === :none, result.rows)]
    biorth = result.rows[findfirst(row -> row.mode === :dual_biorth && row.refinement === :none, result.rows)]
    galerkin = result.rows[findfirst(row -> row.mode === :galerkin && row.refinement === :none, result.rows)]

    @test result.expected == 12
    @test dual.matched == result.expected
    @test biorth.matched == result.expected
    @test dual.spurious_good == 0
    @test galerkin.returned_inside > result.expected
    @test galerkin.matched == 0
    @test galerkin.good == 0
    @test galerkin.reduced_max <= 1e-12
    @test galerkin.original_max > 1e-3
end

@testitem "experimental moment RII: dual residual Laurent update needs both sides" tags=[:slow] begin
    include(joinpath(@__DIR__, "..", "..", "experiments", "moment_rii", "run.jl"))

    result = run_dual_residual_laurent_two_sided_update_control(; print_rows=false)

    initial = result.rows[findfirst(row -> row.stage === :initial, result.rows)]
    two_sided = result.rows[findfirst(row -> row.stage === :two_sided_residual_laurent, result.rows)]
    right_only = result.rows[findfirst(row -> row.stage === :right_only_truncated, result.rows)]
    left_only = result.rows[findfirst(row -> row.stage === :left_only_truncated, result.rows)]

    @test result.expected == 12
    @test result.initial_basis == (right=3, left=3)
    @test result.updated_basis == (right=6, left=6)
    @test initial.matched == 0
    @test two_sided.matched == result.expected
    @test two_sided.spurious_good == 0
    @test two_sided.max_residual <= 1e-12
    @test right_only.truncated
    @test left_only.truncated
    @test right_only.raw_right_basis > right_only.raw_left_basis
    @test left_only.raw_left_basis > left_only.raw_right_basis
    @test right_only.matched == initial.matched
    @test left_only.matched == initial.matched
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

@testitem "experimental moment RII: sparse linear operator flows through moment pipeline" tags=[:slow] begin
    include(joinpath(@__DIR__, "..", "..", "experiments", "moment_rii", "run.jl"))

    result = run_sparse_linear_moment_pipeline_smoke(; print_rows=false)

    @test result.sparse_matrix
    @test result.expected == 8
    @test result.initial.inside == result.expected
    @test result.initial.matched == 0
    @test result.initial.max_residual > 1e-10
    @test result.updated.matched == result.expected
    @test result.updated.spurious_good == 0
    @test result.updated.max_residual <= 1e-10
    @test result.updated.right_residual_rank <= result.expected
    @test result.updated.left_residual_rank <= result.expected
end

@testitem "experimental moment RII: sparse stored factors reproduce residual Laurent update" tags=[:slow] begin
    include(joinpath(@__DIR__, "..", "..", "experiments", "moment_rii", "run.jl"))

    result = run_sparse_stored_factor_residual_laurent_smoke(; print_rows=false)

    @test result.sparse_matrix
    @test result.expected == 8
    @test result.direct.matched == result.expected
    @test result.cached.matched == result.expected
    @test result.cached.spurious_good == 0
    @test result.x_projection_gap <= 1e-12
    @test result.y_projection_gap <= 1e-12
    @test result.repeat_x_projection_gap <= 1e-12
    @test result.repeat_y_projection_gap <= 1e-12
    @test result.after_first.right_factorizations == 64
    @test result.after_first.left_factorizations == 64
    @test result.factorization_reuse.right_factorizations_constant
    @test result.factorization_reuse.left_factorizations_constant
    @test result.factorization_reuse.right_second_solves == 64
    @test result.factorization_reuse.left_second_solves == 64
    @test result.cached_stats.right_candidate_cols == result.direct_stats.right_candidate_cols
    @test result.cached_stats.left_candidate_cols == result.direct_stats.left_candidate_cols
end

@testitem "experimental moment RII: sparse symbolic reuse reproduces residual Laurent update" tags=[:slow] begin
    include(joinpath(@__DIR__, "..", "..", "experiments", "moment_rii", "run.jl"))

    result = run_sparse_symbolic_reuse_residual_laurent_smoke(; print_rows=false)

    @test result.sparse_matrix
    @test result.prototype_sparse
    @test result.expected == 6
    @test result.direct.matched == result.expected
    @test result.symbolic.matched == result.expected
    @test result.symbolic.spurious_good == 0
    @test result.x_projection_gap <= 5e-12
    @test result.y_projection_gap <= 5e-12
    @test result.repeat_x_projection_gap <= 5e-12
    @test result.repeat_y_projection_gap <= 5e-12
    @test result.after_first.right_symbolic_initializations == 1
    @test result.after_first.left_symbolic_initializations == 1
    @test result.after_second.right_symbolic_initializations == 1
    @test result.after_second.left_symbolic_initializations == 1
    @test result.after_first.right_numeric_refactors == 95
    @test result.after_first.left_numeric_refactors == 95
    @test result.after_second.right_numeric_refactors == 191
    @test result.after_second.left_numeric_refactors == 191
    @test result.after_first.right_solves == 96
    @test result.after_first.left_solves == 96
    @test result.after_second.right_solves == 192
    @test result.after_second.left_solves == 192
    @test result.after_first.right_solution_buffers == 1
    @test result.after_first.left_solution_buffers == 1
    @test result.after_second.right_solution_buffers == 1
    @test result.after_second.left_solution_buffers == 1
    @test result.symbolic_stats.right_candidate_cols == result.direct_stats.right_candidate_cols
    @test result.symbolic_stats.left_candidate_cols == result.direct_stats.left_candidate_cols
end

@testitem "experimental moment RII: sparse nonlinear gallery operator flows through moment pipeline" tags=[:slow] begin
    include(joinpath(@__DIR__, "..", "..", "experiments", "moment_rii", "run.jl"))

    result = run_sparse_nonlinear_gallery_moment_pipeline_smoke(; print_rows=false)

    @test result.sparse_matrix
    @test result.derivative_sparse
    @test result.prototype_sparse
    @test result.expected == 6
    @test result.updated.matched == result.expected
    @test result.updated.spurious_good == 0
    @test result.updated.max_residual <= 1e-10
    @test result.updated.right_residual_rank <= 2 * result.expected
    @test result.updated.left_residual_rank <= 2 * result.expected
end

@testitem "experimental moment RII: sparse Schrodinger gallery count and residual repair" tags=[:slow] begin
    include(joinpath(@__DIR__, "..", "..", "experiments", "moment_rii", "run.jl"))

    result = run_sparse_schrodinger_moment_gallery_smoke(; print_rows=false)

    @test result.sparse_matrix
    @test result.prototype_sparse
    @test result.target_count == 3
    @test result.target_count_reliable
    @test result.initial.inside == result.target_count
    @test result.updated.inside == result.target_count
    @test result.updated.good == result.target_count
    @test result.updated.max_inside_residual <= 1e-7
    @test result.updated.max_inside_residual < result.initial.max_inside_residual
    @test result.updated.right_residual_rank == result.target_count
    @test result.updated.left_residual_rank == result.target_count
end

@testitem "experimental moment RII: sparse nonlinear remote workers reuse stored contour factors" tags=[:slow, :distributed] begin
    if !isdefined(Main, :run_sparse_nonlinear_remote_stored_factor_worker_smoke)
        Base.include(Main, joinpath(@__DIR__, "..", "..", "experiments", "moment_rii", "run.jl"))
    end

    result = Main.run_sparse_nonlinear_remote_stored_factor_worker_smoke(; worker_count=2, print_rows=false)

    @test result.sparse_matrix
    @test result.prototype_sparse
    @test result.expected == 6
    @test length(result.workers) == 2
    @test result.serial.matched == result.expected
    @test result.remote.matched == result.expected
    @test result.remote.spurious_good == 0
    @test result.x_projection_gap <= 5e-12
    @test result.y_projection_gap <= 5e-12
    @test result.repeat_x_projection_gap <= 5e-12
    @test result.repeat_y_projection_gap <= 5e-12
    @test sum(length, result.assignments) == 96
    @test result.first_worker_factorizations == 192
    @test result.second_worker_factorizations == result.first_worker_factorizations
    @test result.first_worker_solution_buffers == result.first_worker_factorizations
    @test result.second_worker_solution_buffers == result.first_worker_solution_buffers
    @test result.first_worker_solves == 192
    @test result.second_worker_solves == 2 * result.first_worker_solves
end

@testitem "experimental moment RII: sparse Schrodinger remote workers reuse stored contour factors" tags=[:slow, :distributed] begin
    if !isdefined(Main, :run_sparse_schrodinger_remote_stored_factor_worker_smoke)
        Base.include(Main, joinpath(@__DIR__, "..", "..", "experiments", "moment_rii", "run.jl"))
    end

    result = Main.run_sparse_schrodinger_remote_stored_factor_worker_smoke(; worker_count=2, print_rows=false)

    @test result.sparse_matrix
    @test result.prototype_sparse
    @test result.derivative_sparse
    @test result.target_count == 3
    @test result.target_count_reliable
    @test length(result.workers) == 2
    @test result.serial.matched == result.target_count
    @test result.remote.matched == result.target_count
    @test result.remote.spurious_good == 0
    @test result.x_projection_gap <= 1e-10
    @test result.y_projection_gap <= 1e-10
    @test result.repeat_x_projection_gap <= 1e-10
    @test result.repeat_y_projection_gap <= 1e-10
    @test sum(length, result.assignments) == 48
    @test result.first_worker_factorizations == 96
    @test result.second_worker_factorizations == result.first_worker_factorizations
    @test result.first_worker_solution_buffers == result.first_worker_factorizations
    @test result.second_worker_solution_buffers == result.first_worker_solution_buffers
    @test result.first_worker_solves == 96
    @test result.second_worker_solves == 2 * result.first_worker_solves
    @test result.setup_elapsed_ns > 0
    @test result.serial_update_elapsed_ns > 0
    @test result.remote_first_elapsed_ns > 0
    @test result.remote_second_elapsed_ns > 0
    @test result.remote_repeats == 2
    @test length(result.remote_elapsed_ns) == result.remote_repeats
    @test result.remote_steady_min_elapsed_ns > 0
    @test result.remote_steady_mean_elapsed_ns > 0
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

@testitem "experimental moment RII: rational coordinates do not replace Loewner realization" tags=[:slow, :moment_heavy] begin
    include(joinpath(@__DIR__, "..", "..", "experiments", "moment_rii", "run.jl"))

    result = run_rational_coordinate_boundary_diagnostic(; print_rows=false)

    @test result.loewner.success
    @test result.inverse.matched == 0
    @test result.mobius_plus.matched < result.mobius_plus.expected
    @test result.mobius_minus.matched == result.mobius_minus.expected
    @test result.mobius_minus.spurious > 0
    @test !result.mobius_minus.success
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

@testitem "experimental moment RII: target-limited refinement repairs radius-20 analytic support" tags=[:slow, :moment_heavy] begin
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

@testitem "experimental moment RII: count-driven refinement stops without exact roots" tags=[:slow] begin
    include(joinpath(@__DIR__, "..", "..", "experiments", "moment_rii", "run.jl"))

    result = run_three_function_count_driven_adaptive_refinement(; print_rows=false)
    base = result.rows[1]
    final = result.rows[end]

    @test result.count.count_estimate == result.count.expected
    @test result.count.count_error <= 1e-8
    @test result.stop_reason == :target_count_complete
    @test result.algebraic_retained_count == result.count.count_estimate
    @test isempty(result.multiplicities)
    @test !base.count_complete
    @test final.count_complete
    @test final.retained == final.target_count
    @test final.validation_success
    @test final.validation_matched == final.expected
    @test length(result.rows) == 3
end

@testitem "experimental moment RII: count-driven refinement accounts for multiplicity" tags=[:slow] begin
    include(joinpath(@__DIR__, "..", "..", "experiments", "moment_rii", "run.jl"))

    result = run_triangular_count_driven_adaptive_refinement(; print_rows=false)
    diagnostic = count_driven_chart_diagnostic_summary(result; outer_radius=6.0)
    base = result.rows[1]
    final = result.rows[end]

    @test result.count.count_estimate == result.count.expected + 1
    @test result.count.count_error <= 1e-2
    @test result.stop_reason == :target_algebraic_count_complete
    @test result.algebraic_retained_count == result.count.count_estimate
    @test count(item -> item.multiplicity == 2, result.multiplicities) == 1
    @test count(item -> item.multiplicity == 1, result.multiplicities) == final.retained - 1
    @test !base.count_complete
    @test !final.count_complete
    @test final.retained + 1 == final.target_count
    @test final.validation_success
    @test final.validation_matched == final.expected
    @test length(result.rows) == 3
    @test diagnostic.base.union_good > diagnostic.base.retained
    @test diagnostic.base.weak_inside_clusters > 0
    @test diagnostic.base.selected_count_error_bad > 0
    @test diagnostic.final.retained == final.retained
end

@testitem "experimental moment RII: count-driven refinement handles repeated analytic roots" tags=[:slow] begin
    include(joinpath(@__DIR__, "..", "..", "experiments", "moment_rii", "run.jl"))

    result = run_squared_sine_count_driven_adaptive_refinement(; print_rows=false)
    base = result.rows[1]
    final = result.rows[end]

    @test result.count.count_estimate == 2 * result.count.expected
    @test result.count.count_error <= 1e-8
    @test result.stop_reason == :target_algebraic_count_complete
    @test result.algebraic_retained_count == result.count.count_estimate
    @test all(item.multiplicity == 2 for item in result.multiplicities)
    @test !base.count_complete
    @test !final.count_complete
    @test final.retained == final.expected
    @test final.validation_success
    @test final.validation_matched == final.expected
    @test length(result.rows) == 2
end

@testitem "experimental moment RII: count-driven refinement works without root oracle" tags=[:slow] begin
    include(joinpath(@__DIR__, "..", "..", "experiments", "moment_rii", "run.jl"))

    result = run_delay_count_driven_adaptive_refinement(; print_rows=false)
    final = result.rows[end]

    @test result.count.expected == 0
    @test result.count.count_estimate == 3
    @test result.count.count_error <= 1e-8
    @test result.stop_reason == :target_count_complete
    @test result.algebraic_retained_count == result.count.count_estimate
    @test isempty(result.multiplicities)
    @test final.count_complete
    @test final.retained == result.count.count_estimate
    @test final.target_count == result.count.count_estimate
    @test final.validation_matched == 0
    @test length(result.rows) == 1
end

@testitem "experimental moment RII: count-driven refinement swaps reduced extractor without root oracle" tags=[:slow] begin
    include(joinpath(@__DIR__, "..", "..", "experiments", "moment_rii", "run.jl"))

    policy = CountDrivenPolicyConfig(;
        base_spacing=1.8,
        chart_radii=(1.2, 2.0),
    )
    numerics = CountDrivenNumericsConfig(;
        basis_moments=6,
        basis_nodes=48,
        determinant_capacity=64,
        extractor=:ss_counted,
        reduced_moments=12,
        reduced_nodes=512,
        component_scaling=:none,
    )
    diagnostic = run_count_driven_policy_diagnostic(;
        label="SS extractor no-oracle delay",
        cases=(scalar_delay_case(),),
        outer_radius=6.0,
        policy=policy,
        numerics=numerics,
        print_rows=false,
    )
    result = diagnostic.result
    final = result.rows[end]

    @test result.count.expected == 0
    @test result.count.count_estimate == 3
    @test result.count.count_error <= 1e-8
    @test result.stop_reason == :target_count_complete
    @test result.algebraic_retained_count == result.count.count_estimate
    @test isempty(result.multiplicities)
    @test final.count_complete
    @test final.retained == result.count.count_estimate
    @test final.validation_matched == 0
    @test diagnostic.diagnostic.final.retained == result.count.count_estimate
end

@testitem "experimental moment RII: count-driven refinement agrees across extractors without root oracle" tags=[:slow] begin
    include(joinpath(@__DIR__, "..", "..", "experiments", "moment_rii", "run.jl"))

    result = run_delay_count_driven_extractor_agreement(; print_rows=false)

    @test result.summary.extractors == 2
    @test result.summary.target_count == 3
    @test result.summary.supported == result.summary.target_count
    @test result.summary.success
    @test all(row.stop_reason === :target_count_complete for row in result.rows)
    @test all(row.retained == result.summary.target_count for row in result.rows)
    @test all(row.algebraic_retained == result.summary.target_count for row in result.rows)
    @test all(row.count_error <= 1e-8 for row in result.rows)
end

@testitem "experimental moment RII: count-driven refinement handles oracle-free nonnormal delay" tags=[:slow] begin
    include(joinpath(@__DIR__, "..", "..", "experiments", "moment_rii", "run.jl"))

    result = run_multi_delay_count_driven_adaptive_refinement(;
        base_spacing=2.4,
        chart_radii=(1.0, 1.6),
        max_refinement_rounds=2,
        print_rows=false,
    )
    diagnostic = count_driven_chart_diagnostic_summary(result; outer_radius=6.0)
    final = result.rows[end]

    @test result.count.expected == 0
    @test result.count.count_estimate == 9
    @test result.count.count_error <= 1e-8
    @test result.stop_reason == :target_count_complete
    @test result.algebraic_retained_count == result.count.count_estimate
    @test isempty(result.multiplicities)
    @test final.count_complete
    @test final.retained == result.count.count_estimate
    @test final.target_count == result.count.count_estimate
    @test final.validation_matched == 0
    @test length(result.rows) == 2
    @test diagnostic.base.union_good == result.count.count_estimate
    @test diagnostic.base.retained < result.count.count_estimate
    @test diagnostic.base.weak_inside_clusters > 0
    @test diagnostic.final.retained == result.count.count_estimate
end

@testitem "experimental moment RII: count-driven refinement handles oracle-free two-delay scalar" tags=[:slow] begin
    include(joinpath(@__DIR__, "..", "..", "experiments", "moment_rii", "run.jl"))

    result = run_two_delay_count_driven_adaptive_refinement(; print_rows=false)
    final = result.rows[end]

    @test result.count.expected == 0
    @test result.count.count_estimate == 3
    @test result.count.count_error <= 1e-8
    @test result.stop_reason == :target_count_complete
    @test result.algebraic_retained_count == result.count.count_estimate
    @test isempty(result.multiplicities)
    @test final.count_complete
    @test final.retained == result.count.count_estimate
    @test final.target_count == result.count.count_estimate
    @test final.validation_matched == 0
    @test length(result.rows) == 1
end

@testitem "experimental moment RII: count-driven refinement handles oracle-free coupled two-delay" tags=[:slow] begin
    include(joinpath(@__DIR__, "..", "..", "experiments", "moment_rii", "run.jl"))

    result = run_coupled_two_delay_count_driven_adaptive_refinement(; print_rows=false)
    final = result.rows[end]

    @test result.count.expected == 0
    @test result.count.count_estimate == 6
    @test result.count.count_error <= 1e-6
    @test result.stop_reason == :target_count_complete
    @test result.algebraic_retained_count == result.count.count_estimate
    @test isempty(result.multiplicities)
    @test final.count_complete
    @test final.retained == result.count.count_estimate
    @test final.target_count == result.count.count_estimate
    @test final.validation_matched == 0
    @test length(result.rows) == 1
end

@testitem "experimental moment RII: count-driven refinement handles fully coupled mixed diagnostics" tags=[:slow] begin
    include(joinpath(@__DIR__, "..", "..", "experiments", "moment_rii", "run.jl"))

    stress = run_coupled_two_delay_mixed_policy_stress(; print_rows=false)
    result = stress.result
    diagnostic = stress.diagnostic
    final = result.rows[end]

    @test result.count.expected == 0
    @test result.count.count_estimate == 16
    @test result.count.count_error <= 1e-8
    @test result.stop_reason == :target_count_complete
    @test result.algebraic_retained_count == result.count.count_estimate
    @test final.count_complete
    @test final.retained == result.count.count_estimate
    @test length(result.rows) == 2
    @test length(result.added_centers) == 1
    @test diagnostic.base.union_good > diagnostic.base.retained
    @test diagnostic.base.retained < result.count.count_estimate
    @test diagnostic.base.weak_inside_clusters > 0
    @test diagnostic.base.selected_count_error_bad > 0
    @test diagnostic.final.union_good > diagnostic.final.retained
    @test diagnostic.final.weak_inside_clusters == 0
    @test diagnostic.final.retained == result.count.count_estimate
end

@testitem "experimental moment RII: count-driven refinement handles dense multi-delay weak support" tags=[:slow] begin
    include(joinpath(@__DIR__, "..", "..", "experiments", "moment_rii", "run.jl"))

    stress = run_dense_multi_delay_weak_support_stress(; print_rows=false)
    result = stress.result
    diagnostic = stress.diagnostic
    final = result.rows[end]

    @test result.count.expected == 0
    @test result.count.count_estimate == 9
    @test result.count.count_error <= 1e-8
    @test result.stop_reason == :target_count_complete
    @test result.algebraic_retained_count == result.count.count_estimate
    @test final.count_complete
    @test final.retained == result.count.count_estimate
    @test length(result.rows) == 2
    @test length(result.added_centers) == 4
    @test diagnostic.base.union_good > result.count.count_estimate
    @test diagnostic.base.retained < result.count.count_estimate
    @test diagnostic.base.weak_inside_clusters > 0
    @test diagnostic.final.union_good > diagnostic.final.retained
    @test diagnostic.final.weak_inside_clusters == 0
    @test diagnostic.final.retained == result.count.count_estimate
end

@testitem "experimental moment RII: count-driven refinement diagnoses sparse coupled cover" tags=[:slow] begin
    include(joinpath(@__DIR__, "..", "..", "experiments", "moment_rii", "run.jl"))

    result = run_coupled_two_delay_count_driven_adaptive_refinement(;
        coupling=1.0,
        outer_radius=12.0,
        base_spacing=4.0,
        chart_radii=(1.2, 2.0),
        max_refinement_rounds=4,
        print_rows=false,
    )
    final = result.rows[end]

    @test result.count.expected == 0
    @test result.count.count_estimate == 12
    @test result.stop_reason == :count_multiplicity_or_unresolved_defect
    @test final.retained < result.count.count_estimate
    @test result.algebraic_retained_count < result.count.count_estimate
    @test !final.count_complete
    @test length(result.rows) > 1
    @test !isempty(result.added_centers)
end

@testitem "experimental moment RII: count-driven refinement repairs sparse coupled cover with larger charts" tags=[:slow] begin
    include(joinpath(@__DIR__, "..", "..", "experiments", "moment_rii", "run.jl"))

    result = run_coupled_two_delay_count_driven_adaptive_refinement(;
        coupling=1.0,
        outer_radius=12.0,
        base_spacing=4.0,
        chart_radii=(1.2, 3.0),
        max_refinement_rounds=4,
        print_rows=false,
    )
    final = result.rows[end]

    @test result.count.expected == 0
    @test result.count.count_estimate == 12
    @test result.stop_reason == :target_count_complete
    @test result.algebraic_retained_count == result.count.count_estimate
    @test final.count_complete
    @test final.retained == result.count.count_estimate
    @test length(result.rows) == 2
    @test length(result.added_centers) == 1
end

@testitem "experimental moment RII: count-driven refinement radius ladder repairs sparse coupled cover" tags=[:slow] begin
    include(joinpath(@__DIR__, "..", "..", "experiments", "moment_rii", "run.jl"))

    result = run_coupled_two_delay_radius_ladder_refinement(; print_rows=false)
    first_stage = result.stages[1].result
    final = result.rows[end]

    @test length(result.stages) == 2
    @test first_stage.stop_reason == :count_multiplicity_or_unresolved_defect
    @test first_stage.rows[end].retained == 10
    @test result.stop_reason == :target_count_complete
    @test result.count.count_estimate == 12
    @test result.algebraic_retained_count == result.count.count_estimate
    @test final.count_complete
    @test final.retained == result.count.count_estimate
end

@testitem "experimental moment RII: canonical NLFEAST limit agrees with scalar and compressed updates" tags=[:slow] begin
    include(joinpath(@__DIR__, "..", "..", "experiments", "moment_rii", "run.jl"))

    result = run_canonical_nlfeast_limit_diagnostic(; print_rows=false)

    @test length(result.expected) == 3
    for summary in (result.canonical_summary, result.scalar_summary, result.compressed_summary)
        @test summary.good == length(result.expected)
        @test summary.matched == length(result.expected)
        @test summary.spurious_good == 0
        @test summary.max_residual <= 1e-8
    end
end

@testitem "experimental moment RII: count-driven refinement handles oracle-free multiplicity" tags=[:slow] begin
    include(joinpath(@__DIR__, "..", "..", "experiments", "moment_rii", "run.jl"))

    result = run_duplicate_delay_count_driven_adaptive_refinement(; print_rows=false)
    final = result.rows[end]

    @test result.count.expected == 0
    @test result.count.count_estimate == 6
    @test result.count.count_error <= 1e-8
    @test result.stop_reason == :target_algebraic_count_complete
    @test result.algebraic_retained_count == result.count.count_estimate
    @test final.retained == 3
    @test !final.count_complete
    @test all(item.multiplicity == 2 for item in result.multiplicities)
    @test length(result.multiplicities) == final.retained
    @test final.validation_matched == 0
    @test length(result.rows) == 1
end

@testitem "experimental moment RII: count-driven refinement handles oracle-free near-pole rational" tags=[:slow] begin
    include(joinpath(@__DIR__, "..", "..", "experiments", "moment_rii", "run.jl"))

    result = run_near_pole_rational_count_driven_adaptive_refinement(; print_rows=false)
    final = result.rows[end]

    @test result.count.expected == 0
    @test result.count.count_estimate == 5
    @test result.count.count_error <= 1e-6
    @test result.stop_reason == :target_count_complete
    @test result.algebraic_retained_count == result.count.count_estimate
    @test isempty(result.multiplicities)
    @test final.count_complete
    @test final.retained == result.count.count_estimate
    @test final.target_count == result.count.count_estimate
    @test final.validation_matched == 0
    @test length(result.rows) == 1
end

@testitem "experimental moment RII: near-pole rational count diagnostic rejects too-close poles" tags=[:slow] begin
    include(joinpath(@__DIR__, "..", "..", "experiments", "moment_rii", "run.jl"))

    result = run_near_pole_rational_count_driven_adaptive_refinement(; gap=0.005, print_rows=false)

    @test result.count.expected == 0
    @test result.count.count_estimate == 5
    @test result.count.count_error > 1e-6
    @test result.stop_reason == :target_count_unreliable
    @test result.algebraic_retained_count == 0
    @test isempty(result.multiplicities)
end

@testitem "experimental moment RII: adaptive radius-20 analytic solve is Loewner-layout stable" tags=[:slow, :moment_heavy] begin
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

@testitem "experimental moment RII: adaptive retention score distinguishes support and local count stress" tags=[:slow, :moment_heavy] begin
    include(joinpath(@__DIR__, "..", "..", "experiments", "moment_rii", "run.jl"))

    result = run_three_function_retention_score_diagnostic(; print_rows=false)
    summary = result.summary

    @test summary.exact_support2_global
    @test summary.target_count_estimate == summary.expected
    @test summary.target_count_error <= 1e-8
    @test summary.target_count_reliable
    @test summary.support1_global_matched == summary.expected
    @test summary.support2_global_matched == summary.expected
    @test summary.support2_global == summary.expected
    @test summary.support2_count_complete
    @test summary.support3_global < summary.expected
    @test !summary.support3_count_complete
    @test summary.weak_inside_clusters == 0
    @test summary.count_deficit_records > 0
    @test summary.selected_count_deficit_records > 0
    @test summary.max_count_error > 1e-2
    @test summary.max_record_residual <= 1e-7
    @test result.plan.retained_count == summary.expected
    @test result.plan.weak_target_count == 0
    @test result.plan.count_stressed_count == summary.selected_count_deficit_records
    @test :split_or_shrink_count_stressed_charts in result.plan.actions
    @test :do_not_raise_support_threshold_without_cover_density in result.plan.actions
    @test result.evidence.support_and_target
    @test result.evidence.oracle_free_count_complete
    @test result.evidence.local_count_warning
    @test result.evidence.residual_ok
end

@testitem "experimental moment RII: automatic retention policy escalates count warnings to agreement checks" tags=[:slow, :moment_heavy] begin
    include(joinpath(@__DIR__, "..", "..", "experiments", "moment_rii", "run.jl"))

    result = run_three_function_automatic_retention_policy(; print_rows=false)
    initial = result.initial_decision
    final = result.final_decision

    @test initial.status == :escalate
    @test :request_loewner_layout_agreement in initial.actions
    @test :request_reduced_extractor_agreement in initial.actions
    @test :do_not_raise_support_threshold_without_cover_density in initial.actions
    @test :split_or_shrink_count_stressed_charts_before_strict_acceptance in initial.actions
    @test final.status == :accept_with_chart_warnings
    @test final.retained == final.expected
    @test final.expected == final.validation_expected
    @test final.target_count_error <= 1e-8
    @test final.support_ok
    @test final.residual_ok
    @test final.local_count_warning
    @test final.layout_ok
    @test final.extractor_ok
    @test :layout_agreement_certified in final.actions
    @test :extractor_agreement_certified in final.actions
    @test :split_or_shrink_count_stressed_charts_before_strict_acceptance in final.actions
end

@testitem "experimental moment RII: candidate-centered split repairs count-stressed chart" tags=[:slow, :moment_heavy] begin
    include(joinpath(@__DIR__, "..", "..", "experiments", "moment_rii", "run.jl"))

    result = run_count_stressed_split_probe(; print_rows=false)

    @test result.base.plan.count_stressed_count > 0
    @test length(result.child_centers) == 4
    @test result.child_radius == result.parent.radius / 2
    @test length(result.candidate_centers) == result.parent.good
    @test length(result.refinement_centers) > length(result.candidate_centers)
    @test result.parent.good < result.parent.count_estimate
    @test result.parent.stress === :count_deficit
    @test result.parent.action === :shrink_around_residual_candidates
    @test result.parent.rationale === :missing_local_count
    @test length(result.naive_child.found) == 0
    @test result.naive_child.matched == 0
    @test result.candidate_child.matched == length(result.candidate_child.expected)
    @test result.candidate_child.support2_global_matched == length(result.candidate_child.expected)
    @test all(row.naive_child.matched == 0 for row in result.rows)
    @test all(row.candidate_child.matched == row.expected for row in result.rows)
    @test all(row.candidate_child.support2_global_matched == row.expected for row in result.rows)
end

@testitem "experimental moment RII: count-error-only split preserves nonnormal chart radius" tags=[:slow, :moment_heavy] begin
    include(joinpath(@__DIR__, "..", "..", "experiments", "moment_rii", "run.jl"))

    result = run_triangular_count_error_split_probe(; print_rows=false)

    @test result.parent.stress === :count_error
    @test result.parent.action === :preserve_parent_radius_for_count_error
    @test result.parent.rationale === :unstable_local_count
    @test result.parent.good == result.parent.count_estimate
    @test result.parent.count_error > 1e-2
    @test maximum(result.parent.candidate_radii) == result.parent.radius
    @test result.shrink_child.matched < length(result.shrink_child.expected)
    @test result.policy_child.matched == length(result.policy_child.expected)
    @test result.policy_child.support2_global_matched == length(result.policy_child.expected)
end

@testitem "experimental moment RII: near-pole rational chart is stable" tags=[:slow] begin
    include(joinpath(@__DIR__, "..", "..", "experiments", "moment_rii", "run.jl"))

    result = run_near_pole_rational_boundary_diagnostic(; print_rows=false)

    @test result.gap == 0.005
    @test length(result.rows) == 8
    @test result.success
    @test all(row.matched == row.expected for row in result.rows)
    @test all(row.spurious == 0 for row in result.rows)
    @test all(row.max_residual <= 1e-8 for row in result.rows)
end

@testitem "experimental moment RII: residual Laurent update preserves compact realization" tags=[:slow] begin
    include(joinpath(@__DIR__, "..", "..", "experiments", "moment_rii", "run.jl"))

    result = run_residual_laurent_compression_diagnostic(; print_rows=false)

    @test result.expected == 20
    @test result.compressed_initial.good == 0
    @test result.compressed_updated.matched == result.expected
    @test result.compressed_updated.spurious_good == 0
    @test result.scalar_updated.matched < result.expected
    @test result.compressed_updated.right_residual_rank < result.expected
    @test result.compressed_updated.left_residual_rank < result.expected
    @test result.compressed_updated.right_candidate_cols < result.scalar_updated.right_candidate_cols
    @test result.compressed_updated.left_candidate_cols < result.scalar_updated.left_candidate_cols
    @test result.compressed_updated.right_basis_cols > result.scalar_updated.right_basis_cols
    @test result.efficiency.right_candidate_saved > 0
    @test result.efficiency.left_candidate_saved > 0
    @test result.efficiency.right_candidate_ratio < 1
    @test result.efficiency.left_candidate_ratio < 1
    @test result.efficiency.right_basis_gain > 0
    @test result.efficiency.residual_rank_complete
    @test result.efficiency.scalar_expanded_worse
end

@testitem "experimental moment RII: residual Laurent low-rank compression preserves update" tags=[:slow] begin
    include(joinpath(@__DIR__, "..", "..", "experiments", "moment_rii", "run.jl"))

    result = run_residual_laurent_low_rank_equivalence_diagnostic(; print_rows=false)

    @test result.expected == 20
    @test result.compressed.matched == result.expected
    @test result.full.matched == result.expected
    @test result.compressed.spurious_good == 0
    @test result.full.spurious_good == 0
    @test result.x_projection_gap <= 1e-12
    @test result.y_projection_gap <= 1e-12
    @test result.right_candidate_saved > 0
    @test result.left_candidate_saved > 0
    @test result.compressed_stats.right_residual_rank < result.full_stats.right_residual_rank
    @test result.compressed_stats.left_residual_rank < result.full_stats.left_residual_rank
    @test result.compressed_stats.right_candidate_cols < result.full_stats.right_candidate_cols
    @test result.compressed_stats.left_candidate_cols < result.full_stats.left_candidate_cols
end

@testitem "experimental moment RII: residual Laurent update is residual-coordinate invariant" tags=[:slow] begin
    include(joinpath(@__DIR__, "..", "..", "experiments", "moment_rii", "run.jl"))

    result = run_residual_laurent_residual_coordinate_invariance_diagnostic(; print_rows=false)

    @test result.expected == 20
    @test result.reference.matched == result.expected
    @test result.mixed.matched == result.expected
    @test result.reference.spurious_good == 0
    @test result.mixed.spurious_good == 0
    @test result.x_projection_gap <= 1e-12
    @test result.y_projection_gap <= 1e-12
    @test result.right_residual_rank == result.right_mixed_residual_rank
    @test result.left_residual_rank == result.left_mixed_residual_rank
end

@testitem "experimental moment RII: scalar Laurent truncation alone is not the proof" tags=[:slow] begin
    include(joinpath(@__DIR__, "..", "..", "experiments", "moment_rii", "run.jl"))

    result = run_residual_laurent_scalar_truncation_boundary_diagnostic(; print_rows=false)

    @test result.max_alpha < 0.35
    @test !result.monotone
    @test result.naive_truncation_failed
    @test result.final_error > 1.0
end

@testitem "experimental moment RII: residual Laurent update decomposes across contour partitions" tags=[:slow] begin
    include(joinpath(@__DIR__, "..", "..", "experiments", "moment_rii", "run.jl"))

    result = run_residual_laurent_partition_diagnostic(; partitions=4, print_rows=false)

    @test result.expected == 20
    @test result.partitions == 4
    @test result.serial.matched == result.expected
    @test result.partitioned.matched == result.expected
    @test result.partitioned.spurious_good == 0
    @test result.partitioned.max_residual <= 1e-8
    @test result.x_projection_gap <= 1e-12
    @test result.y_projection_gap <= 1e-12
    @test result.partitioned_stats.right_candidate_cols == result.serial_stats.right_candidate_cols
    @test result.partitioned_stats.left_candidate_cols == result.serial_stats.left_candidate_cols
    @test result.partitioned_stats.right_residual_rank == result.serial_stats.right_residual_rank
    @test result.partitioned_stats.left_residual_rank == result.serial_stats.left_residual_rank
    @test length(result.partitioned_stats.partitions) == result.partitions
    @test sum(row.nodes for row in result.partitioned_stats.partitions) == 128
end

@testitem "experimental moment RII: residual Laurent update runs on remote contour workers" tags=[:slow, :distributed] begin
    if !isdefined(Main, :run_remote_residual_laurent_worker_diagnostic)
        Base.include(Main, joinpath(@__DIR__, "..", "..", "experiments", "moment_rii", "run.jl"))
    end

    result = Main.run_remote_residual_laurent_worker_diagnostic(; worker_count=2, print_rows=false)

    @test length(result.workers) == 2
    @test result.expected == 20
    @test result.serial.matched == result.expected
    @test result.remote.matched == result.expected
    @test result.serial_second.matched == result.expected
    @test result.remote_second.matched == result.expected
    @test result.remote.spurious_good == 0
    @test result.remote_second.spurious_good == 0
    @test result.x_projection_gap <= 1e-12
    @test result.y_projection_gap <= 1e-12
    @test result.x_projection_gap_second <= 1e-12
    @test result.y_projection_gap_second <= 1e-12
    @test length(result.remote_stats.workers) == 2
    @test sum(report.nodes for report in result.remote_stats.workers) == 128
    @test length(result.remote_stats_second.workers) == 2
    @test sum(report.nodes for report in result.remote_stats_second.workers) == 128
    @test result.assignments == [[i for i in 1:2:128], [i for i in 2:2:128]]
    @test result.remote_stats.right_candidate_cols == result.serial_stats.right_candidate_cols
    @test result.remote_stats.left_candidate_cols == result.serial_stats.left_candidate_cols
    @test result.timing.plan_setup_ns > 0
    @test result.timing.serial_update_ns > 0
    @test result.timing.remote_update_ns > 0
    @test 0 < result.timing.remote_worker_max_ns <= result.timing.remote_worker_sum_ns
    @test 0 < result.timing.remote_second_worker_max_ns <= result.timing.remote_second_worker_sum_ns
end

@testitem "experimental moment RII: sparse residual Laurent update runs on remote contour workers" tags=[:slow, :distributed] begin
    if !isdefined(Main, :run_sparse_remote_residual_laurent_worker_smoke)
        Base.include(Main, joinpath(@__DIR__, "..", "..", "experiments", "moment_rii", "run.jl"))
    end

    result = Main.run_sparse_remote_residual_laurent_worker_smoke(; worker_count=2, print_rows=false)

    @test result.sparse_matrix
    @test length(result.workers) == 2
    @test result.expected == 8
    @test result.serial.matched == result.expected
    @test result.remote.matched == result.expected
    @test result.remote.spurious_good == 0
    @test result.x_projection_gap <= 1e-12
    @test result.y_projection_gap <= 1e-12
    @test length(result.remote_stats.workers) == 2
    @test sum(report.nodes for report in result.remote_stats.workers) == 64
    @test result.remote_stats.right_candidate_cols == result.serial_stats.right_candidate_cols
    @test result.remote_stats.left_candidate_cols == result.serial_stats.left_candidate_cols
    @test result.timing.plan_setup_ns > 0
    @test result.timing.serial_update_ns > 0
    @test result.timing.remote_update_ns > 0
    @test 0 < result.timing.remote_worker_max_ns <= result.timing.remote_worker_sum_ns
end

@testitem "experimental moment RII: sparse remote workers reuse stored contour factors" tags=[:slow, :distributed] begin
    if !isdefined(Main, :run_sparse_remote_stored_factor_worker_smoke)
        Base.include(Main, joinpath(@__DIR__, "..", "..", "experiments", "moment_rii", "run.jl"))
    end

    result = Main.run_sparse_remote_stored_factor_worker_smoke(; worker_count=2, print_rows=false)

    @test result.sparse_matrix
    @test length(result.workers) == 2
    @test result.expected == 8
    @test result.serial.matched == result.expected
    @test result.remote.matched == result.expected
    @test result.remote.spurious_good == 0
    @test result.x_projection_gap <= 1e-12
    @test result.y_projection_gap <= 1e-12
    @test result.repeat_x_projection_gap <= 1e-12
    @test result.repeat_y_projection_gap <= 1e-12
    @test sum(length, result.assignments) == 64
    @test result.first_worker_factorizations == 128
    @test result.second_worker_factorizations == result.first_worker_factorizations
    @test result.first_worker_solution_buffers == result.first_worker_factorizations
    @test result.second_worker_solution_buffers == result.first_worker_solution_buffers
    @test result.first_worker_solves == 128
    @test result.second_worker_solves == 2 * result.first_worker_solves
    @test result.remote_stats.right_candidate_cols == result.serial_stats.right_candidate_cols
    @test result.remote_stats.left_candidate_cols == result.serial_stats.left_candidate_cols
end

@testitem "experimental moment RII: residual Laurent update repairs nonnormal chart cover" tags=[:slow, :moment_heavy] begin
    include(joinpath(@__DIR__, "..", "..", "experiments", "moment_rii", "run.jl"))

    result = run_residual_laurent_update_ladder_diagnostic(; print_rows=false)
    rows = result.rows

    @test [row.iterations for row in rows] == [0, 1, 2]
    @test all(row.expected == 44 for row in rows)
    @test rows[1].matched == 34
    @test rows[2].matched == 42
    @test rows[3].matched == 44
    @test rows[1].retained_matched < rows[2].retained_matched < rows[3].retained_matched
    @test rows[3].retained == rows[3].expected
    @test rows[3].retained_matched == rows[3].expected
end

@testitem "experimental moment RII: adaptive radius-20 analytic solve agrees across reduced extractors" tags=[:slow, :moment_heavy] begin
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

@testitem "experimental moment RII: analytic block Newton is only a local refinement rung" tags=[:slow, :moment_heavy] begin
    include(joinpath(@__DIR__, "..", "..", "experiments", "moment_rii", "run.jl"))

    result = run_analytic_block_newton_boundary_diagnostic(; print_rows=false)

    @test first(result.small_block_initial.newton_ratios) < 1
    @test result.small_block_updated.matched == result.small.expected
    @test result.large_none_updated.matched < result.large.expected
    @test result.large_block_updated.matched < result.large_none_updated.matched
    @test first(result.large_block_updated.newton_ratios) == 1
end
