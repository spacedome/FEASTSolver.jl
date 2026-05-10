# Legacy all-in-one driver for the early moment-RII exploration.
#
# The current experiment is better consumed through `showcases.jl`,
# `torture_matrix.jl`, and the focused tests. This driver remains useful when
# replaying the original exploratory sequence from `julia experiments/moment_rii/run.jl`.

function main()
    run_case("diagonal_linear", diagonal_linear_problem; nodes=16, iterations=3, moment_counts=(1, 2), ranktol=1e-12, residual_tol=1e-10)
    run_case("deficient_quadratic", deficient_quadratic_problem; nodes=64, iterations=5, moment_counts=(1, 2, 3), ranktol=1e-9, residual_tol=1e-8)
    run_case("butterfly", butterfly_problem; nodes=32, iterations=3, moment_counts=(1, 2), ranktol=1e-9, residual_tol=1e-8, keep_extra=3)
    run_node_sweep("deficient_quadratic", deficient_quadratic_problem; node_values=(8, 12, 16, 24, 32), iterations=5, moment_count=2, ranktol=1e-9, residual_tol=1e-8)
    run_node_sweep("butterfly", butterfly_problem; node_values=(8, 12, 16, 24, 32), iterations=5, moment_count=2, ranktol=1e-9, residual_tol=1e-8, keep_extra=3)
    run_projected_hankel_case("deficient_quadratic", deficient_quadratic_problem; nodes=16, iterations=5, moment_count=2, ranktol=1e-9, residual_tol=1e-8)
    run_projected_hankel_case("butterfly", butterfly_problem; nodes=32, iterations=3, moment_count=2, ranktol=1e-9, residual_tol=1e-8, keep_extra=3)
    run_linear_ss_feast_control("many_eigenvalue_diagonal", many_eigenvalue_diagonal_matrix; nodes=32, iterations=4, moment_count=3, ranktol=1e-11, residual_tol=1e-10, target_count=true)
    run_linear_ss_feast_control("grcar_nonnormal", grcar_linear_matrix; nodes=48, iterations=5, moment_count=2, ranktol=1e-10, residual_tol=1e-8)
    run_polynomial_companion_control(
        "many_eigenvalue_nonnormal_polynomial",
        many_eigenvalue_nonnormal_polynomial_problem;
        nodes=16,
        iterations=6,
        moment_count=5,
        ranktol=1e-10,
        residual_tol=1e-8,
    )
    run_polynomial_projective_chart_control()
    run_dual_reduced_polynomial_control()
    run_dual_reduced_polynomial_control(;
        name="dual_sensitive_polynomial",
        make_problem=dual_sensitive_polynomial_problem,
        basis_nodes=48,
        basis_ranktol=1e-10,
        residual_tol=1e-6,
        match_atol=1e-3,
    )
    run_dual_scalar_rii_polynomial_experiment(;
        name="dual_sensitive_polynomial_bad_initial",
        make_problem=dual_sensitive_polynomial_problem,
        basis_nodes=6,
        rii_nodes=48,
        iterations=1,
        basis_ranktol=1e-6,
        residual_ranktol=1e-10,
        residual_tol=1e-6,
        match_atol=1e-3,
    )
    run_dual_moment_compressed_rii_polynomial_experiment(;
        name="dual_sensitive_polynomial_bad_initial",
        make_problem=dual_sensitive_polynomial_problem,
        basis_nodes=6,
        rii_nodes=48,
        update_moments=(1, 2),
        basis_ranktol=1e-6,
        residual_ranktol=1e-10,
        residual_tol=1e-6,
        match_atol=1e-3,
    )
    run_dual_moment_compressed_rii_polynomial_experiment(;
        name="many_eigenvalue_nonnormal_rank_deficient_initial",
        make_problem=many_eigenvalue_nonnormal_polynomial_problem,
        basis_nodes=6,
        rii_nodes=48,
        update_moments=(1,),
        basis_ranktol=1e-1,
        residual_ranktol=1e-10,
        residual_tol=1e-7,
        match_atol=1e-6,
    )
    run_dual_moment_compressed_rii_analytic_iteration(;
        cases=(scalar_sine_case(), scalar_cosine_case(), scalar_shifted_sine_case()),
        radius=20.0,
        basis_moments=4,
        basis_nodes=8,
        rii_nodes=256,
        update_moment_count=1,
        iterations=1,
        basis_ranktol=0.5,
        determinant_nodes=1024,
        determinant_capacity=80,
        extractor=:ss_counted,
        reduced_moments=16,
        reduced_nodes=1024,
        residual_tol=1e-8,
        match_atol=1e-6,
    )
    run_reduced_loewner_extractor_comparison()
    run_dual_grid_chart_cover_analytic(;
        spacing=2.4,
        chart_radius=1.8,
        basis_nodes=32,
        rii_nodes=512,
        iterations=2,
        basis_ranktol=1e-8,
        determinant_nodes=1024,
        reduced_nodes=1024,
        residual_normalization=:vector,
        component_scaling=:contour_max,
        residual_tol=1e-8,
        match_atol=1e-6,
    )
    run_dual_grid_chart_cover_triangular_analytic(;
        coupling=10.0,
        spacing=2.4,
        chart_radii=(0.8, 1.2, 1.8, 2.4, 3.0),
        basis_nodes=32,
        rii_nodes=512,
        iterations=2,
        basis_ranktol=1e-8,
        determinant_nodes=1024,
        reduced_nodes=1024,
        residual_normalization=:vector,
        component_scaling=:contour_max,
        residual_tol=1e-8,
        match_atol=1e-6,
    )
    run_dual_multiple_root_analytic_stress(;
        radius=10.0,
        basis_moments=8,
        basis_nodes=64,
        iterations=1,
        determinant_capacity=64,
        residual_tol=1e-8,
        match_atol=1e-6,
    )
    run_dual_multiple_root_analytic_stress(;
        cases=(scalar_sine_case(), scalar_expm1_case()),
        operator_builder=triangular_operator_builder(; coupling=3.0),
        operator_label="triangular(coupling=3.0)",
        radius=1.0,
        basis_moments=4,
        basis_nodes=32,
        iterations=1,
        determinant_capacity=16,
        reduced_moments=8,
        reduced_nodes=512,
        residual_tol=1e-8,
        match_atol=1e-6,
    )
    run_nonlinear_update_comparison("deficient_quadratic", deficient_quadratic_problem; nodes=16, iterations=5, moment_count=2, ranktol=1e-9, residual_tol=1e-8)
    run_nonlinear_update_comparison("butterfly", butterfly_problem; nodes=32, iterations=3, moment_count=2, ranktol=1e-9, residual_tol=1e-8, keep_extra=3)
    run_nonlinear_update_comparison("butterfly_target", butterfly_problem; nodes=32, iterations=3, moment_count=2, ranktol=1e-9, residual_tol=1e-8, target_count=true, modes=(:projected, :shifted, :shifted_gauge_balanced, :projected_newton))
    run_nonlinear_update_comparison(
        "many_eigenvalue_nonnormal_polynomial_K4_capacity_failure",
        many_eigenvalue_nonnormal_polynomial_problem;
        nodes=32,
        iterations=6,
        moment_count=4,
        ranktol=1e-10,
        residual_tol=1e-8,
        target_count=true,
        modes=(:projected, :shifted, :shifted_gauge_balanced),
    )
    run_nonlinear_update_comparison(
        "many_eigenvalue_nonnormal_polynomial_low_nodes",
        many_eigenvalue_nonnormal_polynomial_problem;
        nodes=8,
        iterations=8,
        moment_count=5,
        ranktol=1e-10,
        residual_tol=1e-8,
        target_count=true,
        modes=(:projected, :shifted, :shifted_gauge_balanced),
    )
    run_nonlinear_update_comparison(
        "many_eigenvalue_nonnormal_polynomial",
        many_eigenvalue_nonnormal_polynomial_problem;
        nodes=16,
        iterations=6,
        moment_count=5,
        ranktol=1e-10,
        residual_tol=1e-8,
        target_count=true,
        modes=(:projected, :shifted, :shifted_gauge_balanced),
    )
    run_scalar_sine_sweep()
    run_scalar_sine_scaled_sweep()
    run_scalar_sine_realization_sweep()
    run_scalar_sine_realization_sweep(radius=20.0, node_values=(128,), iterations=8, moment_counts=(13,), ranktol=1e-10, residual_tol=1e-8)
    run_scalar_nested_contour_demo()
    run_scalar_adaptive_chart_demo()
    run_scalar_rank_adaptive_chart_demo()
    run_scalar_rank_estimation_stress()
    run_scalar_rank_adaptive_stress()
    run_diagonal_analytic_moment_stress()
    run_diagonal_analytic_rank_adaptive_stress()
    run_dual_reduced_determinant_diagonal_stress()
end
