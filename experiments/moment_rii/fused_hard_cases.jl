# Hard fused-realization controls for the moment-NLFEAST research line.
#
# These diagnostics keep larger or more realistic problems separate from the
# main prototype file. They are still experiment-layer code, not public API.

function fused_schrodinger_refinement_row(;
    n,
    center,
    radius,
    nodes,
    moment_count,
    seed,
    residual_tol,
    ranktol,
)
    problem = sparse_schrodinger_movebc_moment_context(n, center, radius)
    _, target_count, count_sums = determinant_power_sums(
        problem.ctx.Tmatrix,
        problem.ctx.Tderivative;
        center=center,
        radius=radius,
        nodes=512,
        capacity=16,
    )

    Random.seed!(seed)
    probe_cols = max(target_count + 2, 6)
    right_probe = rand(ComplexF64, n, probe_cols)
    left_probe = rand(ComplexF64, n, probe_cols)
    cache = build_contour_sample_cache(
        problem.ctx.Tsolve,
        problem.ctx.Tadjoint_solve,
        right_probe,
        left_probe,
        problem.chart,
        nodes;
        source=:fused_schrodinger_refinement,
    )
    right_blocks = right_moments(cache, moment_count)
    left_blocks = left_moments(cache, moment_count)

    Xfused, Sfused, fused_rank, fused_singulars = projected_hankel_pair_identity(
        right_blocks,
        left_probe,
        moment_count;
        ranktol=ranktol,
        maxrank=target_count,
    )
    Ffused = eigen(Sfused)
    raw_values = center .+ radius .* ComplexF64.(Ffused.values)
    raw_vectors = Xfused * Ffused.vectors
    normalize_columns_local!(raw_vectors)
    raw_residuals = matrix_vector_residuals(problem.ctx.Tmatrix, raw_values, raw_vectors; normalization=:vector)
    raw_inside = FEASTSolver.in_contour(raw_values, center, radius)
    raw_good = raw_inside .& (raw_residuals .<= residual_tol)

    Xbasis, _ = moment_block_basis(right_blocks, moment_count; ranktol=ranktol)
    Ybasis, _ = moment_block_basis(left_blocks, moment_count; ranktol=ranktol)
    d = min(size(Xbasis, 2), size(Ybasis, 2))
    Xbasis = Xbasis[:, 1:d]
    Ybasis = Ybasis[:, 1:d]
    Tred = z -> adjoint(Ybasis) * problem.ctx.Tmatrix(z) * Xbasis
    Tred_derivative = z -> adjoint(Ybasis) * problem.ctx.Tderivative(z) * Xbasis
    refined_values, corrections = refine_reduced_analytic_triplets(
        Tred,
        Tred_derivative,
        raw_values;
        steps=4,
        step_limit=0.25 * radius,
    )
    _, refined_right = reduced_left_right_singular_vectors(Tred, refined_values)
    refined_vectors = Xbasis * refined_right
    normalize_columns_local!(refined_vectors)
    refined_residuals = matrix_vector_residuals(problem.ctx.Tmatrix, refined_values, refined_vectors; normalization=:vector)
    refined_inside = FEASTSolver.in_contour(refined_values, center, radius)
    refined_good = refined_inside .& (refined_residuals .<= residual_tol)

    (
        nodes=nodes,
        target_count=target_count,
        target_count_error=abs(count_sums[1] - target_count),
        rank=fused_rank,
        raw_inside=count(raw_inside),
        raw_good=count(raw_good),
        raw_max_inside_residual=any(raw_inside) ? maximum(raw_residuals[raw_inside]) : Inf,
        refined_inside=count(refined_inside),
        refined_good=count(refined_good),
        refined_max_inside_residual=any(refined_inside) ? maximum(refined_residuals[refined_inside]) : Inf,
        max_refinement_correction=isempty(corrections) ? 0.0 : maximum(corrections),
        basis_dim=d,
        fused_sigma=isempty(fused_singulars) ? NaN : fused_singulars[min(end, fused_rank)] / fused_singulars[1],
    )
end

function run_fused_schrodinger_refinement_sweep(;
    n=128,
    center=-35.0 + 0.0im,
    radius=4.2,
    nodes_values=(16, 24, 32, 48),
    moment_count=4,
    seed=9901,
    residual_tol=1e-5,
    ranktol=1e-10,
    print_rows=true,
)
    rows = [
        fused_schrodinger_refinement_row(;
            n=n,
            center=center,
            radius=radius,
            nodes=nodes,
            moment_count=moment_count,
            seed=seed,
            residual_tol=residual_tol,
            ranktol=ranktol,
        )
        for nodes in nodes_values
    ]

    if print_rows
        println()
        println("Fused Schrodinger refinement sweep")
        println("  moving-boundary sparse Schrodinger gallery; measures when Tred cleanup is needed")
        for row in rows
            @printf(
                "  nodes=%d count=%d rank=%d raw_good=%d max=%.3e refined_good=%d max=%.3e max_corr=%.3e basis=%d\n",
                row.nodes,
                row.target_count,
                row.rank,
                row.raw_good,
                row.raw_max_inside_residual,
                row.refined_good,
                row.refined_max_inside_residual,
                row.max_refinement_correction,
                row.basis_dim,
            )
        end
    end

    (
        n=n,
        center=ComplexF64(center),
        radius=Float64(radius),
        moment_count=moment_count,
        residual_tol=residual_tol,
        rows=rows,
    )
end

function schrodinger_dd_interface_context(;
    subdomains=16,
    interior_per_subdomain=12,
    potential_amplitude=8.0,
    potential_frequency=19.0,
    center=900.0 + 0.0im,
    radius=1000.0,
)
    subdomains >= 2 || error("domain-decomposition Schrodinger control needs at least two subdomains")
    interior_per_subdomain >= 2 || error("each subdomain needs at least two interior points")
    block = interior_per_subdomain + 1
    nfull = subdomains * block - 1
    h = inv(nfull + 1)
    x = collect((1:nfull) .* h)
    potential = potential_amplitude .* (1 .+ sin.(potential_frequency .* x) .+ 0.25 .* cos.(0.37 * potential_frequency .* x))
    A = SymTridiagonal(2.0 ./ h^2 .+ potential, -ones(nfull - 1) ./ h^2) |> Matrix{Float64}

    interface = collect(block:block:(nfull - 1))
    interiors = setdiff(1:nfull, interface)
    AII = Matrix{ComplexF64}(A[interiors, interiors])
    AIB = Matrix{ComplexF64}(A[interiors, interface])
    ABI = Matrix{ComplexF64}(A[interface, interiors])
    ABB = Matrix{ComplexF64}(A[interface, interface])
    Iint = Matrix{ComplexF64}(I, length(interiors), length(interiors))
    Ibd = Matrix{ComplexF64}(I, length(interface), length(interface))

    function Tmatrix(z)
        K = AII .- z .* Iint
        ABB .- z .* Ibd .- ABI * (K \ AIB)
    end
    function Tderivative(z)
        K = AII .- z .* Iint
        KinvAIB = K \ AIB
        -Ibd .- ABI * (K \ KinvAIB)
    end
    Tsolve = (z, B) -> Tmatrix(z) \ B
    Tadjoint_solve = (z, B) -> adjoint(Tmatrix(z)) \ B

    full_values = eigvals(Symmetric(A))
    expected = ComplexF64[λ for λ in full_values if abs(λ - real(center)) <= radius]
    local_poles = eigvals(AII)
    pole_dist = isempty(local_poles) ? Inf : minimum(abs.(ComplexF64.(local_poles) .- center))
    pole_boundary_margin = pole_dist - radius

    (
        full_matrix=A,
        interface=interface,
        interiors=interiors,
        chart=ContourChart(center, radius),
        expected=expected,
        pole_distance=pole_dist,
        pole_boundary_margin=pole_boundary_margin,
        ctx=(
            Tmatrix=Tmatrix,
            Tderivative=Tderivative,
            Tsolve=Tsolve,
            Tadjoint_solve=Tadjoint_solve,
            expected=expected,
            n=length(interface),
            component_scales=ones(Float64, length(interface)),
        ),
    )
end

function run_fused_schrodinger_dd_interface_diagnostic(;
    subdomains=16,
    interior_per_subdomain=12,
    center=900.0 + 0.0im,
    radius=1000.0,
    nodes=128,
    moment_count=3,
    seed=20260506,
    ranktol=1e-10,
    residual_tol=1e-8,
    match_atol=1e-7,
    print_rows=true,
)
    problem = schrodinger_dd_interface_context(;
        subdomains=subdomains,
        interior_per_subdomain=interior_per_subdomain,
        center=center,
        radius=radius,
    )
    target_count = length(problem.expected)
    Random.seed!(seed)
    probe_cols = max(target_count + 2, problem.ctx.n)
    right_probe = rand(ComplexF64, problem.ctx.n, probe_cols)
    left_probe = rand(ComplexF64, problem.ctx.n, probe_cols)
    cache = build_contour_sample_cache(
        problem.ctx.Tsolve,
        problem.ctx.Tadjoint_solve,
        right_probe,
        left_probe,
        problem.chart,
        nodes;
        source=:fused_schrodinger_dd_interface,
    )
    right_blocks = right_moments(cache, moment_count)
    left_blocks = left_moments(cache, moment_count)

    Xfused, Sfused, fused_rank, fused_singulars = projected_hankel_pair_identity(
        right_blocks,
        left_probe,
        moment_count;
        ranktol=ranktol,
        maxrank=target_count,
    )
    Ffused = eigen(Sfused)
    raw_values = center .+ radius .* ComplexF64.(Ffused.values)
    raw_vectors = Xfused * Ffused.vectors
    normalize_columns_local!(raw_vectors)
    raw_residuals = matrix_vector_residuals(problem.ctx.Tmatrix, raw_values, raw_vectors; normalization=:vector)
    raw_inside = FEASTSolver.in_contour(raw_values, center, radius)
    raw_good = raw_inside .& (raw_residuals .<= residual_tol)
    raw_matched = match_expected_count(raw_values[raw_good], problem.expected; atol=match_atol)

    Xbasis, _ = moment_block_basis(right_blocks, moment_count; ranktol=ranktol)
    Ybasis, _ = moment_block_basis(left_blocks, moment_count; ranktol=ranktol)
    d = min(size(Xbasis, 2), size(Ybasis, 2))
    Xbasis = Xbasis[:, 1:d]
    Ybasis = Ybasis[:, 1:d]
    Tred = z -> adjoint(Ybasis) * problem.ctx.Tmatrix(z) * Xbasis
    Tred_derivative = z -> adjoint(Ybasis) * problem.ctx.Tderivative(z) * Xbasis
    refined_values, corrections = refine_reduced_analytic_triplets(
        Tred,
        Tred_derivative,
        raw_values;
        steps=6,
        step_limit=0.25 * radius,
    )
    _, refined_right = reduced_left_right_singular_vectors(Tred, refined_values)
    refined_vectors = Xbasis * refined_right
    normalize_columns_local!(refined_vectors)
    refined_residuals = matrix_vector_residuals(problem.ctx.Tmatrix, refined_values, refined_vectors; normalization=:vector)
    refined_inside = FEASTSolver.in_contour(refined_values, center, radius)
    refined_good = refined_inside .& (refined_residuals .<= residual_tol)
    refined_matched = match_expected_count(refined_values[refined_good], problem.expected; atol=match_atol)

    if print_rows
        println()
        println("Fused Schrodinger domain-decomposition interface diagnostic")
        println("  Schur-complement NEP from eliminating local interiors of a linear Schrodinger operator")
        println("  contour is kept below the first eliminated-interior pole so the sampled NEP is analytic")
        @printf(
            "  full_n=%d interface_n=%d expected=%d pole_margin=%.3e nodes=%d rank=%d basis=%d\n",
            size(problem.full_matrix, 1),
            problem.ctx.n,
            target_count,
            problem.pole_boundary_margin,
            nodes,
            fused_rank,
            d,
        )
        @printf(
            "  raw good=%d matched=%d max=%.3e refined good=%d matched=%d max=%.3e max_corr=%.3e\n",
            count(raw_good),
            raw_matched,
            any(raw_inside) ? maximum(raw_residuals[raw_inside]) : Inf,
            count(refined_good),
            refined_matched,
            any(refined_inside) ? maximum(refined_residuals[refined_inside]) : Inf,
            isempty(corrections) ? 0.0 : maximum(corrections),
        )
    end

    (
        full_n=size(problem.full_matrix, 1),
        interface_n=problem.ctx.n,
        expected=target_count,
        pole_distance=problem.pole_distance,
        pole_boundary_margin=problem.pole_boundary_margin,
        rank=fused_rank,
        basis_dim=d,
        raw_inside=count(raw_inside),
        raw_good=count(raw_good),
        raw_matched=raw_matched,
        raw_max_inside_residual=any(raw_inside) ? maximum(raw_residuals[raw_inside]) : Inf,
        refined_inside=count(refined_inside),
        refined_good=count(refined_good),
        refined_matched=refined_matched,
        refined_max_inside_residual=any(refined_inside) ? maximum(refined_residuals[refined_inside]) : Inf,
        max_refinement_correction=isempty(corrections) ? 0.0 : maximum(corrections),
        fused_singulars=Float64.(fused_singulars),
    )
end
