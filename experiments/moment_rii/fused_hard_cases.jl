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

struct SchrodingerDDLocalBlock
    interface_positions::Vector{Int}
    AII::Matrix{ComplexF64}
    AIB::Matrix{ComplexF64}
    ABI::Matrix{ComplexF64}
    I::Matrix{ComplexF64}
end

struct SchrodingerDDInterfaceOperator
    full_operator::SymTridiagonal{Float64, Vector{Float64}}
    interface::Vector{Int}
    ABB::Matrix{ComplexF64}
    Ibd::Matrix{ComplexF64}
    blocks::Vector{SchrodingerDDLocalBlock}
    local_poles::Vector{Float64}
end

Base.@kwdef struct FusedSchrodingerDDConfig
    subdomains::Int = 16
    interior_per_subdomain::Int = 12
    potential_amplitude::Float64 = 8.0
    potential_frequency::Float64 = 19.0
    center::ComplexF64 = 900.0 + 0.0im
    radius::Float64 = 1000.0
    nodes::Int = 128
    moment_count::Int = 3
    seed::Int = 20260506
    ranktol::Float64 = 1e-10
    residual_tol::Float64 = 1e-8
    match_atol::Float64 = 1e-7
    print_rows::Bool = true
end

function schrodinger_dd_full_operator(subdomains, interior_per_subdomain, potential_amplitude, potential_frequency)
    block = interior_per_subdomain + 1
    nfull = subdomains * block - 1
    h = inv(nfull + 1)
    x = collect((1:nfull) .* h)
    potential = potential_amplitude .* (1 .+ sin.(potential_frequency .* x) .+ 0.25 .* cos.(0.37 * potential_frequency .* x))
    SymTridiagonal(2.0 ./ h^2 .+ potential, -ones(nfull - 1) ./ h^2)
end

function schrodinger_dd_interface_operator(;
    subdomains=16,
    interior_per_subdomain=12,
    potential_amplitude=8.0,
    potential_frequency=19.0,
)
    subdomains >= 2 || error("domain-decomposition Schrodinger control needs at least two subdomains")
    interior_per_subdomain >= 2 || error("each subdomain needs at least two interior points")
    block = interior_per_subdomain + 1
    A = schrodinger_dd_full_operator(subdomains, interior_per_subdomain, potential_amplitude, potential_frequency)
    nfull = size(A, 1)
    interface = collect(block:block:(nfull - 1))
    interface_lookup = Dict(index => pos for (pos, index) in pairs(interface))
    ABB = Matrix{ComplexF64}(A[interface, interface])
    Ibd = Matrix{ComplexF64}(I, length(interface), length(interface))

    blocks = SchrodingerDDLocalBlock[]
    local_poles = Float64[]
    for subdomain in 1:subdomains
        first_index = (subdomain - 1) * block + 1
        last_index = subdomain == subdomains ? nfull : subdomain * block - 1
        interiors = collect(first_index:last_index)
        local_interface_indices = Int[]
        if subdomain > 1
            push!(local_interface_indices, first_index - 1)
        end
        if subdomain < subdomains
            push!(local_interface_indices, last_index + 1)
        end
        interface_positions = [interface_lookup[index] for index in local_interface_indices]
        AII = Matrix{ComplexF64}(A[interiors, interiors])
        AIB = Matrix{ComplexF64}(A[interiors, local_interface_indices])
        ABI = Matrix{ComplexF64}(A[local_interface_indices, interiors])
        Iint = Matrix{ComplexF64}(I, length(interiors), length(interiors))
        append!(local_poles, eigvals(Symmetric(real.(AII))))
        push!(blocks, SchrodingerDDLocalBlock(interface_positions, AII, AIB, ABI, Iint))
    end

    SchrodingerDDInterfaceOperator(A, interface, ABB, Ibd, blocks, sort!(local_poles))
end

function materialize!(M, op::SchrodingerDDInterfaceOperator, z; derivative=false)
    if derivative
        copyto!(M, -op.Ibd)
    else
        copyto!(M, op.ABB)
        M .-= z .* op.Ibd
    end
    for block in op.blocks
        K = block.AII .- z .* block.I
        if derivative
            KinvAIB = K \ block.AIB
            contribution = -block.ABI * (K \ KinvAIB)
        else
            contribution = -block.ABI * (K \ block.AIB)
        end
        M[block.interface_positions, block.interface_positions] .+= contribution
    end
    M
end

function materialize(op::SchrodingerDDInterfaceOperator, z; derivative=false)
    M = similar(op.ABB)
    materialize!(M, op, z; derivative=derivative)
end

function schrodinger_dd_dense_reference_materialize(op::SchrodingerDDInterfaceOperator, z; derivative=false)
    A = Matrix(op.full_operator)
    interface = op.interface
    interiors = setdiff(1:size(A, 1), interface)
    AII = Matrix{ComplexF64}(A[interiors, interiors])
    AIB = Matrix{ComplexF64}(A[interiors, interface])
    ABI = Matrix{ComplexF64}(A[interface, interiors])
    ABB = Matrix{ComplexF64}(A[interface, interface])
    Iint = Matrix{ComplexF64}(I, length(interiors), length(interiors))
    Ibd = Matrix{ComplexF64}(I, length(interface), length(interface))
    K = AII .- z .* Iint
    if derivative
        KinvAIB = K \ AIB
        -Ibd .- ABI * (K \ KinvAIB)
    else
        ABB .- z .* Ibd .- ABI * (K \ AIB)
    end
end

function schrodinger_dd_local_assembly_error(op::SchrodingerDDInterfaceOperator, z)
    local_T = materialize(op, z)
    dense_T = schrodinger_dd_dense_reference_materialize(op, z)
    local_dT = materialize(op, z; derivative=true)
    dense_dT = schrodinger_dd_dense_reference_materialize(op, z; derivative=true)
    (
        T=norm(local_T - dense_T) / max(norm(dense_T), eps(Float64)),
        derivative=norm(local_dT - dense_dT) / max(norm(dense_dT), eps(Float64)),
    )
end

function fused_schrodinger_dd_diagnosis(;
    expected,
    rank,
    raw_good,
    refined_good,
    refined_matched,
    refined_max_inside_residual,
    residual_tol,
    pole_boundary_margin,
)
    if pole_boundary_margin <= 0
        :unsafe_contour_pole
    elseif rank != expected
        :rank_mismatch
    elseif refined_matched != expected || refined_good != expected || refined_max_inside_residual > residual_tol
        :underresolved_or_ill_conditioned
    elseif raw_good != expected
        :cleanup_required
    else
        :raw_sufficient
    end
end

function schrodinger_dd_interface_context(;
    subdomains=16,
    interior_per_subdomain=12,
    potential_amplitude=8.0,
    potential_frequency=19.0,
    center=900.0 + 0.0im,
    radius=1000.0,
)
    op = schrodinger_dd_interface_operator(;
        subdomains=subdomains,
        interior_per_subdomain=interior_per_subdomain,
        potential_amplitude=potential_amplitude,
        potential_frequency=potential_frequency,
    )

    Tmatrix = z -> materialize(op, z)
    Tderivative = z -> materialize(op, z; derivative=true)
    Tsolve = (z, B) -> Tmatrix(z) \ B
    Tadjoint_solve = (z, B) -> adjoint(Tmatrix(z)) \ B

    full_values = eigvals(op.full_operator)
    expected = ComplexF64[λ for λ in full_values if abs(λ - real(center)) <= radius]
    pole_dist = isempty(op.local_poles) ? Inf : minimum(abs.(ComplexF64.(op.local_poles) .- center))
    pole_boundary_margin = pole_dist - radius

    (
        operator=op,
        full_matrix=op.full_operator,
        interface=op.interface,
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
            n=length(op.interface),
            component_scales=ones(Float64, length(op.interface)),
        ),
    )
end

function run_fused_schrodinger_dd_interface_diagnostic(config::FusedSchrodingerDDConfig)
    problem = schrodinger_dd_interface_context(;
        subdomains=config.subdomains,
        interior_per_subdomain=config.interior_per_subdomain,
        potential_amplitude=config.potential_amplitude,
        potential_frequency=config.potential_frequency,
        center=config.center,
        radius=config.radius,
    )
    target_count = length(problem.expected)
    Random.seed!(config.seed)
    probe_cols = max(target_count + 2, problem.ctx.n)
    right_probe = rand(ComplexF64, problem.ctx.n, probe_cols)
    left_probe = rand(ComplexF64, problem.ctx.n, probe_cols)
    cache = build_contour_sample_cache(
        problem.ctx.Tsolve,
        problem.ctx.Tadjoint_solve,
        right_probe,
        left_probe,
        problem.chart,
        config.nodes;
        source=:fused_schrodinger_dd_interface,
    )
    right_blocks = right_moments(cache, config.moment_count)
    left_blocks = left_moments(cache, config.moment_count)

    Xfused, Sfused, fused_rank, fused_singulars = projected_hankel_pair_identity(
        right_blocks,
        left_probe,
        config.moment_count;
        ranktol=config.ranktol,
        maxrank=target_count,
    )
    Ffused = eigen(Sfused)
    raw_values = config.center .+ config.radius .* ComplexF64.(Ffused.values)
    raw_vectors = Xfused * Ffused.vectors
    normalize_columns_local!(raw_vectors)
    raw_residuals = matrix_vector_residuals(problem.ctx.Tmatrix, raw_values, raw_vectors; normalization=:vector)
    raw_inside = FEASTSolver.in_contour(raw_values, config.center, config.radius)
    raw_good = raw_inside .& (raw_residuals .<= config.residual_tol)
    raw_matched = match_expected_count(raw_values[raw_good], problem.expected; atol=config.match_atol)

    Xbasis, _ = moment_block_basis(right_blocks, config.moment_count; ranktol=config.ranktol)
    Ybasis, _ = moment_block_basis(left_blocks, config.moment_count; ranktol=config.ranktol)
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
        step_limit=0.25 * config.radius,
    )
    _, refined_right = reduced_left_right_singular_vectors(Tred, refined_values)
    refined_vectors = Xbasis * refined_right
    normalize_columns_local!(refined_vectors)
    refined_residuals = matrix_vector_residuals(problem.ctx.Tmatrix, refined_values, refined_vectors; normalization=:vector)
    refined_inside = FEASTSolver.in_contour(refined_values, config.center, config.radius)
    refined_good = refined_inside .& (refined_residuals .<= config.residual_tol)
    refined_matched = match_expected_count(refined_values[refined_good], problem.expected; atol=config.match_atol)
    refined_max_inside = any(refined_inside) ? maximum(refined_residuals[refined_inside]) : Inf
    diagnosis = fused_schrodinger_dd_diagnosis(;
        expected=target_count,
        rank=fused_rank,
        raw_good=count(raw_good),
        refined_good=count(refined_good),
        refined_matched=refined_matched,
        refined_max_inside_residual=refined_max_inside,
        residual_tol=config.residual_tol,
        pole_boundary_margin=problem.pole_boundary_margin,
    )

    if config.print_rows
        println()
        println("Fused Schrodinger domain-decomposition interface diagnostic")
        println("  Schur-complement NEP from eliminating local interiors of a linear Schrodinger operator")
        println("  contour is kept below the first eliminated-interior pole so the sampled NEP is analytic")
        @printf(
            "  full_n=%d interface_n=%d expected=%d pole_margin=%.3e nodes=%d rank=%d basis=%d diagnosis=%s\n",
            size(problem.full_matrix, 1),
            problem.ctx.n,
            target_count,
            problem.pole_boundary_margin,
            config.nodes,
            fused_rank,
            d,
            string(diagnosis),
        )
        @printf(
            "  raw good=%d matched=%d max=%.3e refined good=%d matched=%d max=%.3e max_corr=%.3e\n",
            count(raw_good),
            raw_matched,
            any(raw_inside) ? maximum(raw_residuals[raw_inside]) : Inf,
            count(refined_good),
            refined_matched,
            refined_max_inside,
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
        refined_max_inside_residual=refined_max_inside,
        max_refinement_correction=isempty(corrections) ? 0.0 : maximum(corrections),
        diagnosis=diagnosis,
        compression_ratio=size(problem.full_matrix, 1) / problem.ctx.n,
        local_blocks=length(problem.operator.blocks),
        max_local_block_size=maximum(block -> size(block.AII, 1), problem.operator.blocks),
        fused_singulars=Float64.(fused_singulars),
    )
end

function run_fused_schrodinger_dd_interface_diagnostic(; kwargs...)
    run_fused_schrodinger_dd_interface_diagnostic(FusedSchrodingerDDConfig(; kwargs...))
end

function run_fused_schrodinger_dd_interface_refinement_sweep(;
    nodes_values=(64, 96, 128),
    moment_count=3,
    print_rows=true,
    kwargs...,
)
    rows = [
        run_fused_schrodinger_dd_interface_diagnostic(;
            nodes=nodes,
            moment_count=moment_count,
            print_rows=false,
            kwargs...,
        )
        for nodes in nodes_values
    ]

    if print_rows
        println()
        println("Fused Schrodinger domain-decomposition refinement sweep")
        println("  tests when reduced Tred cleanup is enough versus when contour quadrature is still underresolved")
        for (nodes, row) in zip(nodes_values, rows)
            @printf(
                "  nodes=%d expected=%d rank=%d raw_good=%d raw_max=%.3e refined_good=%d refined_matched=%d refined_max=%.3e max_corr=%.3e\n",
                nodes,
                row.expected,
                row.rank,
                row.raw_good,
                row.raw_max_inside_residual,
                row.refined_good,
                row.refined_matched,
                row.refined_max_inside_residual,
                row.max_refinement_correction,
            )
        end
    end

    (
        nodes_values=nodes_values,
        moment_count=moment_count,
        rows=rows,
    )
end

function run_fused_schrodinger_dd_scale_smoke(;
    config=FusedSchrodingerDDConfig(;
        subdomains=64,
        interior_per_subdomain=32,
        nodes=128,
        residual_tol=1e-7,
        match_atol=1e-6,
    ),
    print_rows=true,
)
    local_result = run_fused_schrodinger_dd_interface_diagnostic(;
        subdomains=config.subdomains,
        interior_per_subdomain=config.interior_per_subdomain,
        potential_amplitude=config.potential_amplitude,
        potential_frequency=config.potential_frequency,
        center=config.center,
        radius=config.radius,
        nodes=config.nodes,
        moment_count=config.moment_count,
        seed=config.seed,
        ranktol=config.ranktol,
        residual_tol=config.residual_tol,
        match_atol=config.match_atol,
        print_rows=false,
    )
    small_op = schrodinger_dd_interface_operator(;
        subdomains=8,
        interior_per_subdomain=6,
        potential_amplitude=config.potential_amplitude,
        potential_frequency=config.potential_frequency,
    )
    assembly_error = schrodinger_dd_local_assembly_error(small_op, config.center + 0.31 * config.radius * im)

    if print_rows
        println()
        println("Fused Schrodinger domain-decomposition scale smoke")
        @printf(
            "  full_n=%d interface_n=%d compression=%.1f local_blocks=%d block_n=%d diagnosis=%s\n",
            local_result.full_n,
            local_result.interface_n,
            local_result.compression_ratio,
            local_result.local_blocks,
            local_result.max_local_block_size,
            string(local_result.diagnosis),
        )
        @printf(
            "  expected=%d rank=%d refined_matched=%d refined_max=%.3e local_assembly_error=(T %.3e, dT %.3e)\n",
            local_result.expected,
            local_result.rank,
            local_result.refined_matched,
            local_result.refined_max_inside_residual,
            assembly_error.T,
            assembly_error.derivative,
        )
    end

    (;
        local_result...,
        assembly_error=assembly_error,
    )
end

function run_fused_schrodinger_dd_baseline_comparison(;
    config=FusedSchrodingerDDConfig(),
    feast_subspace_multiplier=2,
    feast_extra_columns=8,
    feast_iter=4,
    feast_nodes=64,
    feast_tol=1e-7,
    print_rows=true,
)
    op = schrodinger_dd_interface_operator(;
        subdomains=config.subdomains,
        interior_per_subdomain=config.interior_per_subdomain,
        potential_amplitude=config.potential_amplitude,
        potential_frequency=config.potential_frequency,
    )
    direct_start = time_ns()
    full_values = eigvals(op.full_operator)
    direct_s = (time_ns() - direct_start) / 1e9
    expected = ComplexF64[λ for λ in full_values if abs(λ - real(config.center)) <= config.radius]
    target_count = length(expected)

    fused_start = time_ns()
    fused = run_fused_schrodinger_dd_interface_diagnostic(FusedSchrodingerDDConfig(;
        subdomains=config.subdomains,
        interior_per_subdomain=config.interior_per_subdomain,
        potential_amplitude=config.potential_amplitude,
        potential_frequency=config.potential_frequency,
        center=config.center,
        radius=config.radius,
        nodes=config.nodes,
        moment_count=config.moment_count,
        seed=config.seed,
        ranktol=config.ranktol,
        residual_tol=config.residual_tol,
        match_atol=config.match_atol,
        print_rows=false,
    ))
    fused_s = (time_ns() - fused_start) / 1e9

    Random.seed!(config.seed + 17)
    subspace_cols = max(feast_subspace_multiplier * target_count + feast_extra_columns, target_count + 4)
    A_sparse = sparse(Matrix(op.full_operator))
    X = rand(ComplexF64, size(op.full_operator, 1), subspace_cols)
    stats = DenseFeastStats()
    feast_start = time_ns()
    feast_values, _, feast_residuals = feast!(
        X,
        A_sparse;
        nodes=feast_nodes,
        iter=feast_iter,
        c=config.center,
        r=config.radius,
        ϵ=feast_tol,
        store=false,
        stats=stats,
    )
    feast_s = (time_ns() - feast_start) / 1e9
    feast_matched = match_expected_count(feast_values[feast_residuals .<= feast_tol], expected; atol=config.match_atol)

    if print_rows
        println()
        println("Fused Schrodinger/DD baseline comparison")
        println("  compares compressed nonlinear interface solve against direct full eigvals and sparse full linear FEAST")
        @printf(
            "  full_n=%d interface_n=%d compression=%.1f expected=%d\n",
            size(op.full_operator, 1),
            length(op.interface),
            size(op.full_operator, 1) / length(op.interface),
            target_count,
        )
        @printf(
            "  direct eigvals: count=%d elapsed=%.3fs\n",
            target_count,
            direct_s,
        )
        @printf(
            "  fused DD: matched=%d diagnosis=%s residual=%.3e elapsed=%.3fs nodes=%d\n",
            fused.refined_matched,
            string(fused.diagnosis),
            fused.refined_max_inside_residual,
            fused_s,
            config.nodes,
        )
        @printf(
            "  full sparse FEAST: returned=%d matched=%d max_res=%.3e elapsed=%.3fs nodes=%d m=%d iterations=%d\n",
            length(feast_values),
            feast_matched,
            isempty(feast_residuals) ? Inf : maximum(feast_residuals),
            feast_s,
            feast_nodes,
            subspace_cols,
            stats.iterations,
        )
    end

    (
        full_n=size(op.full_operator, 1),
        interface_n=length(op.interface),
        expected=target_count,
        compression_ratio=size(op.full_operator, 1) / length(op.interface),
        direct_count=target_count,
        direct_s=direct_s,
        fused_matched=fused.refined_matched,
        fused_diagnosis=fused.diagnosis,
        fused_residual=fused.refined_max_inside_residual,
        fused_s=fused_s,
        feast_returned=length(feast_values),
        feast_matched=feast_matched,
        feast_max_residual=isempty(feast_residuals) ? Inf : maximum(feast_residuals),
        feast_s=feast_s,
        feast_iterations=stats.iterations,
        feast_subspace_cols=subspace_cols,
    )
end

function fused_schrodinger_dd_packet_state(config::FusedSchrodingerDDConfig)
    problem = schrodinger_dd_interface_context(;
        subdomains=config.subdomains,
        interior_per_subdomain=config.interior_per_subdomain,
        potential_amplitude=config.potential_amplitude,
        potential_frequency=config.potential_frequency,
        center=config.center,
        radius=config.radius,
    )
    target_count = length(problem.expected)
    Random.seed!(config.seed)
    probe_cols = max(target_count + 2, problem.ctx.n)
    right_probe = rand(ComplexF64, problem.ctx.n, probe_cols)
    left_probe = rand(ComplexF64, problem.ctx.n, probe_cols)
    cache = build_contour_sample_cache(
        problem.ctx.Tsolve,
        problem.ctx.Tadjoint_solve,
        right_probe,
        left_probe,
        problem.chart,
        config.nodes;
        source=:fused_schrodinger_dd_packet_state,
    )
    right_blocks = right_moments(cache, config.moment_count)
    left_blocks = left_moments(cache, config.moment_count)
    Xbasis, _ = moment_block_basis(right_blocks, config.moment_count; ranktol=config.ranktol)
    Ybasis, _ = moment_block_basis(left_blocks, config.moment_count; ranktol=config.ranktol)
    d = min(size(Xbasis, 2), size(Ybasis, 2))
    Xbasis = Xbasis[:, 1:d]
    Ybasis = Ybasis[:, 1:d]

    Xfused, Sfused, fused_rank, fused_singulars = projected_hankel_pair_identity(
        right_blocks,
        left_probe,
        config.moment_count;
        ranktol=config.ranktol,
        maxrank=target_count,
    )
    Ffused = eigen(Sfused)
    raw_values = config.center .+ config.radius .* ComplexF64.(Ffused.values)
    Tred = z -> adjoint(Ybasis) * problem.ctx.Tmatrix(z) * Xbasis
    Tred_derivative = z -> adjoint(Ybasis) * problem.ctx.Tderivative(z) * Xbasis
    raw_left_reduced, raw_right_reduced = reduced_left_right_singular_vectors(Tred, raw_values)
    raw_right = Xbasis * raw_right_reduced
    raw_left = Ybasis * raw_left_reduced
    normalize_columns_local!(raw_right)
    normalize_columns_local!(raw_left)
    raw_residuals = matrix_vector_residuals(problem.ctx.Tmatrix, raw_values, raw_right; normalization=:vector)
    raw_inside = FEASTSolver.in_contour(raw_values, config.center, config.radius)
    raw_good = raw_inside .& (raw_residuals .<= config.residual_tol)

    refined_values, corrections = refine_reduced_analytic_triplets(
        Tred,
        Tred_derivative,
        raw_values;
        steps=6,
        step_limit=0.25 * config.radius,
    )
    refined_left_reduced, refined_right_reduced = reduced_left_right_singular_vectors(Tred, refined_values)
    refined_right = Xbasis * refined_right_reduced
    refined_left = Ybasis * refined_left_reduced
    normalize_columns_local!(refined_right)
    normalize_columns_local!(refined_left)
    refined_residuals = matrix_vector_residuals(problem.ctx.Tmatrix, refined_values, refined_right; normalization=:vector)
    refined_inside = FEASTSolver.in_contour(refined_values, config.center, config.radius)
    refined_good = refined_inside .& (refined_residuals .<= config.residual_tol)
    refined_matched = match_expected_count(refined_values[refined_good], problem.expected; atol=config.match_atol)

    (
        problem=problem,
        target_count=target_count,
        rank=fused_rank,
        singulars=Float64.(fused_singulars),
        Xbasis=Xbasis,
        Ybasis=Ybasis,
        raw=(values=raw_values, right=raw_right, left=raw_left, residuals=raw_residuals, inside=raw_inside, good=raw_good),
        refined=(
            values=refined_values,
            right=refined_right,
            left=refined_left,
            residuals=refined_residuals,
            inside=refined_inside,
            good=refined_good,
            matched=refined_matched,
            corrections=corrections,
        ),
    )
end

function run_fused_schrodinger_dd_packet_defect_diagnostic(;
    config=FusedSchrodingerDDConfig(),
    reference_nodes=256,
    candidate_nodes=(64, 96, 128),
    print_rows=true,
)
    reference_config = FusedSchrodingerDDConfig(;
        subdomains=config.subdomains,
        interior_per_subdomain=config.interior_per_subdomain,
        potential_amplitude=config.potential_amplitude,
        potential_frequency=config.potential_frequency,
        center=config.center,
        radius=config.radius,
        nodes=reference_nodes,
        moment_count=config.moment_count,
        seed=config.seed,
        ranktol=config.ranktol,
        residual_tol=config.residual_tol,
        match_atol=config.match_atol,
        print_rows=false,
    )
    reference = fused_schrodinger_dd_packet_state(reference_config)
    Preference = oblique_packet_projector(reference.refined.right, reference.refined.left; ranktol=config.ranktol)
    rows = map(candidate_nodes) do nodes
        candidate_config = FusedSchrodingerDDConfig(;
            subdomains=config.subdomains,
            interior_per_subdomain=config.interior_per_subdomain,
            potential_amplitude=config.potential_amplitude,
            potential_frequency=config.potential_frequency,
            center=config.center,
            radius=config.radius,
            nodes=nodes,
            moment_count=config.moment_count,
            seed=config.seed,
            ranktol=config.ranktol,
            residual_tol=config.residual_tol,
            match_atol=config.match_atol,
            print_rows=false,
        )
        candidate = fused_schrodinger_dd_packet_state(candidate_config)
        Praw = oblique_packet_projector(candidate.raw.right, candidate.raw.left; ranktol=config.ranktol)
        Prefined = oblique_packet_projector(candidate.refined.right, candidate.refined.left; ranktol=config.ranktol)
        raw_split = projector_defect_split(Praw, Preference)
        refined_split = projector_defect_split(Prefined, Preference)
        schedule = packet_schedule_proxy(Praw, Prefined, Preference)
        monitor = packet_schedule_monitor(schedule; inner_steps=nodes)
        raw_compatibility = packet_trial_test_compatibility(
            candidate.raw.right,
            candidate.raw.left,
            candidate.Xbasis,
            candidate.Ybasis;
            ranktol=config.ranktol,
        )
        refined_compatibility = packet_trial_test_compatibility(
            candidate.refined.right,
            candidate.refined.left,
            candidate.Xbasis,
            candidate.Ybasis;
            ranktol=config.ranktol,
        )
        row = (
            nodes=nodes,
            rank=candidate.rank,
            raw_good=count(candidate.raw.good),
            raw_max=any(candidate.raw.inside) ? maximum(candidate.raw.residuals[candidate.raw.inside]) : Inf,
            refined_good=count(candidate.refined.good),
            refined_matched=candidate.refined.matched,
            refined_max=any(candidate.refined.inside) ? maximum(candidate.refined.residuals[candidate.refined.inside]) : Inf,
            max_correction=isempty(candidate.refined.corrections) ? 0.0 : maximum(candidate.refined.corrections),
            raw_defect=raw_split,
            refined_defect=refined_split,
            schedule=schedule,
            monitor=monitor,
            raw_trial_test_compatibility=raw_compatibility,
            refined_trial_test_compatibility=refined_compatibility,
        )
        status = packet_defect_status(row, reference.target_count)
        acceptance = packet_acceptance_certificate(row, reference.target_count)
        staged = merge(row, (status=status, acceptance=acceptance))
        merge(staged, (update_stage=packet_update_stage(staged),))
    end

    if print_rows
        println()
        println("Fused Schrodinger/DD packet-defect diagnostic")
        println("  Lean-facing proxy: split candidate packet-projector defect into reference-packet-visible and invisible parts")
        @printf(
            "  reference_nodes=%d expected=%d reference_rank=%d reference_matched=%d\n",
            reference_nodes,
            reference.target_count,
            reference.rank,
            reference.refined.matched,
        )
        for row in rows
            @printf(
                "  nodes=%d status=%s rank=%d raw_good=%d raw_max=%.3e refined=%d/%d refined_max=%.3e raw_visible=%.3e refined_visible=%.3e contraction=%.3e q_current=%.3e q_error=%.3e visible_repair=%.3e repair=%.3e compat=%.3e coord_relerr=%.3e\n",
                row.nodes,
                string(row.status),
                row.rank,
                row.raw_good,
                row.raw_max,
                row.refined_matched,
                reference.target_count,
                row.refined_max,
                row.raw_defect.visible,
                row.refined_defect.visible,
                row.schedule.visible_contraction,
                row.schedule.q_current,
                row.schedule.q_error,
                row.schedule.visible_repair,
                row.monitor.observed_scheduled_repair,
                row.refined_trial_test_compatibility.max_gap,
                row.schedule.correction_coordinate_relative_error,
            )
        end
    end

    (
        expected=reference.target_count,
        reference_nodes=reference_nodes,
        reference_rank=reference.rank,
        reference_matched=reference.refined.matched,
        rows=rows,
    )
end

function run_fused_schrodinger_dd_packet_policy_diagnostic(;
    config=FusedSchrodingerDDConfig(),
    reference_nodes=256,
    candidate_nodes=(64, 96, 128),
    print_rows=true,
)
    diagnostic = run_fused_schrodinger_dd_packet_defect_diagnostic(;
        config=config,
        reference_nodes=reference_nodes,
        candidate_nodes=candidate_nodes,
        print_rows=false,
    )
    rows = map(diagnostic.rows) do row
        action = packet_defect_policy_action(row.status, row.acceptance)
        merge(row, (action=action,))
    end
    steering = packet_steering_trace(rows)
    selected = steering.selected

    if print_rows
        println()
        println("Fused Schrodinger/DD packet policy diagnostic")
        println("  Lean-guided action policy using packet-visible defect status")
        for row in rows
            @printf(
                "  nodes=%d status=%s action=%s update_stage=%s accepted=%s score=%d/%d matched=%d/%d visible=%.3e contraction=%.3e q_current=%.3e q_error=%.3e repair=%.3e outer_budget=%.3e coord_relerr=%.3e\n",
                row.nodes,
                string(row.status),
                string(row.action),
                string(row.update_stage),
                string(row.acceptance.accepted),
                row.acceptance.score,
                row.acceptance.total,
                row.refined_matched,
                diagnostic.expected,
                row.refined_defect.visible,
                row.schedule.visible_contraction,
                row.schedule.q_current,
                row.schedule.q_error,
                row.monitor.observed_scheduled_repair,
                row.monitor.outer_visible_budget,
                row.schedule.correction_coordinate_relative_error,
            )
        end
        if selected === nothing
            println("  selected: none")
        else
            @printf("  selected: nodes=%d action=%s\n", selected.nodes, string(selected.action))
        end
    end

    (
        expected=diagnostic.expected,
        reference_nodes=diagnostic.reference_nodes,
        rows=rows,
        steering=steering,
        selected_nodes=selected === nothing ? nothing : selected.nodes,
        selected_status=selected === nothing ? nothing : selected.status,
        selected_action=selected === nothing ? nothing : selected.action,
    )
end

function run_fused_schrodinger_dd_packet_steering_diagnostic(;
    config=FusedSchrodingerDDConfig(),
    reference_nodes=256,
    node_ladder=(64, 96, 128),
    print_rows=true,
)
    result = run_fused_schrodinger_dd_packet_policy_diagnostic(;
        config=config,
        reference_nodes=reference_nodes,
        candidate_nodes=node_ladder,
        print_rows=false,
    )
    steering = result.steering

    if print_rows
        println()
        println("Fused Schrodinger/DD packet steering diagnostic")
        println("  consumes the node ladder as an adaptive packet/update-stage policy")
        for (idx, row) in enumerate(result.rows)
            @printf(
                "  step=%d nodes=%d action=%s update_stage=%s accepted=%s score=%d/%d visible=%.3e outer_budget=%.3e\n",
                idx,
                row.nodes,
                string(row.action),
                string(row.update_stage),
                string(row.acceptance.accepted),
                row.acceptance.score,
                row.acceptance.total,
                row.refined_defect.visible,
                row.monitor.outer_visible_budget,
            )
            row.acceptance.accepted && break
        end
        if steering.accepted
            @printf(
                "  selected: step=%d nodes=%d\n",
                steering.selected_index,
                steering.selected.nodes,
            )
        else
            @printf(
                "  selected: none final_action=%s final_update_stage=%s\n",
                string(steering.final_action),
                string(steering.final_update_stage),
            )
        end
    end

    (
        expected=result.expected,
        reference_nodes=result.reference_nodes,
        node_ladder=node_ladder,
        rows=result.rows,
        steering=steering,
        selected_nodes=result.selected_nodes,
    )
end

function run_fused_schrodinger_dd_packet_chart_ladder_diagnostic(;
    config=FusedSchrodingerDDConfig(),
    reference_nodes=192,
    nodes=64,
    radius_ladder=(0.8 * config.radius, config.radius, 1.2 * config.radius),
    print_rows=true,
)
    rows = map(radius_ladder) do radius
        chart_config = FusedSchrodingerDDConfig(;
            subdomains=config.subdomains,
            interior_per_subdomain=config.interior_per_subdomain,
            potential_amplitude=config.potential_amplitude,
            potential_frequency=config.potential_frequency,
            center=config.center,
            radius=radius,
            nodes=config.nodes,
            moment_count=config.moment_count,
            seed=config.seed,
            ranktol=config.ranktol,
            residual_tol=config.residual_tol,
            match_atol=config.match_atol,
            print_rows=false,
        )
        result = run_fused_schrodinger_dd_packet_policy_diagnostic(;
            config=chart_config,
            reference_nodes=reference_nodes,
            candidate_nodes=(nodes,),
            print_rows=false,
        )
        row = only(result.rows)
        merge(row, (radius=radius, expected=result.expected))
    end
    target_index = findfirst(radius -> radius == config.radius, radius_ladder)
    target_expected = target_index === nothing ? rows[1].expected : rows[target_index].expected
    trace = packet_chart_ladder_trace(rows, target_expected)

    if print_rows
        println()
        println("Fused Schrodinger/DD packet chart ladder diagnostic")
        println("  fixed low-node solve across nearby chart radii; accepted changed-count packets are not accepted for the original target")
        @printf(
            "  nodes=%d reference_nodes=%d target_radius=%.3e target_expected=%d action=%s\n",
            nodes,
            reference_nodes,
            config.radius,
            target_expected,
            string(trace.action),
        )
        for row in rows
            @printf(
                "  radius=%.3e expected=%d status=%s accepted=%s matched=%d visible=%.3e contraction=%.3e\n",
                row.radius,
                row.expected,
                string(row.status),
                string(row.acceptance.accepted),
                row.refined_matched,
                row.refined_defect.visible,
                row.schedule.visible_contraction,
            )
        end
    end

    (
        target_expected=target_expected,
        reference_nodes=reference_nodes,
        nodes=nodes,
        radius_ladder=radius_ladder,
        rows=rows,
        trace=trace,
    )
end
