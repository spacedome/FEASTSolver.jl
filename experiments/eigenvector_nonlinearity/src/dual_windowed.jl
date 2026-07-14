Base.@kwdef struct DualWindowConfig
    moment_depth::Int = 1
    probe_width::Int
    window_blocks::Int = 4
    refresh_iterations::Int = 60
    inner_iterations::Int = 3
    inner_mixing::Float64 = 0.4
    inner_history_depth::Int = 8
    inner_regularization::Float64 = 1e-12
    maximum_step_ratio::Float64 = 4.0
    density_tolerance::Float64 = 1e-9
    residual_tolerance::Float64 = 1e-9
    subspace_ranktol::Float64 = 1e-10
    overlap_ranktol::Float64 = 1e-10
end

function dual_raw_moment_block(H, chart::CircularChart, right_probe, left_probe, depth)
    size(right_probe) == size(left_probe) || throw(DimensionMismatch(
        "dual probes must have equal dimensions",
    ))
    factors = contour_factorizations(H, chart)
    count_diagnostic = determinant_count(factors)
    right_moments = [zeros(ComplexF64, size(right_probe)) for _ in 1:depth]
    left_moments = [zeros(ComplexF64, size(left_probe)) for _ in 1:depth]
    for (z, weight, factorization) in zip(chart.nodes, chart.weights, factors)
        right_response = factorization \ right_probe
        left_response = adjoint(factorization) \ left_probe
        coordinate = (z - chart.center) / chart.radius
        right_power = one(ComplexF64)
        left_power = one(ComplexF64)
        for index in 1:depth
            right_moments[index] .+= (weight * right_power) .* right_response
            left_moments[index] .+= (conj(weight) * left_power) .* left_response
            right_power *= coordinate
            left_power *= conj(coordinate)
        end
    end
    (
        right=reduce(hcat, right_moments),
        left=reduce(hcat, left_moments),
        count_diagnostic=count_diagnostic,
        solve_count=length(chart.nodes),
        rhs_count=2length(chart.nodes) * size(right_probe, 2),
    )
end

function dual_window_basis(right_blocks, left_blocks, subspace_ranktol, overlap_ranktol)
    right_window = window_basis(right_blocks, subspace_ranktol)
    left_window = window_basis(left_blocks, subspace_ranktol)
    maximum_count = min(right_window.rank, left_window.rank)
    overlap = adjoint(left_window.basis) * right_window.basis
    overlap_svd = svd(overlap)
    overlap_count = Base.count(
        singular -> singular >= overlap_ranktol * overlap_svd.S[1],
        overlap_svd.S,
    )
    retained_count = min(maximum_count, overlap_count)
    retained_count > 0 || error("dual accumulated space has zero overlap rank")
    inverse_root = Diagonal(1.0 ./ sqrt.(overlap_svd.S[1:retained_count]))
    right = right_window.basis * overlap_svd.V[:, 1:retained_count] * inverse_root
    left = left_window.basis * overlap_svd.U[:, 1:retained_count] * inverse_root
    (
        right=right,
        left=left,
        rank=retained_count,
        overlap_singular_values=Float64.(overlap_svd.S),
    )
end

function dual_reduced_state(
    problem::SimilarityMeanField1D,
    chart::CircularChart,
    right_basis,
    left_basis,
    rho,
)
    H = hamiltonian(problem, rho)
    reduced = adjoint(left_basis) * H * right_basis
    decomposition = eigen(reduced)
    indices = findall(
        value -> abs(value - chart.center) < chart.radius,
        decomposition.values,
    )
    length(indices) == problem.base.occupied || error(
        "dual reduced extraction found $(length(indices)) target states, expected $(problem.base.occupied)",
    )
    right_coefficients = decomposition.vectors
    left_coefficients = inv(right_coefficients)'
    right = right_basis * right_coefficients[:, indices]
    left = left_basis * left_coefficients[:, indices]
    right, left = biorthogonalize(right, left, problem.base.occupied)
    output_density = oblique_density(problem, right, left)
    (
        right=right,
        left=left,
        density=output_density,
        values=ComplexF64.(decomposition.values[indices]),
        residual=dual_invariant_residual(problem, rho, right, left),
        hamiltonian=H,
    )
end

function solve_dual_windowed_nlfeast(
    problem::SimilarityMeanField1D,
    chart::CircularChart,
    initial_right,
    initial_left;
    config::DualWindowConfig,
)
    occupied = problem.base.occupied
    config.moment_depth * config.probe_width >= occupied || throw(ArgumentError(
        "dual moment capacity is below the occupied count",
    ))
    config.window_blocks >= 0 || throw(ArgumentError(
        "window_blocks must be nonnegative; zero selects growing memory",
    ))
    config.refresh_iterations > 0 || throw(ArgumentError(
        "refresh_iterations must be positive",
    ))
    config.inner_iterations > 0 || throw(ArgumentError(
        "inner_iterations must be positive",
    ))
    config.inner_history_depth > 0 || throw(ArgumentError(
        "inner_history_depth must be positive",
    ))
    0 < config.inner_mixing <= 1 || throw(ArgumentError(
        "inner_mixing must lie in (0,1]",
    ))
    right, left = biorthogonalize(initial_right, initial_left, occupied)
    rho = oblique_density(problem, right, left)
    tangent = moment_tangent(occupied, config.probe_width)
    right_blocks = Matrix{ComplexF64}[]
    left_blocks = Matrix{ComplexF64}[]
    right_basis = right
    left_basis = left
    history = NamedTuple[]
    converged = false

    for refresh in 1:config.refresh_iterations
        filtered = dual_raw_moment_block(
            hamiltonian(problem, rho),
            chart,
            right * tangent,
            left * tangent,
            config.moment_depth,
        )
        filtered.count_diagnostic.count == occupied || throw(ContourCountError(
            occupied,
            filtered.count_diagnostic.count,
            filtered.count_diagnostic.winding,
            filtered.count_diagnostic.maximum_phase_step,
        ))
        push!(right_blocks, filtered.right)
        push!(left_blocks, filtered.left)
        while config.window_blocks > 0 && length(right_blocks) > config.window_blocks
            popfirst!(right_blocks)
            popfirst!(left_blocks)
        end
        window = dual_window_basis(
            right_blocks,
            left_blocks,
            config.subspace_ranktol,
            config.overlap_ranktol,
        )
        window.rank >= occupied || error(
            "dual accumulated basis rank is below the occupied count",
        )
        right_basis = window.right
        left_basis = window.left
        densities = Vector{Float64}[]
        residuals = Vector{Float64}[]
        state = dual_reduced_state(problem, chart, right_basis, left_basis, rho)
        density_defect = Inf
        performed = 0
        for inner in 1:config.inner_iterations
            state = dual_reduced_state(problem, chart, right_basis, left_basis, rho)
            fixed_point_residual = state.density - rho
            density_defect = norm(fixed_point_residual) / max(norm(rho), eps(Float64))
            performed = inner
            if density_defect <= config.density_tolerance &&
                    state.residual <= config.residual_tolerance
                converged = true
                break
            end
            push!(densities, copy(rho))
            push!(residuals, fixed_point_residual)
            while length(densities) > config.inner_history_depth + 1
                popfirst!(densities)
                popfirst!(residuals)
            end
            correction = anderson_correction(
                densities,
                residuals,
                config.inner_mixing,
                config.inner_regularization,
            )
            maximum_step = config.maximum_step_ratio * max(
                norm(fixed_point_residual),
                eps(Float64),
            )
            norm(correction) > maximum_step && (
                correction .*= maximum_step / norm(correction)
            )
            rho += correction
            normalize_density!(problem.base, rho)
        end
        state = dual_reduced_state(problem, chart, right_basis, left_basis, rho)
        right = state.right
        left = state.left
        density_defect = norm(state.density - rho) / max(norm(rho), eps(Float64))
        converged = density_defect <= config.density_tolerance &&
            state.residual <= config.residual_tolerance
        push!(history, (
            refresh=refresh,
            inner_iterations=performed,
            density_defect=density_defect,
            residual=state.residual,
            basis_dimension=window.rank,
            overlap_ratio=window.overlap_singular_values[window.rank] /
                window.overlap_singular_values[1],
            solve_count=filtered.solve_count,
            rhs_count=filtered.rhs_count,
        ))
        converged && break
    end

    final_state = dual_reduced_state(problem, chart, right_basis, left_basis, rho)
    (
        variant=config.window_blocks == 0 ? :dual_growing : :dual_window,
        inner_method=:anderson,
        right=final_state.right,
        left=final_state.left,
        density=rho,
        right_basis=right_basis,
        left_basis=left_basis,
        history=history,
        converged=converged,
        density_defect=norm(final_state.density - rho) / max(norm(rho), eps(Float64)),
        residual=final_state.residual,
        refresh_count=length(history),
        inner_iterations=sum(record.inner_iterations for record in history),
        solve_count=sum(record.solve_count for record in history),
        rhs_count=sum(record.rhs_count for record in history),
    )
end
