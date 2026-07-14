export CombinedCacheConfig, FrozenPolynomialCache
export build_polynomial_cache, polynomial_cache_stats
export polynomial_corrected_moments!, solve_cache_native_combined

mutable struct FrozenPolynomialCache{P,C,F}
    problem::P
    chart::C
    density::Vector{Float64}
    factors::F
    valid::Bool
    factorization_count::Int
    base_rhs_count::Int
    correction_rhs_count::Int
    correction_actions::Int
    correction_ranks::Vector{Int}
end

function build_polynomial_cache(problem, chart::PMF.CircularChart, rho)
    factors = [
        lu(PMF.polynomial_matrix(problem, rho, z))
        for z in chart.nodes
    ]
    FrozenPolynomialCache(
        problem,
        chart,
        Float64.(rho),
        factors,
        true,
        length(factors),
        0,
        0,
        0,
        Int[],
    )
end

function assert_valid(cache::FrozenPolynomialCache)
    cache.valid || throw(ArgumentError(
        "the frozen polynomial cache was invalidated by a nonlinear-state update",
    ))
end

function invalidate!(cache::FrozenPolynomialCache)
    cache.valid = false
    cache
end

polynomial_cache_stats(cache::FrozenPolynomialCache) = (
    valid=cache.valid,
    factorization_count=cache.factorization_count,
    base_rhs_count=cache.base_rhs_count,
    correction_rhs_count=cache.correction_rhs_count,
    total_rhs_count=cache.base_rhs_count + cache.correction_rhs_count,
    correction_actions=cache.correction_actions,
    correction_ranks=copy(cache.correction_ranks),
)

function polynomial_corrected_moments!(
    cache::FrozenPolynomialCache,
    orbitals,
    state,
    count,
    probe_width,
    ;
    residual_ranktol=1e-12,
)
    assert_valid(cache)
    occupied = size(orbitals, 2)
    tangent = PMF.moment_tangent(occupied, probe_width)
    probe = orbitals * tangent
    moments = [
        zeros(ComplexF64, size(orbitals, 1), probe_width)
        for _ in 1:count
    ]
    residual = state === nothing ? nothing :
        -PMF.hamiltonian(cache.problem.base, cache.density) * orbitals +
        orbitals * state + cache.problem.quadratic * orbitals * state^2
    residual_basis = residual
    residual_coefficients = nothing
    residual_rank = 0
    if residual !== nothing
        decomposition = svd(residual)
        residual_rank = isempty(decomposition.S) || iszero(decomposition.S[1]) ? 0 :
            Base.count(
                value -> value >= residual_ranktol * decomposition.S[1],
                decomposition.S,
            )
        if residual_rank < size(residual, 2)
            residual_basis = decomposition.U[:, 1:residual_rank]
            residual_coefficients = Diagonal(decomposition.S[1:residual_rank]) *
                decomposition.Vt[1:residual_rank, :]
        end
    end
    state_identity = Matrix{ComplexF64}(I, occupied, occupied)
    for (z, weight, factor) in zip(
        cache.chart.nodes,
        cache.chart.weights,
        cache.factors,
    )
        response = if state === nothing
            factor \ probe
        else
            solved_residual = factor \ residual_basis
            residual_coefficients === nothing ||
                (solved_residual = solved_residual * residual_coefficients)
            corrected = (orbitals - solved_residual) /
                (z .* state_identity .- state)
            corrected * tangent
        end
        coordinate = (z - cache.chart.center) / cache.chart.radius
        power = one(ComplexF64)
        for moment in moments
            moment .+= (weight * power) .* response
            power *= coordinate
        end
    end
    if state === nothing
        cache.base_rhs_count += length(cache.chart.nodes) * probe_width
    else
        cache.correction_rhs_count += length(cache.chart.nodes) * residual_rank
        cache.correction_actions += 1
        push!(cache.correction_ranks, residual_rank)
    end
    (moments=moments, probe=probe)
end

function polynomial_extraction(cache, probe, moments, depth, ranktol)
    occupied = cache.problem.base.occupied
    H0, H1 = PMF.block_hankel(moments, depth, probe)
    decomposition = svd(H0)
    decomposition.S[occupied] >= ranktol * decomposition.S[1] || error(
        "cached polynomial moment pencil has insufficient numerical rank",
    )
    U = decomposition.U[:, 1:occupied]
    V = decomposition.V[:, 1:occupied]
    inverse_singulars = Diagonal(1.0 ./ decomposition.S[1:occupied])
    coordinate_state = adjoint(U) * H1 * V * inverse_singulars
    realized_state = cache.chart.center .* I + cache.chart.radius .* coordinate_state
    raw_map = reduce(hcat, moments[1:depth]) * V * inverse_singulars
    map_qr = qr(raw_map)
    orbitals = Matrix(map_qr.Q)[:, 1:occupied]
    gauge = Matrix(map_qr.R)[1:occupied, 1:occupied]
    state = gauge * realized_state / gauge
    (
        orbitals=orbitals,
        state=state,
        residual=PMF.polynomial_invariant_residual(
            cache.problem,
            cache.density,
            orbitals,
            state,
        ),
        singular_values=Float64.(decomposition.S),
    )
end

Base.@kwdef struct CombinedCacheConfig
    moment_depth::Int
    probe_width::Int
    spectral_corrections::Int = 1
    spectral_tolerance::Float64 = 0.0
    residual_ranktol::Float64 = 1e-12
    window_blocks::Int = 4
    refresh_iterations::Int = 30
    reduced_inner_iterations::Int = 3
    reduced_mixing::Float64 = 0.5
    reduced_history_depth::Int = 8
    reduced_regularization::Float64 = 1e-12
    maximum_step_ratio::Float64 = 4.0
    density_tolerance::Float64 = 1e-9
    residual_tolerance::Float64 = 1e-9
    ranktol::Float64 = 1e-12
    subspace_ranktol::Float64 = 1e-10
end

function validate_combined_config(problem, config::CombinedCacheConfig)
    config.moment_depth * config.probe_width >= problem.base.occupied || throw(
        ArgumentError("combined moment capacity is below the occupied count"),
    )
    config.spectral_corrections > 0 || throw(ArgumentError(
        "spectral_corrections must be positive",
    ))
    config.spectral_tolerance >= 0 || throw(ArgumentError(
        "spectral_tolerance must be nonnegative",
    ))
    config.residual_ranktol >= 0 || throw(ArgumentError(
        "residual_ranktol must be nonnegative",
    ))
    config.window_blocks >= 0 || throw(ArgumentError(
        "window_blocks must be nonnegative; zero selects growing memory",
    ))
    config.refresh_iterations > 0 || throw(ArgumentError(
        "refresh_iterations must be positive",
    ))
    config.reduced_inner_iterations > 0 || throw(ArgumentError(
        "reduced_inner_iterations must be positive",
    ))
end

function combined_reduced_inner(problem, chart, basis, initial_density, config)
    rho = copy(initial_density)
    densities = Vector{Float64}[]
    residuals = Vector{Float64}[]
    performed = 0
    reduced = PMF.quadratic_reduced_state(problem, chart, basis, rho)
    for inner in 1:config.reduced_inner_iterations
        reduced = PMF.quadratic_reduced_state(problem, chart, basis, rho)
        fixed_point_residual = reduced.density - rho
        density_defect = norm(fixed_point_residual) / max(norm(rho), eps(Float64))
        performed = inner
        if density_defect <= config.density_tolerance &&
                reduced.residual <= config.residual_tolerance
            break
        end
        push!(densities, copy(rho))
        push!(residuals, fixed_point_residual)
        while length(densities) > config.reduced_history_depth + 1
            popfirst!(densities)
            popfirst!(residuals)
        end
        correction = PMF.anderson_correction(
            densities,
            residuals,
            config.reduced_mixing,
            config.reduced_regularization,
        )
        maximum_step = config.maximum_step_ratio * max(
            norm(fixed_point_residual),
            eps(Float64),
        )
        norm(correction) > maximum_step && (
            correction .*= maximum_step / norm(correction)
        )
        rho += correction
        PMF.normalize_density!(problem.base, rho)
    end
    reduced = PMF.quadratic_reduced_state(problem, chart, basis, rho)
    density_defect = norm(reduced.density - rho) / max(norm(rho), eps(Float64))
    (
        orbitals=reduced.orbitals,
        state=reduced.state,
        density=rho,
        density_defect=density_defect,
        residual=reduced.residual,
        iterations=performed,
    )
end

function solve_cache_native_combined(
    problem,
    chart::PMF.CircularChart,
    initial;
    config::CombinedCacheConfig,
)
    validate_combined_config(problem, config)
    occupied = problem.base.occupied
    orbitals = PMF.orthonormalize(initial, occupied)
    state = nothing
    rho = PMF.density(problem.base, orbitals)
    blocks = Matrix{ComplexF64}[]
    basis = orbitals
    history = NamedTuple[]
    converged = false

    for refresh in 1:config.refresh_iterations
        cache = build_polynomial_cache(problem, chart, rho)
        count_diagnostic = PMF.determinant_count(cache.factors)
        count_diagnostic.count == occupied || throw(PMF.ContourCountError(
            occupied,
            count_diagnostic.count,
            count_diagnostic.winding,
            count_diagnostic.maximum_phase_step,
        ))
        latest = nothing
        spectral_residuals = Float64[]
        local_orbitals = orbitals
        local_state = state
        for _ in 1:config.spectral_corrections
            sampled = polynomial_corrected_moments!(
                cache,
                local_orbitals,
                local_state,
                2config.moment_depth,
                config.probe_width,
                residual_ranktol=config.residual_ranktol,
            )
            latest = sampled
            extracted = polynomial_extraction(
                cache,
                sampled.probe,
                sampled.moments,
                config.moment_depth,
                config.ranktol,
            )
            local_orbitals = extracted.orbitals
            local_state = extracted.state
            push!(spectral_residuals, extracted.residual)
            config.spectral_tolerance > 0 &&
                extracted.residual <= config.spectral_tolerance && break
        end
        push!(blocks, reduce(hcat, latest.moments[1:config.moment_depth]))
        while config.window_blocks > 0 && length(blocks) > config.window_blocks
            popfirst!(blocks)
        end
        window = PMF.window_basis(blocks, config.subspace_ranktol)
        window.rank >= occupied || error(
            "cache-native combined window rank is below the occupied count",
        )
        basis = window.basis
        invalidate!(cache)
        inner = combined_reduced_inner(problem, chart, basis, rho, config)
        orbitals = inner.orbitals
        state = inner.state
        rho = inner.density
        converged = inner.density_defect <= config.density_tolerance &&
            inner.residual <= config.residual_tolerance
        stats = polynomial_cache_stats(cache)
        push!(history, (
            refresh=refresh,
            spectral_residuals=spectral_residuals,
            reduced_inner_iterations=inner.iterations,
            density_defect=inner.density_defect,
            residual=inner.residual,
            basis_dimension=size(basis, 2),
            factorization_count=stats.factorization_count,
            base_rhs_count=stats.base_rhs_count,
            correction_rhs_count=stats.correction_rhs_count,
            correction_actions=stats.correction_actions,
            correction_ranks=stats.correction_ranks,
        ))
        converged && break
    end

    final = PMF.quadratic_reduced_state(problem, chart, basis, rho)
    (
        variant=:cache_native_combined,
        orbitals=final.orbitals,
        state=final.state,
        density=rho,
        basis=basis,
        history=history,
        converged=converged,
        density_defect=norm(final.density - rho) / max(norm(rho), eps(Float64)),
        residual=final.residual,
        refresh_count=length(history),
        factorization_count=sum(record.factorization_count for record in history),
        base_rhs_count=sum(record.base_rhs_count for record in history),
        correction_rhs_count=sum(record.correction_rhs_count for record in history),
    )
end
