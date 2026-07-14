export FrozenDensityCache
export build_density_cache, cache_stats, corrected_moments!
export density_response!, invalidate!, positive_moments!
export cached_projector_step!

mutable struct FrozenDensityCache{P,C,H,F}
    problem::P
    chart::C
    density::Vector{Float64}
    hamiltonian::H
    factors::F
    valid::Bool
    factorization_count::Int
    base_rhs_count::Int
    correction_rhs_count::Int
    response_rhs_count::Int
    response_actions::Int
end

function build_density_cache(problem, chart::PMF.CircularChart, rho)
    H = PMF.hamiltonian(problem, rho)
    factors = PMF.contour_factorizations(H, chart)
    FrozenDensityCache(
        problem,
        chart,
        Float64.(rho),
        H,
        factors,
        true,
        length(factors),
        0,
        0,
        0,
        0,
    )
end

function assert_valid(cache::FrozenDensityCache)
    cache.valid || throw(ArgumentError(
        "the frozen contour cache was invalidated by a nonlinear-state update",
    ))
end

function invalidate!(cache::FrozenDensityCache)
    cache.valid = false
    cache
end

cache_stats(cache::FrozenDensityCache) = (
    valid=cache.valid,
    factorization_count=cache.factorization_count,
    base_rhs_count=cache.base_rhs_count,
    correction_rhs_count=cache.correction_rhs_count,
    response_rhs_count=cache.response_rhs_count,
    total_rhs_count=cache.base_rhs_count + cache.correction_rhs_count +
        cache.response_rhs_count,
    response_actions=cache.response_actions,
)

function positive_moments!(cache::FrozenDensityCache, probe, count)
    assert_valid(cache)
    count > 0 || throw(ArgumentError("moment count must be positive"))
    moments = [
        zeros(ComplexF64, size(cache.hamiltonian, 1), size(probe, 2))
        for _ in 1:count
    ]
    for (z, weight, factor) in zip(
        cache.chart.nodes,
        cache.chart.weights,
        cache.factors,
    )
        response = factor \ probe
        coordinate = (z - cache.chart.center) / cache.chart.radius
        power = one(ComplexF64)
        for moment in moments
            moment .+= (weight * power) .* response
            power *= coordinate
        end
    end
    cache.base_rhs_count += length(cache.chart.nodes) * size(probe, 2)
    moments
end

function explicit_corrected_moments!(
    cache::FrozenDensityCache,
    orbitals,
    state,
    count;
    tangent=Matrix{ComplexF64}(I, size(orbitals, 2), size(orbitals, 2)),
)
    assert_valid(cache)
    size(state) == (size(orbitals, 2), size(orbitals, 2)) || throw(
        DimensionMismatch("the invariant-pair state has the wrong dimensions"),
    )
    size(tangent, 1) == size(orbitals, 2) || throw(DimensionMismatch(
        "the tangent must have one row per invariant-pair column",
    ))
    residual = orbitals * state - cache.hamiltonian * orbitals
    moments = [
        zeros(ComplexF64, size(orbitals, 1), size(tangent, 2))
        for _ in 1:count
    ]
    identity_state = Matrix{ComplexF64}(I, size(state, 1), size(state, 2))
    for (z, weight, factor) in zip(
        cache.chart.nodes,
        cache.chart.weights,
        cache.factors,
    )
        corrected = (orbitals - factor \ residual) /
            (z .* identity_state .- state)
        response = corrected * tangent
        coordinate = (z - cache.chart.center) / cache.chart.radius
        power = one(ComplexF64)
        for moment in moments
            moment .+= (weight * power) .* response
            power *= coordinate
        end
    end
    cache.correction_rhs_count += length(cache.chart.nodes) * size(residual, 2)
    moments
end

function corrected_moments!(
    cache::FrozenDensityCache,
    orbitals,
    state,
    count;
    tangent=Matrix{ComplexF64}(I, size(orbitals, 2), size(orbitals, 2)),
    realization=:linear_identity,
)
    if realization === :linear_identity
        positive_moments!(cache, orbitals * tangent, count)
    elseif realization === :explicit
        explicit_corrected_moments!(
            cache,
            orbitals,
            state,
            count;
            tangent=tangent,
        )
    else
        throw(ArgumentError("realization must be :linear_identity or :explicit"))
    end
end

function density_response!(cache::FrozenDensityCache, orbitals, values, direction)
    assert_valid(cache)
    result = PMF.contour_density_response(
        cache.problem,
        orbitals,
        values,
        direction,
        cache.chart;
        factorizations=cache.factors,
    )
    cache.response_actions += 1
    cache.response_rhs_count += length(cache.chart.nodes) * size(orbitals, 2)
    result
end

function extraction_from_moments(
    cache::FrozenDensityCache,
    probe,
    moments,
    depth,
    target_count,
    ranktol,
)
    H0, H1 = PMF.block_hankel(moments, depth, probe)
    decomposition = svd(H0)
    length(decomposition.S) >= target_count || error(
        "cached moment pencil is smaller than the target count",
    )
    decomposition.S[target_count] >= ranktol * decomposition.S[1] || error(
        "cached moment pencil has numerical rank below the target count",
    )
    U = decomposition.U[:, 1:target_count]
    V = decomposition.V[:, 1:target_count]
    inverse_singulars = Diagonal(1.0 ./ decomposition.S[1:target_count])
    coordinate_state = adjoint(U) * H1 * V * inverse_singulars
    physical_state = cache.chart.center .* I + cache.chart.radius .* coordinate_state
    moment_row = reduce(hcat, moments[1:depth])
    output = moment_row * V * inverse_singulars
    basis = PMF.orthonormalize(output, target_count)
    reduced = eigen(Hermitian(adjoint(basis) * cache.hamiltonian * basis))
    orbitals = basis * reduced.vectors
    (
        orbitals=orbitals,
        values=Float64.(reduced.values),
        state=Matrix{ComplexF64}(physical_state),
        singular_values=Float64.(decomposition.S),
    )
end

function cached_projector_step!(
    cache::FrozenDensityCache,
    orbitals,
    moment_depth,
    probe_width;
    ranktol=1e-12,
    corrected_realization=:linear_identity,
)
    tangent = PMF.moment_tangent(size(orbitals, 2), probe_width)
    reduced_state = adjoint(orbitals) * cache.hamiltonian * orbitals
    moments = corrected_moments!(
        cache,
        orbitals,
        reduced_state,
        2moment_depth;
        tangent=tangent,
        realization=corrected_realization,
    )
    probe = orbitals * tangent
    extracted = extraction_from_moments(
        cache,
        probe,
        moments,
        moment_depth,
        cache.problem.occupied,
        ranktol,
    )
    merge(extracted, (
        moments=moments,
        window_block=reduce(hcat, moments[1:moment_depth]),
        probe=probe,
    ))
end
