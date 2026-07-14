export FrozenDualCache
export build_dual_cache, dual_cache_stats, dual_density_response!
export dual_positive_moments!

mutable struct FrozenDualCache{P,C,H,F}
    problem::P
    chart::C
    density::Vector{Float64}
    hamiltonian::H
    factors::F
    valid::Bool
    factorization_count::Int
    right_rhs_count::Int
    left_rhs_count::Int
    response_rhs_count::Int
    response_actions::Int
end

function build_dual_cache(problem, chart::PMF.CircularChart, rho)
    H = PMF.hamiltonian(problem, rho)
    factors = PMF.contour_factorizations(H, chart)
    FrozenDualCache(
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

function assert_valid(cache::FrozenDualCache)
    cache.valid || throw(ArgumentError(
        "the frozen dual cache was invalidated by a nonlinear-state update",
    ))
end

function invalidate!(cache::FrozenDualCache)
    cache.valid = false
    cache
end

dual_cache_stats(cache::FrozenDualCache) = (
    valid=cache.valid,
    factorization_count=cache.factorization_count,
    right_rhs_count=cache.right_rhs_count,
    left_rhs_count=cache.left_rhs_count,
    response_rhs_count=cache.response_rhs_count,
    total_rhs_count=cache.right_rhs_count + cache.left_rhs_count +
        cache.response_rhs_count,
    response_actions=cache.response_actions,
)

function dual_positive_moments!(cache::FrozenDualCache, right_probe, left_probe, depth)
    assert_valid(cache)
    size(right_probe) == size(left_probe) || throw(DimensionMismatch(
        "right and left probes must have equal dimensions",
    ))
    right_moments = [zeros(ComplexF64, size(right_probe)) for _ in 1:depth]
    left_moments = [zeros(ComplexF64, size(left_probe)) for _ in 1:depth]
    for (z, weight, factor) in zip(
        cache.chart.nodes,
        cache.chart.weights,
        cache.factors,
    )
        right_response = factor \ right_probe
        left_response = adjoint(factor) \ left_probe
        coordinate = (z - cache.chart.center) / cache.chart.radius
        right_power = one(ComplexF64)
        left_power = one(ComplexF64)
        for index in 1:depth
            right_moments[index] .+= (weight * right_power) .* right_response
            left_moments[index] .+=
                (conj(weight) * left_power) .* left_response
            right_power *= coordinate
            left_power *= conj(coordinate)
        end
    end
    node_count = length(cache.chart.nodes)
    cache.right_rhs_count += node_count * size(right_probe, 2)
    cache.left_rhs_count += node_count * size(left_probe, 2)
    (
        right=reduce(hcat, right_moments),
        left=reduce(hcat, left_moments),
    )
end

function dual_modal_state(cache, right_basis, left_basis)
    reduced = adjoint(left_basis) * cache.hamiltonian * right_basis
    decomposition = eigen(reduced)
    indices = findall(
        value -> abs(value - cache.chart.center) < cache.chart.radius,
        decomposition.values,
    )
    length(indices) == cache.problem.base.occupied || error(
        "dual modal extraction found $(length(indices)) target states, expected $(cache.problem.base.occupied)",
    )
    right_coefficients = decomposition.vectors[:, indices]
    left_coefficients = adjoint(inv(decomposition.vectors))[:, indices]
    right = right_basis * right_coefficients
    left = left_basis * left_coefficients
    overlap = adjoint(left) * right
    left = left * adjoint(inv(overlap))
    (
        right=right,
        left=left,
        values=ComplexF64.(decomposition.values[indices]),
        density=PMF.oblique_density(cache.problem, right, left),
        residual=PMF.dual_invariant_residual(
            cache.problem,
            cache.density,
            right,
            left,
        ),
    )
end

function dual_density_response!(cache::FrozenDualCache, right, left, values, direction)
    assert_valid(cache)
    length(direction) == length(cache.density) || throw(DimensionMismatch(
        "dual density direction has the wrong length",
    ))
    overlap = adjoint(left) * right
    norm(overlap - I) <= 1e-8 || throw(ArgumentError(
        "dual response requires modal pairs normalized by YᴴX=I",
    ))
    scaling = Diagonal(cache.problem.similarity)
    inverse_scaling = Diagonal(1.0 ./ cache.problem.similarity)
    derivative = scaling *
        Diagonal(cache.problem.base.coupling .* direction) * inverse_scaling
    coupling = adjoint(left) * derivative * right
    right_forcing = derivative * right - right * coupling
    left_forcing = adjoint(derivative) * left - left * adjoint(coupling)
    right_response = zeros(ComplexF64, size(right))
    left_response = zeros(ComplexF64, size(left))
    for (z, weight, factor) in zip(
        cache.chart.nodes,
        cache.chart.weights,
        cache.factors,
    )
        solved_right = factor \ right_forcing
        solved_right ./= reshape(z .- values, 1, :)
        right_response .+= weight .* solved_right
        solved_left = adjoint(factor) \ left_forcing
        solved_left ./= reshape(conj.(z .- values), 1, :)
        left_response .+= conj(weight) .* solved_left
    end
    right_response .-= right * (adjoint(left) * right_response)
    left_response .-= left * (adjoint(right) * left_response)
    cache.response_actions += 1
    cache.response_rhs_count +=
        2length(cache.chart.nodes) * size(right, 2)
    real.(vec(sum(
        right_response .* conj.(left) .+ right .* conj.(left_response);
        dims=2,
    ))) ./ cache.problem.base.spacing
end
