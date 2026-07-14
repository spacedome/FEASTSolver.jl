export FrozenProjectorCache, build_projector_cache
export projector_cache_stats, projector_response!

mutable struct FrozenProjectorCache{P,C,H,F}
    problem::P
    chart::C
    projector::Matrix{ComplexF64}
    hamiltonian::H
    factors::F
    valid::Bool
    factorization_count::Int
    response_rhs_count::Int
    response_actions::Int
end

function build_projector_cache(problem, chart::PMF.CircularChart, projector)
    H = PMF.hamiltonian(problem, projector)
    factors = PMF.contour_factorizations(H, chart)
    FrozenProjectorCache(
        problem,
        chart,
        Matrix{ComplexF64}(projector),
        H,
        factors,
        true,
        length(factors),
        0,
        0,
    )
end

function assert_valid(cache::FrozenProjectorCache)
    cache.valid || throw(ArgumentError(
        "the frozen projector cache was invalidated by a nonlinear-state update",
    ))
end

function invalidate!(cache::FrozenProjectorCache)
    cache.valid = false
    cache
end

projector_cache_stats(cache::FrozenProjectorCache) = (
    valid=cache.valid,
    factorization_count=cache.factorization_count,
    response_rhs_count=cache.response_rhs_count,
    response_actions=cache.response_actions,
)

function projector_response!(cache::FrozenProjectorCache, orbitals, values, direction)
    assert_valid(cache)
    size(direction) == size(cache.projector) || throw(DimensionMismatch(
        "projector direction has the wrong dimensions",
    ))
    hermitian_direction = Matrix(Hermitian(
        (Matrix{ComplexF64}(direction) + adjoint(direction)) / 2,
    ))
    derivative = cache.problem.coupling .* (
        cache.problem.kernel * hermitian_direction * cache.problem.kernel
    )
    forcing = derivative * orbitals
    forcing .-= orbitals * (adjoint(orbitals) * forcing)
    orbital_response = zeros(ComplexF64, size(orbitals))
    for (z, weight, factor) in zip(
        cache.chart.nodes,
        cache.chart.weights,
        cache.factors,
    )
        solved = factor \ forcing
        solved ./= reshape(z .- values, 1, :)
        orbital_response .+= weight .* solved
    end
    orbital_response .-= orbitals * (adjoint(orbitals) * orbital_response)
    cache.response_actions += 1
    cache.response_rhs_count +=
        length(cache.chart.nodes) * size(orbitals, 2)
    Matrix(Hermitian(
        orbital_response * adjoint(orbitals) +
        orbitals * adjoint(orbital_response),
    ))
end
