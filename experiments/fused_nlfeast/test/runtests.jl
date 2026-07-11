using LinearAlgebra
using Random
using Test

include(joinpath(@__DIR__, "..", "src", "FusedNLFEAST.jl"))
using .FusedNLFEAST

function multiset_distance(actual, expected)
    length(actual) == length(expected) || return Inf
    isempty(actual) && return 0.0
    order(values) = sort(ComplexF64.(values); by=value -> (real(value), imag(value)))
    maximum(abs.(order(actual) .- order(expected)))
end

include("helpers.jl")
include("matching.jl")
include("cache_lifetime.jl")
include("streaming_realization.jl")
include("rectangular_chart.jl")
include("partitioning.jl")

include("linear_limit.jl")
include("canonical_limit.jl")
include("scalar_many_roots.jl")
include("polynomial_bridge.jl")
include("noncommuting_polynomial.jl")
include("nonnormal_two_sided.jl")
include("loewner_coupling.jl")
include("common_realization.jl")
include("state_identities.jl")
include("analytic_actions.jl")
include("two_sided_state.jl")
include("state_multiplicity.jl")
include("state_iteration.jl")
include("state_driver.jl")
include("state_many_roots.jl")
include("shared_eigenvectors.jl")
include("loewner_extractor.jl")
include("counting.jl")
