using LinearAlgebra
using Random
using Test

include(joinpath(@__DIR__, "..", "src", "CacheNativeCorrectedMoments.jl"))
using .CacheNativeCorrectedMoments

include(joinpath(
    @__DIR__,
    "..",
    "..",
    "fused_nlfeast",
    "src",
    "FusedNLFEAST.jl",
))

const PMF = CacheNativeCorrectedMoments.ProjectorMomentFEAST

include("cache_algebra.jl")
include("combined.jl")
include("representation_response.jl")
include("policies.jl")
