using FEASTSolver
using Test
using LinearAlgebra: diagm

@testset "FEASTSolver.jl" begin

    A = diagm(0 => 1:25)
    # e, v, res = feast!(A, rand(25, 5), 4, 10, c=2.0, r=2.0)
    e, v, res = feast!(A, rand(ComplexF64, 25, 5), 8, 10, c=2.0, r=2.0)
    @test sort(real.(e)) ≈ 1:5
    @test maximum(res) < 1e-8

end
