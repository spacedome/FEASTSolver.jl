using FEASTSolver
using Test
using LinearAlgebra: diagm, Diagonal

function approx_in(x::Number, A)
    reduce(|, x .≈ A)
end

function approx_in(X::AbstractArray, A)
    reduce(|, [approx_in(x, A) for x in X])
end

@testset "FEASTSolver.jl" begin

    A = diagm(0 => 1:25)
    # e, v, res = feast!(A, rand(25, 5), 4, 10, c=2.0, r=2.0)
    e, v, res = feast!(rand(ComplexF64, 25, 5), A, nodes=8, iter=10, c=1.5, r=2.0)
    @test approx_in(1:3, real.(e))
    @test maximum(sort(res)[1:3]) < 1e-12
    e, v, res = gen_feast!(rand(ComplexF64, 25, 5), A, Diagonal(ones(25)), nodes=8, iter=100, c=1.5, r=2)
    println(real.(e))
    println(res)
    # @test approx_in(1:3, real.(e))
    @test maximum(sort(res)) < 1e-12
    # e, v, res = dual_gen_feast!(rand(ComplexF64, 25, 5), A, Diagonal(ones(25)), nodes=16, iter=100, c=2.0, r=2.5)
    # println(real.(e))
    # println(res)
    # # @test approx_in(1:4, real.(e))
    # @test maximum(sort(res)) < 1e-12

end
