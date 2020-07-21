using FEASTSolver
using Test
using LinearAlgebra
using SparseArrays

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
    # @test approx_in(1:3, real.(e))
    @test maximum(sort(res)) < 1e-12
    # e, v, res = dual_gen_feast!(rand(ComplexF64, 25, 5), A, Diagonal(ones(25)), nodes=16, iter=100, c=2.0, r=2.5)
    # # @test approx_in(1:4, real.(e))
    # @test maximum(sort(res)) < 1e-12

end


@testset "contours" begin

    A = spdiagm(-1 => fill(-1.0, 99), 0 => fill(2.0, 100), 1 => fill(-1.0, 99))
    C, R = complex(0.05, 0.0), 0.05 # contour

    contour = circular_contour_trapezoidal(C, R, 8)
    e, v, res = feast!(rand(ComplexF64, 100, 20), A, contour, ϵ=10e-15)
    @test size(e,1) == 10
    @test maximum(res) < 10e-15

    contour = rectangular_contour_trapezoidal(0.0 - R*im, 2*R + R*im, 8)
    e, v, res = feast!(rand(ComplexF64, 100, 20), A, contour, ϵ=10e-15)
    @test size(e,1) == 10
    @test maximum(res) < 10e-15

    contour = rectangular_contour_gauss(0.0 - R*im, 2*R + R*im, 8)
    e, v, res = feast!(rand(ComplexF64, 100, 20), A, contour, ϵ=10e-15)
    @test size(e,1) == 10
    @test maximum(res) < 10e-15

end
