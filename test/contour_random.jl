using FEASTSolver
using LinearAlgebra
using DelimitedFiles
using Random


# A = diagm(-1 => fill(-1.0, 99), 0 => fill(2.0, 100), 1 => fill(-1.0, 99))
Random.seed!(1551)
A = randn(ComplexF64, 100, 100)
C, R = complex(0.0, 0.0), 2.0 # contour
# C, R = complex(0.1, 0.0), 0.1 # contour
X = rand(ComplexF64, 100, 20) # initial subspace

# contour = circular_contour_trapezoidal(C, R, 16)
# contour = rectangular_contour_gauss(-R-im*R, R + R*im, 16)
contour = rectangular_contour_trapezoidal(-R-im*R, R + R*im, 32)
writedlm("test/data/contour.dat", zip(real.(contour.nodes), imag.(contour.nodes)))
writedlm("test/data/weights.dat", zip(real.(contour.weights), imag.(contour.weights)))

# e, v, res = feast!(X, A, c=C, r=R, nodes=8, debug=true, ϵ=10e-15)
e, v, res = feast!(X, A, contour, debug=true, ϵ=10e-14)

# info(e,v,res,C,R)
convergence_info(e,v,res,contour)
display(e)
e = eigvals(A)
println()
display(e[abs.(e) .< R])
