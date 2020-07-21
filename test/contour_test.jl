using FEASTSolver
using LinearAlgebra
using DelimitedFiles
using SparseArrays


N = 1000
A = spdiagm(-1 => fill(-1.0, N-1), 0 => fill(2.0, N), 1 => fill(-1.0, N-1))
# A = diagm(-1 => fill(-1.0, N-1), 0 => fill(2.0, N), 1 => fill(-1.0, N-1))
# A = I + sprand(100, 100, 0.1)
R = 500.0/(N^2)
C = complex(R, 0.0) # contour
# C, R = complex(0.1, 0.0), 0.1 # contour
# X = rand(ComplexF64, N, 20) # initial subspace
X = sprand(ComplexF64, N, 20, 0.05)

contour = circular_contour_trapezoidal(C, R, 16)
# contour = circular_contour_gauss(C, R, 16)
# contour = rectangular_contour_gauss(0.0 - R*im, 2*R + R*im, 32)
# contour = rectangular_contour_trapezoidal(0.0 - R*im, 2*R + R*im, 8)
# contour = rectangular_contour_gauss(0.9 - R*im, 1.1 + R*im, 128)
writedlm("test/data/contour.dat", zip(real.(contour.nodes), imag.(contour.nodes)))
writedlm("test/data/weights.dat", zip(real.(contour.weights), imag.(contour.weights)))

# z = 0.025+0.0im
# println(rational_func(z, contour))

# e, v, res = feast!(deepcopy(X), A, contour, debug=true, ϵ=10e-15, iter=0, store=false)
# @timev e, v, res = feast!(deepcopy(X), A, contour, debug=true, ϵ=10e-15, store=false, mixed_prec=false)
# @timev e, v, res = feast!(X, A, contour, debug=true, ϵ=10e-15, store=true)

println(contour_estimate_eig(A, contour, debug=true, samples=100))

# convergence_info(e,v,res,contour)
# display(e)
