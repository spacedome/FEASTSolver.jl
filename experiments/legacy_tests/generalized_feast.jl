using FEASTSolver
using LinearAlgebra
using DelimitedFiles
using SparseArrays


A = spdiagm(-1 => fill(-1.0, 99), 0 => fill(2.0, 100), 1 => fill(-1.0, 99))
# A = diagm(-1 => fill(-1.0, 99), 0 => fill(2.0, 100), 1 => fill(-1.0, 99))
B = sprand(100, 100, 0.2)
B = B + B'
C, R = complex(0.05, 0.0), 0.05 # contour
# C, R = complex(0.1, 0.0), 0.1 # contour
# X = rand(ComplexF64, 100, 20) # initial subspace
X = sprand(ComplexF64, 100, 20, 0.9)

contour = circular_contour_trapezoidal(C, R, 8)
# contour = rectangular_contour_gauss(0.0 - R*im, 2*R + R*im, 8)
# contour = rectangular_contour_trapezoidal(0.0 - R*im, 2*R + R*im, 32)
# contour = rectangular_contour_gauss(0.9 - R*im, 1.1 + R*im, 16)

# z = 0.025+0.0im
# println(rational_func(z, contour))

e, v, res = gen_feast!(X, A, B, contour, debug=true, ϵ=10e-15)
# @timev e, v, res = feast!(X, A, contour, debug=false, ϵ=10e-15, iter=1)

convergence_info(e,v,res,contour)
display(e)
