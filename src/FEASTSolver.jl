__precompile__(true)

module FEASTSolver

using LinearAlgebra: ldiv!, lu!, LU,Factorization, mul!, qr!, rmul!, lmul!, ldiv!, eigen!, svd!, norm, Diagonal, I, diagm
using IterativeSolvers: bicgstabl
using SparseArrays: similar
using IterativeSolvers: gmres!, bicgstabl!, gmres, bicgstabl
using FastGaussQuadrature: gausslegendre
using Distributed: @distributed
using SharedArrays: SharedArray

import Base: length

export feast!, ifeast!, nlfeast!, nlfeast_opt!, nlfeast_it!, nlfeast_moments!, nlfeast_moments_SS!
export gen_feast!, dual_gen_feast!
export beyn, companion, block_SS!, nlfeast_moments_all!
export in_contour, circular_contour_trapezoidal, rectangular_contour_gauss, rectangular_contour_trapezoidal
export convergence_info, rational_func

include("contour.jl")
include("lapack.jl")
include("utils.jl")
include("beyn.jl")
include("companion.jl")
include("feast.jl")
include("feast_experimental.jl")
include("nlfeast.jl")
include("nlfeast_experimental.jl")

end # module
