__precompile__(true)

module FEASTSolver

using LinearAlgebra: BLAS, LAPACK, UniformScaling, ldiv!, lu!, tr, dot, LU, Factorization, mul!, qr!, rmul!, lmul!, ldiv!, eigen!, svd!, norm, Diagonal, I, diagm
using IterativeSolvers: bicgstabl
using SparseArrays: similar, AbstractSparseMatrix, sprandn, sprand
using IterativeSolvers: gmres!, bicgstabl!, gmres, bicgstabl
using FastGaussQuadrature: gausslegendre
using FastLapackInterface: EigenWs, GeneralizedEigenWs, LUWs, QRWs, SVDsddWs
using Distributed: @distributed, myid, remotecall, remotecall_wait, workers
using Random: rand, randn
using SharedArrays: SharedArray

import Base: close, length

export feast!, ifeast!, nlfeast!, nlfeast_opt!, nlfeast_it!, nlfeast_moments!, nlfeast_moments_SS!
export gen_feast!, dual_gen_feast!, distributed_feast!
export DenseDistributedFeastPlan, DenseDistributedFeastStats, DenseDistributedFeastIterationStats
export beyn, companion, block_SS!, nlfeast_moments_all!
export in_contour, circular_contour_trapezoidal, circular_contour_gauss, rectangular_contour_gauss, rectangular_contour_trapezoidal
export convergence_info, rational_func
export contour_estimate_eig

include("contour.jl")
include("lapack.jl")
include("fastlapack.jl")
include("utils.jl")
include("beyn.jl")
include("companion.jl")
include("feast.jl")
include("distributed_feast.jl")
include("feast_experimental.jl")
include("nlfeast.jl")
include("nlfeast_experimental.jl")
include("stochastic.jl")

end # module
