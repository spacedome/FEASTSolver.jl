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

# First-class dense serial FEAST variants.
export feast!, gen_feast!, dual_gen_feast!, nlfeast!
export DenseFeastStats, DenseFeastIterationStats

# Explicit contour-parallel dense FEAST API.
export distributed_feast!, distributed_gen_feast!, distributed_dual_gen_feast!
export DenseDistributedFeastPlan, DenseDistributedGeneralizedFeastPlan, DenseDistributedDualGeneralizedFeastPlan
export DenseDistributedFeastStats, DenseDistributedFeastIterationStats

# Research utilities that are stable enough to use directly.
export beyn, companion, block_SS!
export in_contour, circular_contour_trapezoidal, circular_contour_gauss, rectangular_contour_gauss, rectangular_contour_trapezoidal
export convergence_info, rational_func
export contour_estimate_eig

include("contour.jl")
include("lapack.jl")
include("fastlapack.jl")
include("stats.jl")
include("utils.jl")
include("beyn.jl")
include("companion.jl")
include("feast.jl")
include("distributed/stats.jl")
include("distributed_feast.jl")
include("distributed/workers.jl")
include("feast_experimental.jl")
include("nlfeast.jl")
include("nlfeast_experimental.jl")
include("stochastic.jl")

end # module
