__precompile__(true)

module FEASTSolver

import LinearAlgebra

using LinearAlgebra: BLAS, LAPACK, UniformScaling, ldiv!, lu!, tr, dot, LU, Factorization, axpy!, mul!, qr!, rmul!, lmul!, ldiv!, eigen!, svd!, norm, Diagonal, I, diagm
using IterativeSolvers: bicgstabl
using SparseArrays: similar, AbstractSparseMatrix, sprandn, sprand, sparse, spdiagm, SparseVector, SparseMatrixCSC, nnz
using IterativeSolvers: gmres!, bicgstabl!, gmres, bicgstabl
using FastGaussQuadrature: gausslegendre
using FastLapackInterface: EigenWs, GeneralizedEigenWs, LUWs, QRWs, SVDsddWs
using Distributed: @distributed, myid, remotecall, remotecall_wait, workers
using Random: rand, randn

import Base: close, length
import LinearAlgebra: lu, mul!, qr

# First-class dense serial FEAST variants.
export feast!, gen_feast!, dual_gen_feast!, nlfeast!
export reference_feast!, reference_gen_feast!, reference_dual_gen_feast!, reference_nlfeast!
export AbstractSparseFeastSolver, SparseDirectSolver, SparseBiCGSTABSolver
export DenseFeastStats, DenseFeastIterationStats

# Explicit contour-parallel dense FEAST API.
export distributed_feast!, distributed_gen_feast!, distributed_dual_gen_feast!, distributed_nlfeast!
export DenseDistributedFeastPlan, DenseDistributedGeneralizedFeastPlan, DenseDistributedDualGeneralizedFeastPlan
export DenseDistributedNonlinearFeastPlan
export DenseDistributedFeastStats, DenseDistributedFeastIterationStats

# Research utilities that are stable enough to use directly.
export beyn, companion, block_SS!
export Contour, CircularContour, RectangularContour, CustomContour
export contour_nodes, contour_weights
export in_contour, circular_contour_trapezoidal, circular_contour_gauss, rectangular_contour_gauss, rectangular_contour_trapezoidal
export convergence_info, rational_func
export contour_estimate_eig
export AbstractFeastOperator, feast_gallery
export operator_matrix, operator_prototype, operator_action_workspace
export materialize!, matrix_materializer, matrix_operator

include("contour.jl")
include("lapack.jl")
include("fastlapack.jl")
include("stats.jl")
include("utils.jl")
include("linalg/reduced.jl")
include("linalg/residuals.jl")
include("operators/interface.jl")
include("operators/gallery.jl")
include("linalg/sparse_feast_common.jl")
include("beyn.jl")
include("companion.jl")
include("reference/linear_standard.jl")
include("reference/linear_generalized.jl")
include("reference/linear_dual_generalized.jl")
include("reference/nonlinear.jl")
include("optimized/linear_standard.jl")
include("optimized/linear_generalized.jl")
include("optimized/linear_dual_generalized.jl")
include("optimized/sparse_standard.jl")
include("optimized/sparse_generalized.jl")
include("optimized/nonlinear.jl")
include("moment_rii.jl")
include("distributed/stats.jl")
include("distributed/common.jl")
include("distributed/plans.jl")
include("distributed/nonlinear_plans.jl")
include("distributed/workers.jl")
include("distributed/linear_standard_workers.jl")
include("distributed/linear_generalized_workers.jl")
include("distributed/linear_dual_generalized_workers.jl")
include("distributed/nonlinear_workers.jl")
include("distributed/linear_standard.jl")
include("distributed/linear_generalized.jl")
include("distributed/linear_dual_generalized.jl")
include("distributed/nonlinear.jl")
include("feast_experimental.jl")
include("experimental/nonlinear_legacy.jl")
include("experimental/nonlinear_moments_legacy.jl")
include("stochastic.jl")

end # module
