__precompile__(true)

module FEASTSolver

using LinearAlgebra: ldiv!, lu!, mul!, qr!, rmul!, lmul!, eigen!, svd!, norm, Diagonal, I, diagm
using IterativeSolvers: bicgstabl
using SparseArrays: similar

export feast!, ifeast!, nlfeast!, nlfeast_opt!, nlfeast_it!, nlfeast_moments!
export gen_feast!, dual_gen_feast!
export beyn
export in_contour

function in_contour(λ, c, r)
    abs(λ - c) <= r
end

include("lapack.jl")
include("feast.jl")
include("nlfeast.jl")
include("beyn.jl")

end # module
