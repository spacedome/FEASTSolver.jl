using FEASTSolver
using LinearAlgebra
using MatrixMarket

A0 = Matrix(mmread("data/system5A0.mtx"))
A1 = Matrix(mmread("data/system5A1.mtx"))
A2 = Matrix(mmread("data/system5A2.mtx"))

function T(z::ComplexF64)
	return (A2 .* (z*z)) .+ (A1 .* z) .+ A0
end

e, v, res = @timev nlfeast!(T, rand(ComplexF64,1000,30), 2^3, 30, c=complex(-1.55,0.0), r=0.05)
display(e)
display(res)
