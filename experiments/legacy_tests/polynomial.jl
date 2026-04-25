using FEASTSolver
using LinearAlgebra
using MatrixMarket

A0 = mmread("data/system5A0.mtx")
A1 = mmread("data/system5A1.mtx")
A2 = mmread("data/system5A2.mtx")

function T(z::ComplexF64)
	return (A2 .* (z*z)) .+ (A1 .* z) .+ A0
end

R = 0.05
# C = complex(-0.8, 0.8)
C = complex(-1.55, 0.0)

# e, v, res = nlfeast!(T, rand(ComplexF64,1000,3), 2^2, 0, c=complex(-1.55,0.0), r=0.05)
# e, v, res = @timev nlfeast!(T, rand(ComplexF64,1000,30), 2^3, 10, c=complex(-1.55,0.0), r=0.05)
e, v, res = nlfeast_moments!(T, rand(ComplexF64,1000,10), 2^2, 0, c=C, r=R, debug=false, moments=3)
e, v, res = @timev nlfeast_moments!(T, rand(ComplexF64,1000,80), 2^5, 10, c=C, r=R, debug=true, moments=2, ϵ=10e-16, store=true)
# e, v, res = beyn(T, A0, rand(ComplexF64,1000,120), 2^9; c=complex(-1.55,0.0), r=0.05)
# display(e)
# display(res)
inside(x) = in_contour(x, C, R)
# print("\nmax res inside: ")
# println(maximum(res[inside.(e)]))
# print("number inside : ")
# println(size(e[inside.(e)])[1])
# # display(e)
# display(res[inside.(e)])

in_eig = e[inside.(e)]
in_res = res[inside.(e)]
in_res_conv = in_res[in_res .<= 1e-3]
in_eig_conv = in_eig[in_res .<= 1e-3]
print("\nmax res inside: ")
println(maximum(res[inside.(e)]))
print("max res inside non spurious: ")
println(maximum(in_res_conv))
print("number inside : ")
println(size(e[inside.(e)])[1])
print("number inside converged : ")
println(size(in_eig_conv)[1])
# display(in_eig_conv)
