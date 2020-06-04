using FEASTSolver
using LinearAlgebra
using MatrixMarket

# A0 = Matrix(mmread("data/system5A0.mtx"))
# A1 = Matrix(mmread("data/system5A1.mtx"))
# A2 = Matrix(mmread("data/system5A2.mtx"))
A0 = complex.(rand(15,15))
A1 = complex.(rand(15,15))
A0[:,1] .= 0

function T(z::ComplexF64)
	return (z + 0.2)*(z - 0.1).*A1 .+ A0
end

R = 0.25
# C = complex(-0.8, 0.8)
C = complex(0, 0.0)

# e, v, res = @timev nlfeast!(T, rand(ComplexF64,1000,30), 2^3, 30, c=complex(-1.55,0.0), r=0.05)
e, v, res = nlfeast!(T, rand(ComplexF64,15,5), 2^4, 0, c=C, r=R)
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
in_res_conv = in_res[in_res .<= 1e-2] 
in_eig_conv = in_eig[in_res .<= 1e-2] 
if size(in_eig, 1) > 0
    print("\nmax res inside: ")
    println(maximum(res[inside.(e)]))
end
# print("max res inside non spurious: ")
# println(maximum(in_res_conv))
print("\nnumber inside : ")
println(size(e[inside.(e)])[1])
# print("number inside converged : ")
# println(size(in_eig_conv)[1])
display(res)

