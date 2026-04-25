using FEASTSolver
using LinearAlgebra
using NonlinearEigenproblems: nep_gallery, compute_Mder

function info(Λ, X, residuals, c, r)
    inside(x) = in_contour(x, c, r)
    in_eig = Λ[inside.(Λ)]
    in_res = residuals[inside.(Λ)]
    print("number inside : ")
    println(size(Λ[inside.(Λ)])[1])
    if sum(inside.(Λ)) > 0
        in_res_conv = in_res[in_res .<= 1e-5]
        in_eig_conv = in_eig[in_res .<= 1e-5]
        print("number inside converged : ")
        println(size(in_eig_conv)[1])
        print("max res inside: ")
        println(maximum(residuals[inside.(Λ)]))
        if size(in_res_conv, 1) > 0
            print("max res inside non spurious: ")
            println(maximum(in_res_conv))
        end
    end
    # display(in_eig_conv)
end


nep = nep_gallery("nlevp_native_hadeler", 100, 200)
T(x) = compute_Mder(nep, x)

n = size(nep, 1)

#
# function hadeler(n=200, b₀=100)
#     M0 = zeros(ComplexF64, n, n)
#     M1 = zeros(ComplexF64, n, n)
#     M2 = zeros(ComplexF64, n, n)
#
#     M0 .= b₀ * Matrix(I, n, n)
#     for i=1:n
#         for j=1:n
#             M1[i,j] = (n+1-max(i,j))*i*j
#             M2[i,j] = (n*(i == j)) + 1/(i+j)
#         end
#     end
#
#     function prob(B0, B1, B2)
#         f(z) =  (exp(z)-1).*B1 .+ z^2 .* B2 .- B0
#     end
#
#     return prob(M0, M1, M2)
# end
#
# C = complex(-3.5, 0.0)
# R = 0.05
C = complex(-30.0, 0.0)
R = 10.0
# T = hadeler()
# e, v, res = nlfeast!(T, rand(ComplexF64,n,10), 2^3, 0, c=C, r=R)
e, v, res = nlfeast!(T, (rand(ComplexF64,n,15)), 2^3, 30, c=C, r=R, debug=true, ϵ=10e-16, spurious=1e-2)
# e, v, res = nlfeast_moments!(T, (rand(ComplexF64,1000,20)), 2^6, 10,store=false, moments=2, c=C, r=R, debug=true, ϵ=10e-16, spurious=1e-6)
# e, v, res = @profile nlfeast!(T, rand(ComplexF64,64,30), 2^4, 20, c=complex(1.0,1.0), r=0.5, debug=true, ϵ=10e-16)
# Profile.print()
# e, v, res = nlfeast_it!(T, rand(ComplexF64,64,30), 2^3, 0, c=complex(1.0,1.0), r=0.5)
# e, v, res = @timev nlfeast_it!(T, rand(ComplexF64,64,30), 2^3, 20, c=complex(1.0,1.0), r=0.5, debug=true, ϵ=10e-14)
# display(e)
info(e,v, res, C, R)

inside(x) = in_contour(x, C, R)
in_eig = e[inside.(e)]
in_res = res[inside.(e)]
# in_res_conv = in_res[in_res .<= 1e-3]
# in_eig_conv = in_eig[in_res .<= 1e-3]
# print("\nmax res inside: ")
# println(maximum(res[inside.(e)]))
# print("max res inside non spurious: ")
# println(maximum(in_res_conv))
# print("number inside : ")
# println(size(e[inside.(e)])[1])
# print("number inside converged : ")
# println(size(in_eig_conv)[1])
display(in_eig)
display(in_res)
