using NonlinearEigenproblems: nep_gallery, compute_Mder, contour_block_SS, contour_beyn
using FEASTSolver

using SuiteSparse

SuiteSparse.UMFPACK.umf_ctrl[8] = 0;

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


nep = nep_gallery("nlevp_native_gun")
T(x) = compute_Mder(nep, x)

n = size(nep, 1)
C = complex(62500.0, 0.0)
R = 50000

e, v, res = nlfeast!(T, rand(ComplexF64,n,32), 2^3, 10, c=C, r=R, debug=true, ϵ=10e-10, store=true)
# e, v, res = nlfeast!(T, rand(ComplexF64,n,50), 2^4, 10, c=C, r=2*R, debug=true, ϵ=10e-16, store=false)
#
# e, v, res = nlfeast_moments!(T, rand(ComplexF64,n,4), 2^5, 0, moments=8, c=C, r=R, debug=true, ϵ=10e-16, store=false)
info(e,v,res,C,R)

# e, v = contour_beyn(nep, radius=R, σ=C, N=2^8, k=80)
# e, v = contour_block_SS(nep, radius=2*R, σ=C, N=2^8, k=80, K=4)
# print(e)
