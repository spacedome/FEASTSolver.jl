using NonlinearEigenproblems: nep_gallery, compute_Mder
using FEASTSolver

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
C = complex(140000.0, 0.0)
R = 30000

e, v, res = nlfeast!(T, rand(ComplexF64,n,20), 2^7, 0, c=C, r=R, debug=true, ϵ=10e-16, store=false)

info(e,v,res,C,R)