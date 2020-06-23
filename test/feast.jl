using FEASTSolver
using LinearAlgebra

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


A = diagm(-1 => fill(-1.0, 99), 0 => fill(2.0, 100), 1 => fill(-1.0, 99))
C, R = complex(0.05, 0.0), 0.05 # contour
X = rand(ComplexF64, 100, 20) # initial subspace

e, v, res = feast!(X, A, c=C, r=R, nodes=8, debug=true, ϵ=10e-15)

info(e,v,res,C,R)
display(e)
