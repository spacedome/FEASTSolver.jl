using FEASTSolver
using LinearAlgebra
using DelimitedFiles
using Random

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

function rational_func(z, contour)
    S = 0.0+0.0im
    for i=1:size(contour.nodes,1)
        S += contour.weights[i]  / (contour.nodes[i] - z)
    end
    S
end


# A = diagm(-1 => fill(-1.0, 99), 0 => fill(2.0, 100), 1 => fill(-1.0, 99))
Random.seed!(1551)
A = randn(ComplexF64, 100, 100)
C, R = complex(0.0, 0.0), 2.0 # contour
# C, R = complex(0.1, 0.0), 0.1 # contour
X = rand(ComplexF64, 100, 20) # initial subspace

# contour = circular_contour_trapezoidal(C, R, 16)
contour = rectangular_contour_gauss(-R-im*R, R + R*im, 16)
# contour = rectangular_contour_trapezoidal(-R-im*R, R + R*im, 32)
writedlm("test/data/contour.dat", zip(real.(contour.nodes), imag.(contour.nodes)))
writedlm("test/data/weights.dat", zip(real.(contour.weights), imag.(contour.weights)))

# e, v, res = feast!(X, A, c=C, r=R, nodes=8, debug=true, ϵ=10e-15)
e, v, res = feast!(X, A, contour, debug=true, ϵ=10e-14)

info(e,v,res,C,R)
display(e)
e = eigvals(A)
println()
display(e[abs.(e) .< R])
