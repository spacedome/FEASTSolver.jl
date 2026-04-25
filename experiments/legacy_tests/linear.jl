using FEASTSolver
using LinearAlgebra
using Profile
using Random

function bf()
    A = diagm(-1 => fill(-1.0, 99), 0 => fill(2, 100), 1 => fill(-1.0, 99))
    # Random.seed!(1234)
    # A = randn(100,100)
    # A .= A + A'
    function linprob()
	f(z) = I*z - A
    end

    return linprob()
end


C, R, = complex(0.0, 0.0), 10.0/1033.660731699839 
T = bf()
Y = ComplexF64.(rand(Float64,100,10))
e, v, res = nlfeast!(T, rand(ComplexF64,100,10), 2^3, 0, c=C, r=R)
e, v, res = @timev nlfeast!(T, Y, 2^2, 10, c=C, r=R, debug=true, ϵ=10e-8)
# e, v, res = @profile nlfeast!(T, rand(ComplexF64,64,30), 2^4, 20, c=complex(1.0,1.0), r=0.5, debug=true, ϵ=10e-16)
# Profile.print()
# e, v, res = nlfeast_it!(T, rand(ComplexF64,64,30), 2^3, 0, c=complex(1.0,1.0), r=0.5)
# e, v, res = @timev nlfeast_it!(T, rand(ComplexF64,64,30), 2^3, 20, c=complex(1.0,1.0), r=0.5, debug=true, ϵ=10e-14)
# display(e)
inside(x) = in_contour(x, C, R)
in_eig = e[inside.(e)]
in_res = res[inside.(e)]
in_res_conv = in_res[in_res .<= 1e-6]
in_eig_conv = in_eig[in_res .<= 1e-6]
print("\nmax res inside: ")
println(maximum(res[inside.(e)]))
print("max res inside non spurious: ")
println(maximum(in_res_conv))
print("number inside : ")
println(size(e[inside.(e)])[1])
print("number inside converged : ")
println(size(in_eig_conv)[1])
display(in_eig_conv)
