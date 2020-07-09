using FEASTSolver
using LinearAlgebra
using Profile
using NonlinearEigenproblems: PEP, compute_Mder, contour_block_SS, polyeig, compute_Mlincomb, compute_MM

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


function bf_arrays()
    N = diagm(-1 => [1,1,1,1,1,1,1])
    Mh0 = 1/6*(4*I + N + N')
    Mh1 = N - N'
    Mh2 = -1*(2*I - N - N')
    Mh3 = Mh1
    Mh4 = -Mh2
    c = [0.6 1.3; 1.3 0.1; 0.1 1.2; 1.0 1.0; 1.2 1.0]
    I8 = Matrix(I, 8, 8)
    M0 = c[1,1] * kron(I8, Mh0) + c[1,2] * kron(Mh0, I8)
    M1 = c[2,1] * kron(I8, Mh1) + c[2,2] * kron(Mh1, I8)
    M2 = c[3,1] * kron(I8, Mh2) + c[3,2] * kron(Mh2, I8)
    M3 = c[4,1] * kron(I8, Mh3) + c[4,2] * kron(Mh3, I8)
    M4 = c[5,1] * kron(I8, Mh4) + c[5,2] * kron(Mh4, I8)
    return [M0, M1, M2, M3, M4]
end


function bf()
    N = diagm(-1 => [1,1,1,1,1,1,1])
    Mh0 = 1/6*(4*I + N + N')
    Mh1 = N - N'
    Mh2 = -1*(2*I - N - N')
    Mh3 = Mh1
    Mh4 = -Mh2
    c = [0.6 1.3; 1.3 0.1; 0.1 1.2; 1.0 1.0; 1.2 1.0]
    I8 = Matrix(I, 8, 8)
    M0 = c[1,1] * kron(I8, Mh0) + c[1,2] * kron(Mh0, I8)
    M1 = c[2,1] * kron(I8, Mh1) + c[2,2] * kron(Mh1, I8)
    M2 = c[3,1] * kron(I8, Mh2) + c[3,2] * kron(Mh2, I8)
    M3 = c[4,1] * kron(I8, Mh3) + c[4,2] * kron(Mh3, I8)
    M4 = c[5,1] * kron(I8, Mh4) + c[5,2] * kron(Mh4, I8)
    function butterfly(A0, A1, A2, A3, A4)
        f(z) =  z^4*A4 + z^3*A3 + z^2*A2 + z*A1 + A0
    end

    return butterfly(M0, M1, M2, M3, M4)
end

C = complex(1.0,1.0)
R = 0.5

T = bf()
e, v, res = nlfeast!(T, rand(ComplexF64,64,30), 2^2, 0, c=complex(1.0,1.0), r=0.5)
e, v, res = @timev nlfeast!(T, rand(ComplexF64,64,20), 2^9, 0, c=complex(1.0,1.0), r=0.5, debug=false, ϵ=10e-16, spurious=5e-3)
# e, v, res = nlfeast_moments!(T, rand(ComplexF64,64,15), 2^4, 40, moments=2, c=complex(1.0,1.0), r=0.5, debug=true, ϵ=10e-16, spurious=5e-3)
# # e, v, res = companion(bf_arrays())
# # e, v, res = @profile nlfeast!(T, rand(ComplexF64,64,30), 2^4, 20, c=complex(1.0,1.0), r=0.5, debug=true, ϵ=10e-16)
# # Profile.print()
# # e, v, res = nlfeast_it!(T, rand(ComplexF64,64,30), 2^3, 0, c=complex(1.0,1.0), r=0.5)
# # e, v, res = @timev nlfeast_it!(T, rand(ComplexF64,64,30), 2^3, 20, c=complex(1.0,1.0), r=0.5, debug=true, ϵ=10e-14)
# # display(e)
# inside(x) = in_contour(x, complex(1.0, 1.0), 0.5)
# in_eig = e[inside.(e)]
# in_res = res[inside.(e)]
# in_res_conv = in_res[in_res .<= 5e-3]
# in_eig_conv = in_eig[in_res .<= 5e-3]
# print("\nmax res inside: ")
# println(maximum(res[inside.(e)]))
# # print("max res inside non spurious: ")
# # println(maximum(in_res_conv))
# print("number inside : ")
# println(size(e[inside.(e)])[1])
# # print("number inside converged : ")
# # println(size(in_eig_conv)[1])
# display(in_eig)
# println()
# display(in_res)

# inside(x) = in_contour(x, complex(1.0, 1.0), 0.5)
#
# nep = PEP(bf_arrays())
# # e,v = polyeig(nep)
# e, v = contour_block_SS(nep, radius=R, σ=C, N=2^8, k=30,K=8)
# res = zeros(size(e,1))
# for i=1:size(e,1)
#     res[i] = norm(compute_Mlincomb(nep, e[i], v[:,i]))
# end
#
# info(e, v, res, C, R)
# in_eig = e[inside.(e)]
# display(in_eig)
#
# println()
#
# e, v, res = block_SS!(T, rand(ComplexF64,64,30), 2^6, 4, c=C, r=R)
# # e, v, res = nlfeast_moments_SS!(T, rand(ComplexF64,64,16), 2^6, 1, c=C, r=R, moments=2, debug=true, ϵ=10e-16, spurious=1e-3)
# info(e, v, res, C, R)
# inside(x) = in_contour(x, complex(1.0, 1.0), 0.5)
# in_eig = e[inside.(e)]
# in_res = res[inside.(e)]
# display(in_eig)
# display(in_res)
