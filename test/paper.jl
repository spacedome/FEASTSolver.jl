using FEASTSolver
using LinearAlgebra
using Printf
using Random
using MatrixMarket
using NonlinearEigenproblems: nep_gallery, compute_Mder
using DelimitedFiles

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

function deficient()
	# Random.seed!(1234)
	# A0 = complex.(rand(15,15))
	# A1 = complex.(rand(15,15))
	# A0[:,1] .= 0
	A0 = mmread("data/quadraticM0.mtx")
	A1 = mmread("data/quadraticM1.mtx")

	function T(z::ComplexF64)
		return (z + 0.2)*(z - 0.1).*A1 .+ A0
	end

	return T
end

function quadratic()
	A0 = mmread("data/system5A0.mtx")
	A1 = mmread("data/system5A1.mtx")
	A2 = mmread("data/system5A2.mtx")

	function T(z::ComplexF64)
		return (A2 .* (z*z)) .+ (A1 .* z) .+ A0
	end

	return T
end

function gun()
	nep_gun = nep_gallery("nlevp_native_gun")
	T(z) = compute_Mder(nep_gun, z)
	return T
end

function hadeler()
	nep_hadeler = nep_gallery("nlevp_native_hadeler", 100, 200)
	T(z) = compute_Mder(nep_hadeler, z)
	return T
end


function info(Λ, X, residuals, c, r)
    inside(x) = in_contour(x, c, r)
    in_eig = Λ[inside.(Λ)]
    in_res = residuals[inside.(Λ)]
    print("number inside : ")
    println(size(Λ[inside.(Λ)])[1])
    if sum(inside.(Λ)) > 0
        in_res_conv = in_res[in_res .<= 1e-3]
        in_eig_conv = in_eig[in_res .<= 1e-3]
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

function res_table(feast_call, dim, max_iter, max_nodes, spurious, c, r, err_tol)
	all_res = zeros((max_iter, max_nodes))
	println(repeat("-", max_nodes*8*2))
	for j=1:max_nodes
	    print(2^(j+1))
	    print("\t\t")
	end
	println("\n"*repeat("-", max_nodes*8*2))
	for i=1:max_iter
	    for j=1:max_nodes
			if i>1 && minimum(all_res[1:i-1,j]) < err_tol
				# all_res[i,j] = all_res[i-1,j]
				all_res[i,j] = -Inf
				print("\t\t")
			else
		        inside(z) = in_contour(z, c, r)
		        e, v, res = feast_call(rand(ComplexF64, dim), 2^(j+1), i-1, err_tol)
		        in_eig = e[inside.(e)]
		        in_res = res[inside.(e)]
		        in_res_conv = in_res[in_res .<= spurious]
		        if size(in_res_conv, 1) == 0
		            all_res[i,j] = maximum(res[inside.(e)])
		        else
		            all_res[i,j] = maximum(in_res_conv)
		        end
		        @printf("%2.2e\t", all_res[i,j])
			end
	    end
	    println()
	end
	println(repeat("-", max_nodes*8*2))
	return all_res
end

err_tol_all = 10e-16
max_iter_all = 10


# for i=2:8
# 	println("\nBeyn N=$(2^i)")
# 	e, v, res = nlfeast!(bf(), rand(ComplexF64,64,30), 2^i, 0, c=complex(1.0,1.0), r=0.5)
# 	info(e, v, res, complex(1.0,1.0), 0.5)
# end


Random.seed!(1729)

# println("\nButterfly Problem")
# max_iter_butterfly = 25
# bf_feast_call(X, N, iter, err_tol) = nlfeast!(bf(), X, N, iter, c=complex(1.0,1.0), r=0.5, ϵ=err_tol)
# bf_res = res_table(bf_feast_call, (64,20), max_iter_butterfly, 7, 1e-0, complex(1.0,1.0), 0.5, err_tol_all)
# writedlm("test/data/bf.dat", zip(1:max_iter_butterfly, eachcol(bf_res)...))
#
#
# println("\nDeficient Quadratic Problem")
#
# deficient_feast_call(X, N, iter, err_tol) = nlfeast_moments!(deficient(), X, N, iter, c=complex(0.0,0.0), r=0.25, ϵ=err_tol, moments=2, store=true)
# def_res = res_table(deficient_feast_call, (15,4), max_iter_all, 7, 1e-5, complex(0.0,0.0), 0.25, err_tol_all)
# writedlm("test/data/deficient.dat", zip(1:max_iter_all, eachcol(def_res)...))
#
#
# println("\nLarger Quadratic Problem")
# quad_feast_call(X, N, iter, err_tol) = nlfeast_moments!(quadratic(), X, N, iter, c=complex(-1.55,0.0), r=0.05, ϵ=err_tol, moments=2, store=true)
# quad_res = res_table(quad_feast_call, (1000,70), max_iter_all, 6, 1e-6, complex(-1.55,0.0), 0.05, err_tol_all)
# writedlm("test/data/quad.dat", zip(1:max_iter_all, eachcol(quad_res)...))
#
#
# println("\nGun Cavity Problem")
# gun_feast_call(X, N, iter, err_tol) = nlfeast!(gun(), X, N, iter, c=complex(140000.0, 0.0), r=30000, ϵ=err_tol, store=false)
# gun_res = res_table(gun_feast_call, (9956,32), max_iter_all, 6, 1e-6, complex(140000.0, 0.0), 30000, err_tol_all)
# writedlm("test/data/gun.dat", zip(1:max_iter_all, eachcol(gun_res)...))


println("\nHadeler Problem")
hadeler_feast_call(X, N, iter, err_tol) = nlfeast!(hadeler(), X, N, iter, c=complex(-30.0, 0.0), r=10.0, ϵ=err_tol, store=true)
hadeler_res = res_table(hadeler_feast_call, (200,15), max_iter_all, 6, 1e-4, complex(-30.0, 0.0), 10.0, err_tol_all)
writedlm("test/data/hadeler.dat", zip(1:max_iter_all, eachcol(hadeler_res)...))
