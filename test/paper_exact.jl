using FEASTSolver
using LinearAlgebra
using Printf
using Random
using MatrixMarket
using NonlinearEigenproblems: nep_gallery, compute_Mder, polyeig, PEP
using DelimitedFiles

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


function deficient_arrays()
	# Random.seed!(1234)
	# A0 = complex.(rand(15,15))
	# A1 = complex.(rand(15,15))
	# A0[:,1] .= 0
	A0 = mmread("data/quadraticM0.mtx")
	A1 = mmread("data/quadraticM1.mtx")

	return [(A0 .- 0.02.*A1), (0.1.*A1), A1]
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

# for i=2:8
# 	println("\nBeyn N=$(2^i)")
# 	e, v, res = nlfeast!(bf(), rand(ComplexF64,64,30), 2^i, 0, c=complex(1.0,1.0), r=0.5)
# 	info(e, v, res, complex(1.0,1.0), 0.5)
# end


# Random.seed!(1729)



println("\nButterfly Problem")
nep = PEP(bf_arrays())
e,v = polyeig(nep)
e_in = e[abs.(e .- (1.0+1.0im)) .<= 0.5]
bf_eig = hcat(real.(e), imag.(e))
bf_eig_in = hcat(real.(e_in), imag.(e_in))
writedlm("test/data/bf_eig.dat", bf_eig)
writedlm("test/data/bf_eig_in.dat", bf_eig_in)


println("\nDeficient Quadratic Problem")
nep = PEP(deficient_arrays())
e,v = polyeig(nep)
e_in = e[abs.(e) .<= 0.25]
def_eig = hcat(real.(e), imag.(e))
def_eig_in = hcat(real.(e_in), imag.(e_in))
writedlm("test/data/deficient_eig.dat", def_eig)
writedlm("test/data/deficient_eig_in.dat", def_eig_in)


# println("\nLarger Quadratic Problem")
# quad_feast_call(X, N, iter, err_tol) = nlfeast_moments!(quadratic(), X, N, iter, c=complex(-1.55,0.0), r=0.05, ϵ=err_tol, moments=2, store=true)
# quad_res = res_table(quad_feast_call, (1000,70), max_iter_all, 6, 1e-6, complex(-1.55,0.0), 0.05, err_tol_all)
# writedlm("test/data/quad.dat", zip(1:max_iter_all, eachcol(quad_res)...))


println("\nGun Cavity Problem")
# gun_feast_call(X, N, iter, err_tol) = nlfeast!(gun(), X, N, iter, c=complex(140000.0, 0.0), r=30000, ϵ=err_tol, store=false)
# gun_res = res_table(gun_feast_call, (9956,32), max_iter_all, 6, 1e-6, complex(140000.0, 0.0), 30000, err_tol_all)
e, v, res = nlfeast!(gun(), rand(ComplexF64,9956,84), 2^4, 10, c=complex(140000.0, 0.0), r=2*30000, debug=true, ϵ=10e-16, store=false)
e_conv = e[res .<= 1e-10]
e_in = e_conv[abs.(e_conv .- 140000) .<= 30000]
gun_eig = hcat(real.(e_conv), imag.(e_conv))
gun_eig_in = hcat(real.(e_in), imag.(e_in))
writedlm("test/data/gun_eig.dat", gun_eig)
writedlm("test/data/gun_eig_in.dat", gun_eig_in)


println("\nHadeler Problem")
# gun_feast_call(X, N, iter, err_tol) = nlfeast!(gun(), X, N, iter, c=complex(140000.0, 0.0), r=30000, ϵ=err_tol, store=false)
# gun_res = res_table(gun_feast_call, (9956,32), max_iter_all, 6, 1e-6, complex(140000.0, 0.0), 30000, err_tol_all)
e, v, res = nlfeast!(hadeler(), rand(ComplexF64,200,100), 2^5, 10, c=complex(-30.0, 0.0), r=20, debug=true, ϵ=10e-16, store=true)
e_conv = e[res .<= 1e-10]
e_in = e_conv[abs.(e_conv .+ 30) .<= 10]
hadeler_eig = hcat(real.(e_conv), imag.(e_conv))
hadeler_eig_in = hcat(real.(e_in), imag.(e_in))
writedlm("test/data/hadeler_eig.dat", hadeler_eig)
writedlm("test/data/hadeler_eig_in.dat", hadeler_eig_in)
