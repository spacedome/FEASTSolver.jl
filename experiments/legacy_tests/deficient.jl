using FEASTSolver
using LinearAlgebra
using MatrixMarket
using Random
import FEASTSolver: iter_debug_print, update_R!, residuals, normalize!

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


function moments_expand!(T, X::AbstractMatrix{ComplexF64}, nodes::Integer, iter::Integer;
    c=complex(0.0,0.0), r=1.0, debug=false, ϵ=10e-12, moments=2, spurious=1e-5)

    N, m₀ = size(X)
    Λ = zeros(ComplexF64, moments*m₀)
    θ = LinRange(π/nodes, 2*π-π/nodes, nodes)
    Q = zeros(ComplexF64, 2*moments, N, m₀)
    Q₀, Q₁ = zeros(ComplexF64, moments*N, moments*m₀), zeros(ComplexF64, moments*N, moments*m₀)
	Y = zeros(ComplexF64, N, m₀*moments)
    R = similar(Y, ComplexF64)


    for i=1:nodes
        z = (r*exp(θ[i]*im)+c)
		Temp = (T(z)\X) .* (r*exp(θ[i]*im)/nodes)
        Q[1,:,:] .+= Temp
        for j=2:2*moments
            Q[j,:,:] .+= Temp .* z^(j-1)
        end
    end
    for i=1:moments, j=1:moments
        Q₀[(i-1)*N+1:i*N, (j-1)*m₀+1:j*m₀] .= Q[i+j-1,:,:]
        Q₁[(i-1)*N+1:i*N, (j-1)*m₀+1:j*m₀] .= Q[i+j,:,:]
    end

    S = svd!(Q₀)
    F = eigen!(S.U' * Q₁ * S.V * Diagonal(1 ./ S.S))
    Y = S.U[1:N,:] * F.vectors
    Λ = F.values

    update_R!(Y, R, Λ, T)
    res = residuals(R, Λ, T)

	iter_debug_print(0, Λ, res, c, r, spurious)

	m₁ = moments*m₀
    Q₀, Q₁ = zeros(ComplexF64, moments*N, moments*m₁), zeros(ComplexF64, moments*N, moments*m₁)
	Q = zeros(ComplexF64, 2*moments, N, m₁)

    for i=1:nodes
        z = (r*exp(θ[i]*im)+c)
        resolvent = (1 ./(z .- Λ)) .* (r*exp(θ[i]*im)/nodes)

		Temp = (Y - T(z)\R) * Diagonal(resolvent)

        Q[1,:,:] .+= Temp
        for j=2:2*moments
            Q[j,:,:] .+= Temp .* z^(j-1)
        end
    end

    for i=1:moments, j=1:moments
        Q₀[(i-1)*N+1:i*N, (j-1)*m₁+1:j*m₁] .= Q[i+j-1,:,:]
        Q₁[(i-1)*N+1:i*N, (j-1)*m₁+1:j*m₁] .= Q[i+j,:,:]
    end

	S = svd!(Q₀)
    F = eigen!(S.U' * Q₁ * S.V * Diagonal(1 ./ S.S))
    Y = S.U[1:N,:] * F.vectors
    Λ = F.values

	R = similar(Y, ComplexF64)

	update_R!(Y, R, Λ, T)
    res = residuals(R, Λ, T)

	iter_debug_print(1, Λ, res, c, r, spurious)


	p = sortperm(res)
	res .= res[p]
	X .= Y[:, p][:,1:m₀]
	Λ .= Λ[p]
	R .= R[:, p]

    normalize!(X)
    Λ, X, res
end


function moments_all!(T, X::AbstractMatrix{ComplexF64}, nodes::Integer, iter::Integer;
    c=complex(0.0,0.0), r=1.0, debug=false, ϵ=10e-12, moments=2, spurious=1e-5)

    N, m₀ = size(X)
    Λ = zeros(ComplexF64, moments*m₀)
    θ = LinRange(π/nodes, 2*π-π/nodes, nodes)
    Q = zeros(ComplexF64, 2*moments, N, m₀)
    Q₀, Q₁ = zeros(ComplexF64, moments*N, moments*m₀), zeros(ComplexF64, moments*N, moments*m₀)
	Y = zeros(ComplexF64, N, m₀*moments)
    R = similar(Y, ComplexF64)


    for i=1:nodes
        z = (r*exp(θ[i]*im)+c)
		Temp = (T(z)\X) .* (r*exp(θ[i]*im)/nodes)
        Q[1,:,:] .+= Temp
        for j=2:2*moments
            Q[j,:,:] .+= Temp .* z^(j-1)
        end
    end
    for i=1:moments, j=1:moments
        Q₀[(i-1)*N+1:i*N, (j-1)*m₀+1:j*m₀] .= Q[i+j-1,:,:]
        Q₁[(i-1)*N+1:i*N, (j-1)*m₀+1:j*m₀] .= Q[i+j,:,:]
    end

    S = svd!(Q₀)
    F = eigen!(S.U' * Q₁ * S.V * Diagonal(1 ./ S.S))
    Y = S.U[1:N,:] * F.vectors
    Λ = F.values

    update_R!(Y, R, Λ, T)
    res = residuals(R, Λ, T)

	p = sortperm(res)
	res .= res[p]
	Y .= Y[:, p]
	Λ .= Λ[p]
	R .= R[:, p]


	iter_debug_print(0, Λ, res, c, r, spurious)

	m₁ = moments*m₀
    Q₀, Q₁ = zeros(ComplexF64, moments*N, m₁), zeros(ComplexF64, moments*N, m₁)
	Q = zeros(ComplexF64, 2*moments, N, m₀)

    for i=1:nodes
        z = (r*exp(θ[i]*im)+c)
        resolvent = (1 ./(z .- Λ)) .* (r*exp(θ[i]*im)/nodes)

		# Temp = (Y - T(z)\R) #* Diagonal(resolvent)
		Temp = (Y[:,1:m₀] .- T(z)\R[:,1:m₀])

        Q[1,:,:] .+= Temp[:,1:m₀]  * Diagonal(resolvent[1:m₀])
		Q[1+moments,:,:] .+= z^(moments) .* Temp[:,1:m₀] * Diagonal(resolvent[1:m₀])
        for j=2:moments
			Q[j,:,:] .+= z^(j-1) .* Temp[:,1:m₀] * Diagonal(resolvent[1:m₀])
			Q[j+moments,:,:] .+= z^(j-1+moments) .* Temp[:,1:m₀] * Diagonal(resolvent[1:m₀])
			# Q[j,:,:] .= z^(j-1) .* Temp[:,(j-1)*m₀+1:j*m₀]
			# Q[j+moments,:,:] .= z^(j-1+moments) .* Temp[:,(j-1)*m₀+1:j*m₀]
        end
    end

    for i=1:moments, j=1:moments
        Q₀[(i-1)*N+1:i*N, (j-1)*m₀+1:j*m₀] .= Q[i+j-1,:,:]
        Q₁[(i-1)*N+1:i*N, (j-1)*m₀+1:j*m₀] .= Q[i+j,:,:]
    end

	S = svd!(Q₀)
    F = eigen!(S.U' * Q₁ * S.V * Diagonal(1 ./ S.S))
    Y = S.U[1:N,:] * F.vectors
    Λ = F.values

	R = similar(Y, ComplexF64)

	update_R!(Y, R, Λ, T)
    res = residuals(R, Λ, T)

	iter_debug_print(1, Λ, res, c, r, spurious)


	# p = sortperm(res)
	# res .= res[p]
	# X .= Y[:, p][:,1:m₀]
	# Λ .= Λ[p]
	# R .= R[:, p]

    normalize!(Y)
    Λ, Y, res
end


A0 = Matrix(mmread("data/quadraticM0.mtx"))
A1 = Matrix(mmread("data/quadraticM1.mtx"))

# Random.seed!(1234)
# A0 = complex.(rand(15,15))
# A1 = complex.(rand(15,15))
# A0[:,1] .= 0

function T(z::ComplexF64)
	return (z + 0.2)*(z - 0.1).*A1 .+ A0
end

R = 0.25
# C = complex(-0.8, 0.8)
C = complex(0, 0.0)

# e, v, res = nlfeast!(T, rand(ComplexF64,15,8), 2^6, 0, c=C, r=R, spurious=1e-3,debug=true)
# e, v, res = nlfeast_moments!(T, rand(ComplexF64,15,4), 2^4, 2, c=C, r=R, moments=2, debug=true, ϵ=10e-16, spurious=1e-3)
e, v, res = nlfeast_moments_SS!(T, rand(ComplexF64,15,6), 2^4, 1, c=C, r=R, moments=2, debug=true, ϵ=10e-16, spurious=1e-3)
# e, v, res = moments_expand!(T, rand(ComplexF64,15,7), 2^3, 1, c=C, r=R, moments=2, ϵ=10e-16, spurious=1e-3)
# e, v, res = moments_all!(T, rand(ComplexF64,15,7), 2^3, 1, c=C, r=R, moments=2, ϵ=10e-16, spurious=1e-3)
# e, v, res = block_SS!(T, rand(ComplexF64,15,8), 2^8, 32, c=C, r=R)
# e, v, res = beyn(T, A0, rand(ComplexF64,1000,120), 2^9; c=complex(-1.55,0.0), r=0.05)
# display(e)
# display(res)
#
# inside(x) = in_contour(x, C, R)
# # print("\nmax res inside: ")
# # println(maximum(res[inside.(e)]))
# # print("number inside : ")
# # println(size(e[inside.(e)])[1])
# # # display(e)
# # display(res[inside.(e)])
#
# in_eig = e[inside.(e)]
# in_res = res[inside.(e)]
# in_res_conv = in_res[in_res .<= 1e-3]
# in_eig_conv = in_eig[in_res .<= 1e-3]
# if size(in_eig, 1) > 0
#     print("\nmax res inside: ")
#     println(maximum(res[inside.(e)]))
# end
# print("max res inside non spurious: ")
# println(maximum(in_res_conv))
# print("\nnumber inside : ")
# println(size(e[inside.(e)])[1])
# print("number inside converged : ")
# println(size(in_eig_conv)[1])

# e, v, res = companion([(A0 .- 0.02.*A1), (0.1.*A1), A1])

info(e, v, res, C, R)

display(res)
println()
display(e)
