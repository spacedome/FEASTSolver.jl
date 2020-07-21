
function nlfeast_moments_all!(T, X::AbstractMatrix{ComplexF64}, nodes::Integer, iter::Integer;
    c=complex(0.0,0.0), r=1.0, debug=false, ϵ=10e-12, moments=2, store=true, spurious=1e-5)

    N, m₀ = size(X)
    Λ = zeros(ComplexF64, moments*m₀)
    θ = LinRange(π/nodes, 2*π-π/nodes, nodes)
    Q = zeros(ComplexF64, 2*moments, N, m₀)
    Y = zeros(ComplexF64, N, m₀*moments)
    Q₀, Q₁ = zeros(ComplexF64, moments*N, moments*m₀), zeros(ComplexF64, moments*N, moments*m₀)
    Tinv, Temp, R = similar(X, ComplexF64), similar(X, ComplexF64), similar(Y, ComplexF64)
	Rtemp = similar(Y, ComplexF64)
    resolvent = zeros(ComplexF64, moments*m₀)
    A, B = zeros(ComplexF64, moments*m₀, moments*m₀), zeros(ComplexF64, moments*m₀, moments*m₀)
    res = Array{Float64}(undef, size(R, 2))
	if store
	    M = Array{ComplexF64}(undef, (nodes, N,N))
	    facts = Array{LU}(undef, nodes)
	end

    for i=1:nodes
        z = (r*exp(θ[i]*im)+c)

		if store
			# M[i,:,:] .= ComplexF32.(T(z))
			M[i,:,:] .= T(z)
	        facts[i] = lu!(M[i,:,:])
	        Tinv .= X
	        ldiv!(facts[i], Tinv)
	        Temp .= (r*exp(θ[i]*im)/nodes) .* Tinv
		else
			Temp .= (T(z)\X) .* (r*exp(θ[i]*im)/nodes)
			# Tinv .= (ComplexF32.(T(z))\ComplexF32.(X)) .* (r*exp(θ[i]*im)/nodes)
		end


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
    mul!(A, S.U', Q₁)
    mul!(B, A, S.V)
    mul!(A, B, Diagonal(1 ./ S.S))
    F = eigen!(A)
    mul!(Y, S.U[1:N,:], F.vectors)
    Λ .= F.values

    # beyn_svd_step!(Q₀, Q₁, A, B, X, Λ)
    # beyn_qr_step!(Q₀, Q₁, X, Λ)

	update_R_moments!(Y, R, Λ, res, T, c, r)
    # update_R!(Y, R, Λ, T)
    # res .= residuals(R, Λ, T)
    # p = sortperm(res)
	# # p = sortperm(abs.(Λ .- c))
    # res .= res[p]
    # Y .= Y[:, p]
    # Λ .= Λ[p]
    # R .= R[:, p]

    if debug
        # iter_debug_print(0, Λ[1:m₀], res[1:m₀], c, r, spurious)
		iter_debug_print(0, Λ, res, c, r, spurious)
    end

    for nit=1:iter

        Q .= 0

        for i=1:nodes
            z = (r*exp(θ[i]*im)+c)
            resolvent .= (1 ./(z .- Λ)) .* (r*exp(θ[i]*im)/nodes)

			if store
	            Tinv .= R[:, 1:m₀]
	            ldiv!(facts[i], Tinv)
	            Temp .= X - Tinv
			else
				# Temp .= (X - T(z)\R[:, 1:m₀])
				Rtemp .= (Y - T(z)\R)
				# Tinv .= (X - (ComplexF32.(T(z))\ComplexF32.(R[:, 1:m₀])))
			end
            rmul!(Rtemp, Diagonal(resolvent))


			# Q[1,:,:] .+= Rtemp[:,1:m₀]
			# Q[1+moments,:,:] .+= z^(moments) .* Rtemp[:,1:m₀]
            for j=1:moments
                # Q[j,:,:] .+= z^(j-1) .* Rtemp[:,(j-1)*m₀+1:j*m₀]
				# Q[j+moments,:,:] .+= z^(j-1+moments) .* Rtemp[:,(j-1)*m₀+1:j*m₀]
				Q[j,:,:] .+= z^(j-1) .* Rtemp[:,1:m₀]
				Q[j+moments,:,:] .+= z^(j-1+moments) .* Rtemp[:,1:m₀]
            end
			# Q[1,:,:] .+= Rtemp[:,1:m₀]
			# Q[2,:,:] .+= z .* Rtemp[:,1:m₀]
            # for j=2:moments
            #     Q[2*j-1,:,:] .+= z^(2*j-2) .* Rtemp[:,(j-1)*m₀+1:j*m₀]
			# 	Q[2*j,:,:] .+= z^(2*j-1) .* Rtemp[:,(j-1)*m₀+1:j*m₀]
			# 	# Q[j,:,:] .+= z^(j-1) .* Rtemp[:,1:m₀]
			# 	# Q[j+moments,:,:] .+= z^(j-1+moments) .* Rtemp[:,1:m₀]
            # end
        end

        for i=1:moments, j=1:moments
            Q₀[(i-1)*N+1:i*N, (j-1)*m₀+1:j*m₀] .= Q[i+j-1,:,:]
            Q₁[(i-1)*N+1:i*N, (j-1)*m₀+1:j*m₀] .= Q[i+j,:,:]
        end

        S = svd!(Q₀)
        mul!(A, S.U', Q₁)
        mul!(B, A, S.V)
        mul!(A, B, Diagonal(1 ./ S.S))
        F = eigen!(A)
        mul!(Y, S.U[1:N,:], F.vectors)
        Λ .= F.values

        # beyn_svd_step!(Q₀, Q₁, A, B, X, Λ)
        # beyn_rr_step!(Q₀, Q₁, A, B, X, Λ)
        # beyn_qr_step!(Q₀, Q₁, X, Λ)

        update_R_moments_all!(Y, R, Λ, res, T, c, r)

        if debug
            # iter_debug_print(nit, Λ[1:m₀], res[1:m₀], c, r, spurious)
			iter_debug_print(nit, Λ, res, c, r, spurious)
        end
		res_inside = res[in_contour.(Λ, c, r)]
        if size(res_inside, 1) > 0 && maximum(res_inside) < ϵ
            break
        end
		if nit>1 && sum(res_inside .< spurious) > 0 && maximum(res_inside[res_inside .< spurious]) < ϵ
			break
		end
    end

    normalize!(Y)
    Λ, Y, res
end

function nlfeast_moments_SS!(T, X::AbstractMatrix{ComplexF64}, nodes::Integer, iter::Integer;
    c=complex(0.0,0.0), r=1.0, debug=false, ϵ=10e-12, moments=2, spurious=1e-5)

    N, m₀ = size(X)
    Λ = zeros(ComplexF64, moments*m₀)
    θ = LinRange(π/nodes, 2*π-π/nodes, nodes)
    Q = zeros(ComplexF64, N, (2*moments+1)*m₀)
    Y = zeros(ComplexF64, N, m₀*moments)
    Tinv, Temp, V, R = similar(X, ComplexF64), similar(X, ComplexF64), similar(X, ComplexF64), similar(Y, ComplexF64)
    resolvent = zeros(ComplexF64, m₀)
    A, B = zeros(ComplexF64, moments*m₀, moments*m₀), zeros(ComplexF64, moments*m₀, moments*m₀)
    res = Array{Float64}(undef, size(R, 2))
    M = Array{ComplexF64}(undef, (nodes, N,N))
    facts = Array{LU}(undef, nodes)
	K = moments*m₀
	# X = Matrix(qr(X).Q)

    for i=1:nodes
        z = (r*exp(θ[i]*im)+c)
        Tinv .= (T(z)\X) .* (r*exp(θ[i]*im)/nodes)
        for j=1:2*moments+1
            Q[:, (j-1)*m₀+1:j*m₀] .+= z^(j-1) .* Tinv
        end
    end

	Q₀, Q₁ = zeros(ComplexF64, K, K), zeros(ComplexF64, K, K)


    for i=1:moments, j=1:moments
        Q₀[(i-1)*m₀+1:i*m₀, (j-1)*m₀+1:j*m₀] .= X' * Q[:, (i+j-1)*m₀+1:(i+j)*m₀]
        Q₁[(i-1)*m₀+1:i*m₀, (j-1)*m₀+1:j*m₀] .= X' * Q[:, (i+j)*m₀+1:(i+j+1)*m₀]
    end


	V = svd(Q₀)
	n = min(count(V.S/V.S[1] .> 1e-13), K)

	H1 = V.U[:,1:n]' * Q₁ * V.V[:,1:n]
    H0 = V.U[:,1:n]' * Q₀ * V.V[:,1:n]

    Λ, Xq = eigen!(H1, H0)
    X = Q[:,1:K] * V.V[:,1:n] *  Xq
    # Λ = Λ .+ c


	res = zeros(size(Λ, 1))
	R = similar(X)
    update_R_moments!(X, R, Λ, res, T, c, r)
    X = X[:,1:m₀]

    if debug
        iter_debug_print(0, Λ[1:m₀], res[1:m₀], c, r)
    end

    for nit=1:iter

        Q .= 0
		L = rand(ComplexF64, size(X))

        for i=1:nodes
            z = (r*exp(θ[i]*im)+c)
            resolvent .= (1 ./(z .- Λ[1:m₀])) .* (r*exp(θ[i]*im)/nodes)

            Tinv .= (X - T(z)\R[:, 1:m₀])

            rmul!(Temp, Diagonal(resolvent))

	        for j=1:2*moments+1
	        	Q[:, (j-1)*m₀+1:j*m₀] .+= z^(j-1) .* Tinv
	        end
        end

        for i=1:moments, j=1:moments
			Q₀[(i-1)*m₀+1:i*m₀, (j-1)*m₀+1:j*m₀] .= L' * Q[:,(i+j-2)*m₀+1:(i+j-1)*m₀]
	        Q₁[(i-1)*m₀+1:i*m₀, (j-1)*m₀+1:j*m₀] .= L' * Q[:,(i+j-1)*m₀+1:(i+j)*m₀]
        end

		V = svd(Q₀)
		n = min(count(V.S/V.S[1] .> 1e-13), K)

		H1 = V.U[:,1:n]' * Q₁ * V.V[:,1:n]
	    H0 = V.U[:,1:n]' * Q₀ * V.V[:,1:n]

	    Λ, Xq = eigen!(H1, H0)
	    X = Q[:,1:K] * V.V[:,1:n] *  Xq
	    # Λ = Λ .+ c

		res = zeros(size(Λ, 1))
		R = similar(X)
		update_R_moments!(X, R, Λ, res, T, c, r)
		X = X[:,1:m₀]

        if debug
            iter_debug_print(nit, Λ[1:m₀], res[1:m₀], c, r)
        end

        # if maximum(res[1:m₀][in_contour.(Λ[1:m₀], c, r)]) < ϵ
        #     break
        # end
    end

    normalize!(X)
    Λ[1:m₀], X, res[1:m₀]
end
