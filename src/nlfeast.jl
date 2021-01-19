
function nlfeast!(T, X::AbstractMatrix{ComplexF64}, nodes::Integer, iter::Integer;
    c=complex(0.0, 0.0), r=1.0, debug=false, ϵ=10e-12, store=true, spurious=1e-5,
    factorizer=lu, left_divider=ldiv!)

    N, m₀ = size(X)
    Λ, res = zeros(ComplexF64, m₀), Array{Float64}(undef, m₀)
    θ = LinRange(π / nodes, 2 * π - π / nodes, nodes)
    Q₀, Q₁, R = similar(X, ComplexF64), similar(X, ComplexF64), similar(X, ComplexF64)
    A, B = zeros(ComplexF64, m₀, m₀), zeros(ComplexF64, m₀, m₀)

    qt, rt = qr!(X)
    X .= Matrix(qt)

    l = ReentrantLock()

    if store
        facts = Array{Factorization}(undef, nodes)
        Threads.@threads for i = 1:nodes
            z = (r * exp(θ[i] * im) + c)
            tempfact = factorizer(T(z))
            lock(l) do
                facts[i] = tempfact
            end
            if debug print("*") end
        end
        if debug println() end
    end

    for nit = 0:iter

        Q₀ .= 0
        Q₁ .= 0
        # l = ReentrantLock()

        Threads.@threads for i = 1:nodes
            z = (r * exp(θ[i] * im) + c)
            Tinv = similar(X, ComplexF64)
            if nit == 0
                if store
                    left_divider(Tinv, facts[i], X)
                    Tinv .*= (r * exp(θ[i] * im) / nodes)
                else
                    Tinv .= (T(z) \ X) .* (r * exp(θ[i] * im) / nodes)
                end
            else
                resolvent = (1 ./ (z .- Λ)) .* (r * exp(θ[i] * im) / nodes)
                if store
                    left_divider(Tinv, facts[i], R)
                    Tinv .= X - Tinv
                else
                    Tinv .= (X - T(z) \ R)
                end
                rmul!(Tinv,  Diagonal(resolvent))
            end
            lock(l) do
                Q₀ .+= Tinv
                Q₁ .+= Tinv .* z
            end
    		if debug print(".") end
        end
		if debug println() end

		beyn_svd_step!(Q₀, Q₁, A, B, X, Λ)

        update_R!(X, R, Λ, T)
        res .= residuals(R, Λ, T)

        if debug
            iter_debug_print(nit, Λ, res, c, r, spurious)
        end

		res_inside = res[in_contour.(Λ, c, r)]
        if size(res_inside, 1) > 0 && maximum(res_inside) < ϵ
            break
        end
		if nit > 1 && sum(res_inside .< spurious) > 0 && maximum(res_inside[res_inside .< spurious]) < ϵ
			break
		end
    end

    normalize!(X)
    Λ, X, res
end


function nlfeast_it!(T, X::AbstractMatrix{ComplexF64}, nodes::Integer, iter::Integer;
    c=complex(0.0, 0.0), r=1.0, debug=false, ϵ=0.05)

    N, m₀ = size(X)
    Λ = zeros(ComplexF64, m₀)
    θ = LinRange(π / nodes, 2 * π - π / nodes, nodes)
    Q₀, Q₁ = zeros(ComplexF64, N, m₀), zeros(ComplexF64, N, m₀)
    R = similar(X, ComplexF64)
    Temp = similar(X, ComplexF64)
    resolvent = similar(Λ)
    Tinv = rand(ComplexF64, nodes, N, m₀)
    A, B = zeros(ComplexF64, m₀, m₀), zeros(ComplexF64, m₀, m₀)
    res = Array{Float64}(undef, size(R, 2))

    for i = 1:nodes
        z = (r * exp(θ[i] * im) + c)
        # Temp .= (T(z)\X) .* (r*exp(θ[i]*im)/nodes)
        for j = 1:m₀
            # Tinv[i,:,j], log = gmres(T(z), X[:,j]; tol=10e-8, log=true)
            Tinv[i,:,j] .= bicgstabl(T(z), X[:,j], 2; tol=1e-3)
        end
        Temp .= Tinv[i,:,:] .* (r * exp(θ[i] * im) / nodes)
        Q₀ .+= Temp
        Q₁ .+= Temp .* z
		if debug print(".") end
    end
	if debug println() end

    beyn_svd_step!(Q₀, Q₁, A, B, X, Λ)
    # beyn_qr_step!(Q₀, Q₁, X, Λ)

    update_R!(X, R, Λ, T)

    if (iter == 0)
        res .= residuals(R, Λ, T)
    end
    if debug
        iter_debug_print(0, Λ, res, c, r)
    end

    for nit = 1:iter

        Q₀ .= 0
        Q₁ .= 0

        for i = 1:nodes
            z = (r * exp(θ[i] * im) + c)
            resolvent .= (1 ./ (z .- Λ))
            # Tinv[i,:,:] .= T(z)\R
            for j = 1:m₀
                # Tinv[i,:,j], log = gmres!(Tinv[i,:,j], T(z), R[:,j]; log=true, maxiter=1000, tol=1e-3)
                # Tinv[i,:,j], log = gmres(T(z), R[:,j]; log=true, maxiter=1000, tol=1e-3)
                Tinv[i,:,j], log = bicgstabl!(Tinv[i,:,j], T(z), R[ :,j], 1; log=true, tol=1e-8)
                # Tinv[i,:,j], log = bicgstabl(T(z), R[:,j], 2; log=true, max_mv_products=1000, tol=1e-3)
                # if !log.isconverged println(log) end
                # print(log.iters)
                # print(":\t")
                # println(maximum(log[:resnorm]))
            end
            Temp .= X - Tinv[i,:,:]
            rmul!(Temp, (r * exp(θ[i] * im) / nodes) .* Diagonal(resolvent))
            Q₀ .+= Temp
            Q₁ .+= Temp .* z
    		if debug print(".") end
        end
		if debug println() end

        beyn_svd_step!(Q₀, Q₁, A, B, X, Λ)
        # beyn_qr_step!(Q₀, Q₁, X, Λ)

        update_R!(X, R, Λ, T)
        res .= residuals(R, Λ, T)

        if debug
            iter_debug_print(nit, Λ, res, c, r)
        end

        if maximum(res[in_contour.(Λ, c, r)]) < ϵ
            break
        end
    end

    normalize!(X)
    Λ, X, residuals(R, Λ, T)
end

function nlfeast_moments!(T, X::AbstractMatrix{ComplexF64}, nodes::Integer, iter::Integer;
    c=complex(0.0, 0.0), r=1.0, debug=false, ϵ=10e-12, moments=2, store=true, spurious=1e-5)

    N, m₀ = size(X)
    Λ = zeros(ComplexF64, moments * m₀)
    θ = LinRange(π / nodes, 2 * π - π / nodes, nodes)
    Q = zeros(ComplexF64, 2 * moments, N, m₀)
    Y = zeros(ComplexF64, N, m₀ * moments)
    Q₀, Q₁ = zeros(ComplexF64, moments * N, moments * m₀), zeros(ComplexF64, moments * N, moments * m₀)
    Tinv, Temp, R = similar(X, ComplexF64), similar(X, ComplexF64), similar(Y, ComplexF64)
    resolvent = zeros(ComplexF64, m₀)
    A, B = zeros(ComplexF64, moments * m₀, moments * m₀), zeros(ComplexF64, moments * m₀, moments * m₀)
    res = Array{Float64}(undef, size(R, 2))
    Y = zeros(ComplexF64, N, m₀ * moments)
	if store
	    M = Array{ComplexF64}(undef, (nodes, N, N))
	    facts = Array{LU}(undef, nodes)
	end

    for i = 1:nodes
        z = (r * exp(θ[i] * im) + c)

		if store
			# M[i,:,:] .= ComplexF32.(T(z))
			M[i,:,:] .= T(z)
	        facts[i] = lu!(M[i,:,:])
	        Tinv .= X
	        ldiv!(facts[i], Tinv)
	        Temp .= (r * exp(θ[i] * im) / nodes) .* Tinv
		else
			Temp .= (T(z) \ X) .* (r * exp(θ[i] * im) / nodes)
			# Tinv .= (ComplexF32.(T(z))\ComplexF32.(X)) .* (r*exp(θ[i]*im)/nodes)
		end


        Q[1,:,:] .+= Temp
        for j = 2:2 * moments
            Q[j,:,:] .+= Temp .* z^(j - 1)
        end
    end
    for i in 1:moments, j in 1:moments
        Q₀[(i - 1) * N + 1:i * N, (j - 1) * m₀ + 1:j * m₀] .= Q[i + j - 1,:,:]
        Q₁[(i - 1) * N + 1:i * N, (j - 1) * m₀ + 1:j * m₀] .= Q[i + j,:,:]
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
	X .= Y[:,1:m₀]
    # update_R!(Y, R, Λ, T)
    # res .= residuals(R, Λ, T)
    # p = sortperm(res)
	# p = sortperm(abs.(Λ .- c))
    # res .= res[p]
    # X .= Y[:, p][:,1:m₀]
    # Λ .= Λ[p]
    # R .= R[:, p]

    if debug
        iter_debug_print(0, Λ[1:m₀], res[1:m₀], c, r, spurious)
		# iter_debug_print(0, Λ, res, c, r, spurious)
    end

    for nit = 1:iter

        Q .= 0

        for i = 1:nodes
            z = (r * exp(θ[i] * im) + c)
            resolvent .= (1 ./ (z .- Λ[1:m₀])) .* (r * exp(θ[i] * im) / nodes)


			if store
	            Tinv .= R[:, 1:m₀]
	            ldiv!(facts[i], Tinv)
	            Temp .= X - Tinv
    			else
				Temp .= (X - T(z) \ R[:, 1:m₀])
				# Tinv .= (X - (ComplexF32.(T(z))\ComplexF32.(R[:, 1:m₀])))
			end
            rmul!(Temp, Diagonal(resolvent))

            Q[1,:,:] .+= Temp
            for j = 2:2 * moments
                Q[j,:,:] .+= Temp .* z^(j - 1)
            end
        end

        for i in 1:moments, j in 1:moments
            Q₀[(i - 1) * N + 1:i * N, (j - 1) * m₀ + 1:j * m₀] .= Q[i + j - 1,:,:]
            Q₁[(i - 1) * N + 1:i * N, (j - 1) * m₀ + 1:j * m₀] .= Q[i + j,:,:]
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

        update_R_moments!(Y, R, Λ, res, T, c, r)
		X .= Y[:,1:m₀]
        # res .= residuals(R, Λ, T)
        # # p = sortperm(res)
        # # res .= res[p]
        # # X .= Y[:, p][:,1:m₀]
        # # Λ .= Λ[p]
        # # R .= R[:, p]

        if debug
            iter_debug_print(nit, Λ[1:m₀], res[1:m₀], c, r, spurious)
			# iter_debug_print(nit, Λ, res, c, r, spurious)
        end
		res_inside = res[1:m₀][in_contour.(Λ[1:m₀], c, r)]
        if size(res_inside, 1) > 0 && maximum(res_inside) < ϵ
            break
        end
		if nit > 1 && sum(res_inside .< spurious) > 0 && maximum(res_inside[res_inside .< spurious]) < ϵ
			break
		end
    end

	p = sortperm(res)
	res .= res[p]
	X .= Y[:, p][:,1:m₀]
	Λ .= Λ[p]
	R .= R[:, p]

    normalize!(X)
    # Λ[1:m₀], X, res[1:m₀]
	Λ, Y, res
end
