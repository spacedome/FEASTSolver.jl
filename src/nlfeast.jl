import LinearAlgebra: qr, lu, Diagonal, svd
import IterativeSolvers: gmres!
function nlfeast!(T, X::AbstractMatrix{ComplexF64}, nodes::Integer, iter::Integer;
    c=complex(0.0,0.0), r=1.0, debug=false, ϵ=0.05)

    N, m₀ = size(X)
    Λ = zeros(ComplexF64, m₀)
    θ = LinRange(π/nodes, 2*π-π/nodes, nodes)
    Q₀, Q₁ = zeros(ComplexF64, N, m₀), zeros(ComplexF64, N, m₀)
    Tinv, R = similar(X, ComplexF64), similar(X, ComplexF64)
    resolvent = similar(Λ)
    A, B = zeros(ComplexF64, m₀, m₀), zeros(ComplexF64, m₀, m₀)

    for i=1:nodes
	z = (r*exp(θ[i]*im)+c)
	Tinv .= (T(z)\X) .* (r*exp(θ[i]*im)/nodes)
	Q₀ .+= Tinv
	Q₁ .+= Tinv .* z
    end

    # S = svd!(Q₀)
    # mul!(A, S.U', Q₁)
    # mul!(B, A, S.V)
    # mul!(A, B, Diagonal(1 ./ S.S))
    # F = eigen!(A)
    # mul!(X, S.U, F.vectors)
    # Λ .= F.values 

    qt, rt = qr(Q₀)
    qt = Matrix(qt)
    F = eigen!(qt' * Q₁ * inv(rt))
    mul!(X, qt, F.vectors)
    Λ .= F.values .+ c

    if (iter == 0)
        update_R!(X, R, Λ, T)
    end

    for nit=1:iter

        update_R!(X, R, Λ, T)

        Q₀ .= 0
        Q₁ .= 0

        for i=1:nodes
            z = (r*exp(θ[i]*im)+c)
	    resolvent .= (1 ./(z .- Λ))
	    Tinv .= (X - T(z)\R)
	    rmul!(Tinv, (r*exp(θ[i]*im)/nodes) .* Diagonal(resolvent)) 
	    Q₀ .+= Tinv
	    Q₁ .+= Tinv .* z
        end

    # S = svd!(Q₀)
    # mul!(A, S.U', Q₁)
    # mul!(B, A, S.V)
    # mul!(A, B, Diagonal(1 ./ S.S))
    # F = eigen!(A)
    # mul!(X, S.U, F.vectors)
    # Λ .= F.values 

    qt, rt = qr(Q₀)
    qt = Matrix(qt)
    F = eigen!(qt' * Q₁ * inv(rt))
    mul!(X, qt, F.vectors)
    Λ .= F.values .+ c
	
	# mul!(A, X', Q₁)
	# mul!(B, X', Q₀)
	# F = eigen!(A, B)
	# mul!(X, Q₀, F.vectors)
	# Λ .= F.values

        if debug println(nit) end
    end
    
    normalize!(X)
    Λ, X, residuals(R, Λ, T)
end

function nlfeast_it!(T, X::AbstractMatrix{ComplexF64}, nodes::Integer, iter::Integer;
    c=complex(0.0,0.0), r=1.0, debug=false, ϵ=0.05)

    N, m₀ = size(X)
    Λ = zeros(ComplexF64, m₀)
    θ = LinRange(π/nodes, 2*π-π/nodes, nodes)
    Q₀, Q₁ = zeros(ComplexF64, N, m₀), zeros(ComplexF64, N, m₀)
    Tinv, R = similar(X, ComplexF64), similar(X, ComplexF64)
    resolvent = similar(Λ)
    A, B = zeros(ComplexF64, m₀, m₀), zeros(ComplexF64, m₀, m₀)

    for i=1:nodes
	z = (r*exp(θ[i]*im)+c)
	# Tinv .= (T(z)\X) .* (r*exp(θ[i]*im)/nodes)
	for j=1:m₀
		gmres!(Tinv[:,j], T(z), X[:,j]; tol=1e-3)
	end
	Tinv .*= (r*exp(θ[i]*im)/nodes)
	Q₀ .+= Tinv
	Q₁ .+= Tinv .* z
    end

    # S = svd!(Q₀)
    # mul!(A, S.U', Q₁)
    # mul!(B, A, S.V)
    # mul!(A, B, Diagonal(1 ./ S.S))
    # F = eigen!(A)
    # mul!(X, S.U, F.vectors)
    # Λ .= F.values

    qt, rt = qr(Q₀)
    qt = Matrix(qt)
    F = eigen!(qt' * Q₁ * inv(rt))
    mul!(X, qt, F.vectors)
    Λ .= F.values
    
    for nit=1:iter

        update_R!(X, R, Λ, T)

        Q₀ .= 0
        Q₁ .= 0
	# Tinv .= 0

        for i=1:nodes
            z = (r*exp(θ[i]*im)+c)
	    resolvent .= (1 ./(z .- Λ))
	    Tinv .= (X - T(z)\R)
	    # for j=1:m₀
		 # gmres!(Tinv[:,j], T(z), R[:,j]; tol=1e-3)
	    # end
	    # Tinv .= X - Tinv
	    rmul!(Tinv, (r*exp(θ[i]*im)/nodes) .* Diagonal(resolvent)) 
	    Q₀ .+= Tinv
	    Q₁ .+= Tinv .* z
        end

    # S = svd!(Q₀)
    # mul!(A, S.U', Q₁)
    # mul!(B, A, S.V)
    # mul!(A, B, Diagonal(1 ./ S.S))
    # F = eigen!(A)
    # mul!(X, S.U, F.vectors)
    # Λ .= F.values

    qt, rt = qr(Q₀)
    qt = Matrix(qt)
    F = eigen!(qt' * Q₁ * inv(rt))
    mul!(X, qt, F.vectors)
    Λ .= F.values
	
	# mul!(A, X', Q₁)
	# mul!(B, X', Q₀)
	# F = eigen!(A, B)
	# mul!(X, Q₀, F.vectors)
	# Λ .= F.values

        if debug println(nit) end
    end

    normalize!(X)
    Λ, X, residuals(R, Λ, T)
end

function nlfeast_opt!(T, X::AbstractMatrix{ComplexF64}, nodes::Integer, iter::Integer;
    c=complex(0.0,0.0), r=1.0, debug=false, ϵ=0.05)

    N, m₀ = size(X)
    IM = Matrix(I, m₀, m₀)
    Λ = zeros(ComplexF64, m₀)
    θ = LinRange(π/nodes, 2*π-π/nodes, nodes)
    Q₀, Q₁ = zeros(ComplexF64, N, m₀), zeros(ComplexF64, N, m₀)
    Tinv, R = similar(X, ComplexF64), similar(X, ComplexF64)
    resolvent = similar(Λ)

    A, B = zeros(ComplexF64, m₀, m₀), zeros(ComplexF64, m₀, m₀)

    alpha, beta = similar(Λ), similar(Λ)
    vl, vr = zeros(ComplexF64, 1, m₀), zeros(ComplexF64, m₀, m₀)
    work, rwork = zeros(ComplexF64, 1), zeros(Float64, 8*m₀)

    TLU = [qr(T(r*exp(ang*im)+c)) for ang in θ]

    for i=1:nodes
	z = (r*exp(θ[i]*im)+c)
	Tinv .= (TLU[i] \ X) .* (r*exp(θ[i]*im)/nodes) 
	Q₀ .+= Tinv 
	Q₁ .+= Tinv .* z
    end

    mul!(A, X', Q₁)
    mul!(B, X', Q₀)
    lapack_ggev!(A, B, alpha, beta, vl, vr, work, rwork) 
    mul!(X, Q₀, vr)
    Λ .= alpha ./ beta

    for nit=1:iter

        update_R!(X, R, Λ, T)

        Q₀ .= 0
        Q₁ .= 0

        for i=1:nodes
            z = (r*exp(θ[i]*im)+c)
	    resolvent .= (1 ./(z .- Λ))
	    Tinv .= X
	    Tinv .-= (TLU[i]\R)
	    rmul!(Tinv, (r*exp(θ[i]*im)/nodes) .* Diagonal(resolvent)) 
	    Q₀ .+= Tinv 
	    Q₁ .+= Tinv .* z
        end

        mul!(A, X', Q₁)
        mul!(B, X', Q₀)
        lapack_ggev!(A, B, alpha, beta, vl, vr, work, rwork) 
        mul!(X, Q₀, vr)
        Λ .= alpha ./ beta

        if debug println(nit) end
    end

    normalize!(X)
    Λ, X, residuals(R, Λ, T)
end

function update_R!(X::AbstractMatrix{ComplexF64}, R::AbstractMatrix{ComplexF64}, Λ::Array{ComplexF64}, T::Function)
    for i=1:size(X, 2)
        X[:,i] .= X[:,i]/norm(X[:,i])
        R[:,i] .= T(Λ[i])*X[:,i]
    end
end

function normalize!(X::AbstractMatrix{ComplexF64})
    for i=1:size(X, 2)
        X[:,i] .= X[:,i]/norm(X[:,i])
    end
end

function residuals(R::AbstractMatrix{ComplexF64}, Λ::Array{ComplexF64}, T::Function)
    res = Array{Float64}(undef, size(R, 2))
    for i=1:size(R, 2)
        res[i] = norm(R[:,i])/norm(T(Λ[i]))
    end
    res
end

