import LinearAlgebra: qr, lu, Diagonal
function nlfeast!(T, X::AbstractMatrix{ComplexF64}, nodes::Integer, iter::Integer;
    c=complex(0.0,0.0), r=1.0, debug=false, ϵ=0.05)

    N, m₀ = size(X)
    IM = Matrix(I, m₀, m₀)
    Λ = zeros(ComplexF64, m₀)
    θ = LinRange(π/nodes, 2*π-π/nodes, nodes)
    Q₀, Q₁ = zeros(ComplexF64, N, m₀), zeros(ComplexF64, N, m₀)
    Tinv, R = similar(X, ComplexF64), similar(X, ComplexF64)
    resolvent = similar(Λ)

    TLU = [qr(T(r*exp(ang*im)+c)) for ang in θ]

    for i=1:nodes
	z = (r*exp(θ[i]*im)+c)
	# Tinv .= (T(z)\X)
	# Q₀ .+= Tinv .* (r*exp(θ[i]*im)/nodes)
	# Q₁ .+= Tinv .* (r*exp(θ[i]*im)/nodes) .* (r*exp(θ[i]*im)+c)
	Tinv .= (TLU[i] \ X) .* (r*exp(θ[i]*im)/nodes) 
	Q₀ .+= Tinv 
	Q₁ .+= Tinv .* z
    end
    
    F = eigen!(X' * Q₁, X' * Q₀)
    # X .= Q₀ * F.vectors
    mul!(X, Q₀, F.vectors)
    Λ .= F.values

    for nit=1:iter

        update_R!(X, R, Λ, T)

        Q₀ .= 0
        Q₁ .= 0

        for i=1:nodes
            z = (r*exp(θ[i]*im)+c)
	    resolvent .= (1 ./(z .- Λ))
	    # Tinv .= (X - T(z)\R)
	    # Q₀ .+= Tinv * resolvent .* (r*exp(θ[i]*im)/nodes)
	    # Q₁ .+= Tinv * resolvent .* (r*exp(θ[i]*im)/nodes) .* z
	    Tinv .= X
	    Tinv .-= (TLU[i]\R)
	    rmul!(Tinv, (r*exp(θ[i]*im)/nodes) .* Diagonal(resolvent)) 
	    Q₀ .+= Tinv 
	    Q₁ .+= Tinv .* z
        end

        F = eigen!(X' * Q₁, X' * Q₀)
        # X .= Q₀ * F.vectors
        mul!(X, Q₀, F.vectors)
        Λ .= F.values

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

