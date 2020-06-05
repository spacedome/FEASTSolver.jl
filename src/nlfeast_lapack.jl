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
