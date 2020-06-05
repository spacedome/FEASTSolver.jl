import IterativeSolvers: gmres!, bicgstabl!, gmres, bicgstabl
function nlfeast!(T, X::AbstractMatrix{ComplexF64}, nodes::Integer, iter::Integer;
    c=complex(0.0,0.0), r=1.0, debug=false, ϵ=10e-12)

    N, m₀ = size(X)
    Λ = zeros(ComplexF64, m₀)
    θ = LinRange(π/nodes, 2*π-π/nodes, nodes)
    Q₀, Q₁ = zeros(ComplexF64, N, m₀), zeros(ComplexF64, N, m₀)
    Tinv, R = similar(X, ComplexF64), similar(X, ComplexF64)
    resolvent = similar(Λ)
    A, B = zeros(ComplexF64, m₀, m₀), zeros(ComplexF64, m₀, m₀)
    res = Array{Float64}(undef, size(R, 2))

    for i=1:nodes
        z = (r*exp(θ[i]*im)+c)
        Tinv .= (T(z)\X) .* (r*exp(θ[i]*im)/nodes)
        Q₀ .+= Tinv
        Q₁ .+= Tinv .* z
    end

    beyn_svd_step!(Q₀, Q₁, A, B, X, Λ)
    # beyn_qr_step!(Q₀, Q₁, X, Λ)

    update_R!(X, R, Λ, T)

    if (iter == 0)
        res .= residuals(R, Λ, T)
    end

    for nit=1:iter

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

        beyn_svd_step!(Q₀, Q₁, A, B, X, Λ)
        # beyn_rr_step!(Q₀, Q₁, A, B, X, Λ)
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
    Λ, X, res
end

function nlfeast_it!(T, X::AbstractMatrix{ComplexF64}, nodes::Integer, iter::Integer;
    c=complex(0.0,0.0), r=1.0, debug=false, ϵ=0.05)

    N, m₀ = size(X)
    Λ = zeros(ComplexF64, m₀)
    θ = LinRange(π/nodes, 2*π-π/nodes, nodes)
    Q₀, Q₁ = zeros(ComplexF64, N, m₀), zeros(ComplexF64, N, m₀)
    R = similar(X, ComplexF64)
    resolvent = similar(Λ)
    Tinv = rand(ComplexF64, nodes, N, m₀)
    A, B = zeros(ComplexF64, m₀, m₀), zeros(ComplexF64, m₀, m₀)
    res = Array{Float64}(undef, size(R, 2))

    for i=1:nodes
        z = (r*exp(θ[i]*im)+c)
        # Tinv .= (T(z)\X) .* (r*exp(θ[i]*im)/nodes)
        for j=1:m₀
            # Tinv[i,:,j], log = gmres(T(z), X[:,j]; tol=10e-8, log=true)
            Tinv[i,:,j] .= bicgstabl(T(z), X[:,j], 2; tol=1e-8)
        end
        Tinv[i,:,:] .*= (r*exp(θ[i]*im)/nodes)
        Q₀ .+= Tinv[i,:,:]
        Q₁ .+= Tinv[i,:,:] .* z
    end

    beyn_svd_step!(Q₀, Q₁, A, B, X, Λ)
    # beyn_qr_step!(Q₀, Q₁, X, Λ)

    update_R!(X, R, Λ, T)

    if (iter == 0)
        res .= residuals(R, Λ, T)
    end

    for nit=1:iter

        Q₀ .= 0
        Q₁ .= 0
        # Tinv .= 0

        for i=1:nodes
            z = (r*exp(θ[i]*im)+c)
            resolvent .= (1 ./(z .- Λ))
            # Tinv .= (X - T(z)\R)
            for j=1:m₀
                # Tinv[i,:,j], log = gmres!(Tinv[i,:,j], T(z), R[:,j]; log=true, maxiter=1000, tol=10e-8)
                # Tinv[i,:,j], log = gmres(T(z), R[:,j]; log=true, maxiter=1000, tol=10e-8)
                # Tinv[i,:,j], log = bicgstabl(T(z), R[:,j], 2; log=true, max_mv_products=1000, tol=1e-14)
                Tinv[i,:,j], log = bicgstabl!(Tinv[i,:,j], T(z), R[:,j], 2; log=true, max_mv_products=1000, tol=1e-14)
                if !log.isconverged println(log) end
            end
            # println("--------------------------------------------")
            R .= X - Tinv[i,:,:]
            rmul!(R, (r*exp(θ[i]*im)/nodes) .* Diagonal(resolvent))
            Q₀ .+= R
            Q₁ .+= R .* z
        end

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
    c=complex(0.0,0.0), r=1.0, debug=false, ϵ=10e-12, moments=2)

    N, m₀ = size(X)
    Λ = zeros(ComplexF64, moments*m₀)
    θ = LinRange(π/nodes, 2*π-π/nodes, nodes)
    Q = zeros(ComplexF64, 2*moments, N, m₀)
    Y = zeros(ComplexF64, N, m₀*moments)
    Q₀, Q₁ = zeros(ComplexF64, moments*N, moments*m₀), zeros(ComplexF64, moments*N, moments*m₀)
    Tinv, Temp, R = similar(X, ComplexF64), similar(Y, ComplexF64), similar(Y, ComplexF64)
    resolvent = zeros(ComplexF64, m₀)
    A, B = zeros(ComplexF64, moments*m₀, moments*m₀), zeros(ComplexF64, moments*m₀, moments*m₀)
    res = Array{Float64}(undef, size(R, 2))
    Y = zeros(ComplexF64, N, m₀*moments)

    for i=1:nodes
        z = (r*exp(θ[i]*im)+c)
        Tinv .= (T(z)\X) .* (r*exp(θ[i]*im)/nodes)
        Q[1,:,:] .+= Tinv
        for j=1:2*moments
            Q[j,:,:] .+= Tinv .* z^(j-1)
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

    update_R!(Y, R, Λ, T)
    res .= residuals(R, Λ, T)
    p = sortperm(res)
    res .= res[p]
    X .= Y[:, p][:,1:m₀]
    Λ .= Λ[p]
    R .= R[:, p]

    if debug
        iter_debug_print(0, Λ[1:m₀], res[1:m₀], c, r)
    end

    for nit=1:iter

        Q .= 0

        for i=1:nodes
            z = (r*exp(θ[i]*im)+c)
            resolvent .= (1 ./(z .- Λ[1:m₀]))
            Tinv .= (X - T(z)\R[:, 1:m₀])
            rmul!(Tinv, (r*exp(θ[i]*im)/nodes) .* Diagonal(resolvent))
            Q[1,:,:] .+= Tinv
            for j=1:2*moments
                Q[j,:,:] .+= Tinv .* z^(j-1)
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
        # beyn_rr_step!(Q₀, Q₁, A, B, X, Λ)
        # beyn_qr_step!(Q₀, Q₁, X, Λ)

        update_R!(Y, R, Λ, T)
        res .= residuals(R, Λ, T)
        p = sortperm(res)
        res .= res[p]
        X .= Y[:, p][:,1:m₀]
        Λ .= Λ[p]
        R .= R[:, p]

        if debug
            iter_debug_print(nit, Λ[1:m₀], res[1:m₀], c, r)
        end

        if maximum(res[1:m₀][in_contour.(Λ[1:m₀], c, r)]) < ϵ
            break
        end
    end

    normalize!(X)
    Λ[1:m₀], X, res[1:m₀]
end

function iter_debug_print(nit, Λ, res, c, r)
    print(nit)
    print(":\t")
    print(sum(in_contour.(Λ, c, r)))
    print("\t")
    print(maximum(res[in_contour.(Λ, c, r)]))
    println()
end

function beyn_svd_step!(Q₀::AbstractMatrix{ComplexF64}, Q₁::AbstractMatrix{ComplexF64}, A::AbstractMatrix{ComplexF64}, B::AbstractMatrix{ComplexF64}, X::AbstractMatrix{ComplexF64}, Λ::Array{ComplexF64})
    S = svd!(Q₀)
    mul!(A, S.U', Q₁)
    mul!(B, A, S.V)
    mul!(A, B, Diagonal(1 ./ S.S))
    F = eigen!(A)
    mul!(X, S.U, F.vectors)
    Λ .= F.values
end

function beyn_qr_step!(Q₀::AbstractMatrix{ComplexF64}, Q₁::AbstractMatrix{ComplexF64}, X::AbstractMatrix{ComplexF64}, Λ::Array{ComplexF64})
    qt, rt = qr!(Q₀)
    qt = Matrix(qt)
    F = eigen!(qt' * Q₁ * inv(rt))
    mul!(X, qt, F.vectors)
    Λ .= F.values
end

function beyn_rr_step!(Q₀::AbstractMatrix{ComplexF64}, Q₁::AbstractMatrix{ComplexF64}, A::AbstractMatrix{ComplexF64}, B::AbstractMatrix{ComplexF64}, X::AbstractMatrix{ComplexF64}, Λ::Array{ComplexF64})
    mul!(A, X', Q₁)
    mul!(B, X', Q₀)
    F = eigen!(A, B)
    mul!(X, Q₀, F.vectors)
    Λ .= F.values
end


function update_R!(X::AbstractMatrix{ComplexF64}, R::AbstractMatrix{ComplexF64}, Λ::Array{ComplexF64}, T::Function)
    for i=1:size(X, 2)
        X[:,i] ./= norm(X[:,i])
        R[:,i] .= T(Λ[i]) * X[:,i]
    end
end

function normalize!(X::AbstractMatrix{ComplexF64})
    for i=1:size(X, 2)
        X[:,i] ./= norm(X[:,i])
    end
end

function residuals(R::AbstractMatrix{ComplexF64}, Λ::Array{ComplexF64}, T::Function)
    res = Array{Float64}(undef, size(R, 2))
    for i=1:size(R, 2)
        res[i] = norm(R[:,i])/norm(T(Λ[i]))
    end
    res
end
