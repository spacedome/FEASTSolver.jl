function column_norm(A::AbstractMatrix, j::Integer)
    s = zero(real(eltype(A)))
    @inbounds for i in axes(A, 1)
        s += abs2(A[i, j])
    end
    sqrt(s)
end

function normalize_columns!(X::AbstractMatrix)
    @inbounds for j in axes(X, 2)
        α = inv(column_norm(X, j))
        for i in axes(X, 1)
            X[i, j] *= α
        end
    end
    X
end

function update_R!(X::AbstractMatrix, R::AbstractMatrix, Λ::Array, T)
    normalize_columns!(X)
    for j in axes(X, 2)
        R[:, j] .= T(Λ[j]) * X[:, j]
    end
end

function update_nonlinear_residuals!(
    res::AbstractVector,
    X::AbstractMatrix,
    R::AbstractMatrix,
    Λ::AbstractVector,
    T,
    Tλ::AbstractMatrix,
    x::AbstractVector,
    y::AbstractVector,
)
    @inbounds for j in axes(X, 2)
        xnorm = zero(real(eltype(X)))
        for i in axes(X, 1)
            xnorm += abs2(X[i, j])
        end
        inv_xnorm = inv(sqrt(xnorm))
        for i in axes(X, 1)
            x[i] = X[i, j] * inv_xnorm
            X[i, j] = x[i]
        end
        copyto!(Tλ, T(Λ[j]))
        mul!(y, Tλ, x)
        for i in axes(R, 1)
            R[i, j] = y[i]
        end
        res[j] = norm(y) / norm(Tλ)
    end
    res
end

function update_nonlinear_residuals!(
    res::AbstractVector,
    X::AbstractMatrix,
    R::AbstractMatrix,
    Λ::AbstractVector,
    T,
    x::AbstractVector,
    y::AbstractVector,
)
    @inbounds for j in axes(X, 2)
        xnorm = zero(real(eltype(X)))
        for i in axes(X, 1)
            xnorm += abs2(X[i, j])
        end
        inv_xnorm = inv(sqrt(xnorm))
        for i in axes(X, 1)
            x[i] = X[i, j] * inv_xnorm
            X[i, j] = x[i]
        end
        Tλ = T(Λ[j])
        mul!(y, Tλ, x)
        for i in axes(R, 1)
            R[i, j] = y[i]
        end
        res[j] = norm(y) / norm(Tλ)
    end
    res
end

function update_R!(X::AbstractMatrix, R::AbstractMatrix, Λ::Array, A::AbstractMatrix)
    normalize_columns!(X)
    mul!(R, A, X)
    @inbounds for j in axes(X, 2)
        λ = Λ[j]
        for i in axes(X, 1)
            R[i, j] -= λ * X[i, j]
        end
    end
    R
end

function update_R!(X::AbstractMatrix, R::AbstractMatrix, Λ::Array, A::AbstractMatrix, B::UniformScaling)
    normalize_columns!(X)
    mul!(R, A, X)
    @inbounds for j in axes(X, 2)
        λ = Λ[j] * B.λ
        for i in axes(X, 1)
            R[i, j] -= λ * X[i, j]
        end
    end
    R
end

function update_R!(
    X::AbstractMatrix,
    R::AbstractMatrix,
    Λ::Array,
    A::AbstractMatrix,
    B::UniformScaling,
    BX::AbstractMatrix,
)
    update_R!(X, R, Λ, A, B)
end

function update_R!(X::AbstractMatrix, R::AbstractMatrix, Λ::Array, A::AbstractMatrix, B::AbstractMatrix, BX::AbstractMatrix)
    normalize_columns!(X)
    mul!(R, A, X)
    mul!(BX, B, X)
    @inbounds for j in axes(X, 2)
        λ = Λ[j]
        for i in axes(X, 1)
            R[i, j] -= λ * BX[i, j]
        end
    end
    R
end

function update_R_allocating!(X::AbstractMatrix, R::AbstractMatrix, Λ::Array, A::AbstractMatrix, B)
    for i=1:size(X, 2)
        X[:,i] ./= norm(X[:,i])
        R[:,i] .= (A - Λ[i]*B) * X[:,i]
    end
    R
end

function update_R_shifted!(
    X::AbstractMatrix,
    R::AbstractMatrix,
    Λ::Array,
    A::AbstractMatrix,
    B,
    C::AbstractMatrix,
    x::AbstractVector,
    y::AbstractVector,
)
    for j in axes(X, 2)
        copyto!(x, view(X, :, j))
        x ./= norm(x)
        X[:, j] .= x
        materialize_generalized_shift!(C, A, B, Λ[j])
        mul!(y, C, x)
        R[:, j] .= y
    end
    R
end

function update_R!(X::AbstractMatrix, R::AbstractMatrix, Λ::Array, A::AbstractMatrix, B::AbstractMatrix)
    normalize_columns!(X)
    mul!(R, A, X)
    @inbounds for j in axes(X, 2)
        xj = view(X, :, j)
        bxj = B * xj
        λ = Λ[j]
        for i in axes(X, 1)
            R[i, j] -= λ * bxj[i]
        end
    end
    R
end

function update_R_moments!(X::AbstractMatrix, R::AbstractMatrix, Λ::Array, res::Array, T::Function, c, r)
    for i=1:size(X, 2)
        X[:,i] ./= norm(X[:,i])
        R[:,i] .= T(Λ[i]) * X[:,i]
    end
	residuals!(res, R, Λ, T)
	# p = sortperm(res .- in_contour.(Λ, c, r))
	p = sortperm(res)
	# p = sortperm(abs.(Λ .- c))
	# a = copy(abs.(Λ .- c))
	# b = copy(res)
	# p = sortperm(normalize!(a) .+ normalize!(b))
	res .= res[p]
	X .= X[:, p]
	Λ .= Λ[p]
	R .= R[:, p]
end

function update_R_moments_all!(X::AbstractMatrix, R::AbstractMatrix, Λ::Array, res::Array, T::Function, c, r)
    for i=1:size(X, 2)
        X[:,i] ./= norm(X[:,i])
        R[:,i] .= T(Λ[i]) * X[:,i]
    end
	residuals!(res, R, Λ, T)
end

function normalize!(X::AbstractVecOrMat)
    normalize_columns!(X)
end

function residuals(R::AbstractMatrix, Λ::Array, T::Function)
    res = Array{Float64}(undef, size(Λ, 1))
    for i=1:size(Λ, 1)
        res[i] = norm(R[:,i])/norm(T(Λ[i]))
    end
    res
end

function residuals!(res::Array, R::AbstractMatrix, Λ::Array, T::Function)
    for i=1:size(Λ, 1)
        res[i] = norm(R[:,i])/norm(T(Λ[i]))
    end
    res
end

function residuals!(res::Array, R::AbstractMatrix, Λ::Array, A::AbstractMatrix)
    for i in eachindex(Λ)
        res[i] = column_norm(R, i)
    end
    res
end
