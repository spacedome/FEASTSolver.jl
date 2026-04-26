
function iter_debug_print(nit, Λ, res, c, r, spurious=1e-5)
    print(nit)
    print(":\t")
	in_eig = Λ[in_contour.(Λ, c, r)]
	in_res = res[in_contour.(Λ, c, r)]
    print(sum(in_contour.(Λ, c, r)))
    print(" (")
    print(sum(in_res .< spurious))
    print(")\t")
	if sum(in_contour.(Λ, c, r)) > 0
		print(maximum(res[in_contour.(Λ, c, r)]))
		in_res_conv = in_res[in_res .< spurious]
		if size(in_res_conv, 1) > 0
			print("\t(")
			print(maximum(in_res_conv))
			print(")")
		end
	end
    println()
end

function iter_debug_print(nit, Λ, res, contour::Contour, spurious=1e-5)
    print(nit)
    print(":\t")
	in_eig = Λ[in_contour(Λ, contour)]
	in_res = res[in_contour(Λ, contour)]
    print(sum(in_contour(Λ, contour)))
    print(" (")
    print(sum(in_res .< spurious))
    print(")\t")
	if sum(in_contour(Λ, contour)) > 0
		print(maximum(res[in_contour(Λ, contour)]))
		in_res_conv = in_res[in_res .< spurious]
		if size(in_res_conv, 1) > 0
			print("\t(")
			print(maximum(in_res_conv))
			print(")")
		end
	end
    println()
end

function convergence_info(Λ, X, residuals, contour::Contour, spurious=1e-3)
	in_ind = in_contour(Λ, contour)
    in_eig = Λ[in_ind] # eigenvalues inside contour
    in_res = residuals[in_ind] # residuals of eigenvalues inside contour
    print("Number of eigenvalues inside contour: ")
    println(size(in_eig, 1))
    if sum(in_ind) > 0
        in_res_conv = in_res[in_res .<= spurious]
        in_eig_conv = in_eig[in_res .<= spurious]
        print("Number inside converged : ")
        println(size(in_eig_conv, 1))
        print("Max res inside: ")
        println(maximum(in_res))
        if size(in_res_conv, 1) > 0
            print("Max res inside non spurious: ")
            println(maximum(in_res_conv))
        end
    end
end

function convergence_info(Λ, X, residuals, c, r, spurious=1e-3)
	contour = circular_contour_trapezoidal(c, r, 4)
	convergence_info(Λ, X, residuals, contour, spurious)
end

function beyn_svd_step!(Q₀::AbstractMatrix, Q₁::AbstractMatrix, A::AbstractMatrix, B::AbstractMatrix, X::AbstractMatrix{ComplexF64}, Λ::Array)
    S = svd!(Q₀)
    mul!(A, S.U', Q₁)
    mul!(B, A, S.V)
    mul!(A, B, Diagonal(1 ./ S.S))
    F = eigen!(A)
    mul!(X, S.U, F.vectors)
    Λ .= F.values
end

function beyn_svd_step!(
    Q₀::AbstractMatrix,
    Q₁::AbstractMatrix,
    A::AbstractMatrix,
    B::AbstractMatrix,
    X::AbstractMatrix{ComplexF64},
    Λ::Array,
    svd_ws::SVDsddWs,
    eigen_ws::EigenWs,
    Xq::AbstractMatrix,
)
    U, S, Vt = dense_lapack_svd!(Q₀, svd_ws, 'S')
    mul!(A, adjoint(U), Q₁)
    mul!(B, A, adjoint(Vt))
    copyto!(A, B)
    inv_scale_columns!(A, S)
    dense_lapack_eigen!(Λ, Xq, A, eigen_ws)
    mul!(X, U, Xq)
    Λ
end

function add_weighted_columns!(Y::AbstractMatrix, X::AbstractMatrix, α)
    @inbounds for j in axes(Y, 2), i in axes(Y, 1)
        Y[i, j] += α * X[i, j]
    end
    Y
end

function beyn_qr_step!(Q₀::AbstractMatrix, Q₁::AbstractMatrix, X::AbstractMatrix, Λ::Array)
    qt, rt = qr!(Q₀)
    qt = Matrix(qt)
    F = eigen!(qt' * Q₁ * inv(rt))
    mul!(X, qt, F.vectors)
    Λ .= F.values
end

function beyn_rr_step!(Q₀::AbstractMatrix, Q₁::AbstractMatrix, A::AbstractMatrix, B::AbstractMatrix, X::AbstractMatrix{ComplexF64}, Λ::Array)
    mul!(A, X', Q₁)
    mul!(B, X', Q₀)
    F = eigen!(A, B)
    mul!(X, Q₀, F.vectors)
    Λ .= F.values
end

function beyn_rr_step2!(Q₀::AbstractMatrix, Q₁::AbstractMatrix, A::AbstractMatrix, B::AbstractMatrix, X::AbstractMatrix{ComplexF64}, Λ::Array)
    mul!(A, Q₀', Q₁)
    mul!(B, Q₀', Q₀)
    F = eigen!(A, B)
    mul!(X, Q₀, F.vectors)
    Λ .= F.values
end


function column_norm(A::AbstractMatrix, j::Integer)
    s = zero(real(eltype(A)))
    @inbounds for i in axes(A, 1)
        s += abs2(A[i, j])
    end
    sqrt(s)
end

function fill_resolvent!(resolvent::AbstractVector, z, Λ::AbstractVector)
    @inbounds for j in eachindex(Λ)
        resolvent[j] = inv(z - Λ[j])
    end
    resolvent
end

function fill_adjoint_resolvent!(resolvent::AbstractVector, z, Λ::AbstractVector)
    @inbounds for j in eachindex(Λ)
        resolvent[j] = inv(conj(z - Λ[j]))
    end
    resolvent
end

function maximum_masked(values::AbstractVector, mask::AbstractVector{Bool})
    max_value = typemin(float(real(eltype(values))))
    found = false
    @inbounds for i in eachindex(values, mask)
        if mask[i]
            max_value = found ? max(max_value, values[i]) : values[i]
            found = true
        end
    end
    max_value, found
end

function maximum_below_masked(values::AbstractVector, mask::AbstractVector{Bool}, threshold)
    max_value = typemin(float(real(eltype(values))))
    found = false
    @inbounds for i in eachindex(values, mask)
        if mask[i] && values[i] < threshold
            max_value = found ? max(max_value, values[i]) : values[i]
            found = true
        end
    end
    max_value, found
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

finalize!(x::Any) = nothing

function linsolve!(Y, C, X, factorizer, left_divider)
    F = factorizer(C)
    left_divider(Y, F, X)
    finalize!(F)
end
