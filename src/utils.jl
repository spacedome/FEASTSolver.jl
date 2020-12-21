
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


function update_R!(X::AbstractMatrix, R::AbstractMatrix, Λ::Array, T::Function)
    for i=1:size(X, 2)
        X[:,i] ./= norm(X[:,i])
        R[:,i] .= T(Λ[i]) * X[:,i]
    end
end

function update_R!(X::AbstractMatrix, R::AbstractMatrix, Λ::Array, A::AbstractMatrix, B=I)
    for i=1:size(X, 2)
        X[:,i] ./= norm(X[:,i])
        R[:,i] .= (A - Λ[i]*B) * X[:,i]
    end
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
    for i=1:size(X, 2)
        X[:,i] ./= norm(X[:,i])
    end
	X
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
    for i=1:size(Λ, 1)
        res[i] = norm(R[:,i])
    end
    res
end

finalize!(x::Any) = nothing

function linsolve!(Y, C, X, factorizer, left_divider)
    F = factorizer(C)
    left_divider(Y, F, X)
    finalize!(F)
end
