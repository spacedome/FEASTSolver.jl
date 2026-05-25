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

finalize!(x::Any) = nothing

function linsolve!(Y, C, X, factorizer, left_divider)
    F = factorizer(C)
    left_divider(Y, F, X)
    finalize!(F)
end
