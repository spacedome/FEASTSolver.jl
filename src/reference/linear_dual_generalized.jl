"""
    reference_dual_gen_feast!(Xr, Xl, A, B; kwargs...)

Allocating reference implementation of dense dual generalized FEAST. This keeps
the same visible algorithmic stages as the optimized implementation while using
ordinary Julia linear algebra.
"""
function reference_dual_gen_feast!(Xr::AbstractMatrix, Xl::AbstractMatrix, A::AbstractMatrix, B;
    nodes::Integer=8,
    iter::Integer=10,
    c=complex(0.0, 0.0),
    r=1.0,
    ϵ=1e-12,
    debug=false,
    contour=nothing,
)
    contour = contour === nothing ? circular_contour_trapezoidal(c, r, nodes) : contour
    reference_dual_gen_feast!(Xr, Xl, A, B, contour; iter=iter, ϵ=ϵ, debug=debug)
end

function reference_dual_gen_feast!(
    Xr::AbstractMatrix,
    Xl::AbstractMatrix,
    A::AbstractMatrix,
    B,
    contour::Contour;
    iter::Integer=10,
    ϵ=1e-12,
    debug=false,
)
    size(A, 1) == size(A, 2) || error("Incorrect dimensions of A, must be square")
    size(A, 1) == size(Xr, 1) || error("Incorrect dimensions of X, must match A")
    B isa UniformScaling || size(A) == size(B) || error("Incorrect dimensions of A and B, must match")
    size(Xr) == size(Xl) || error("Incorrect dimensions of Xr and Xl, must match")

    N, m₀ = size(Xr)
    Qr = Matrix{ComplexF64}(Xr)
    Ql = Matrix{ComplexF64}(Xl)
    Xright = zeros(ComplexF64, N, m₀)
    Xleft = zeros(ComplexF64, N, m₀)
    Rr = zeros(ComplexF64, N, m₀)
    Rl = zeros(ComplexF64, N, m₀)
    Λ = zeros(ComplexF64, m₀)
    res = zeros(Float64, m₀)
    inside = falses(m₀)

    for nit in 0:iter
        BQr = B * Qr
        U, S, Vt = LinearAlgebra.svd(Ql' * BQr)
        Qr = Qr * adjoint(Vt)
        Ql = Ql * U
        for j in eachindex(S)
            Qr[:, j] ./= S[j]
            Ql[:, j] ./= S[j]
        end

        AQr = A * Qr
        BQr = B * Qr
        Ared = Ql' * AQr
        Bred = Ql' * BQr
        Fr = LinearAlgebra.eigen(Ared, Bred)
        Fl = LinearAlgebra.eigen(Ared', Bred')
        Λ .= Fr.values
        Xqr = Fr.vectors
        selected = falses(length(Fl.values))
        Xql = similar(Fl.vectors, size(Fl.vectors, 1), length(Λ))
        for (j, λ) in pairs(Λ)
            scores = abs.(conj(λ) .- Fl.values) .+ (selected .* Inf)
            index = argmin(scores)
            selected[index] = true
            Xql[:, j] .= Fl.vectors[:, index]
        end
        Xright .= Qr * Xqr
        Xleft .= Ql * Xql
        copyto!(Xr, Xright)
        copyto!(Xl, Xleft)

        for j in axes(Xright, 2)
            Xright[:, j] ./= norm(view(Xright, :, j))
            Xleft[:, j] ./= norm(view(Xleft, :, j))
        end
        Rr .= A * Xright .- (B * Xright) * Diagonal(Λ)
        Rl .= A' * Xleft .- (B' * Xleft) * Diagonal(Λ)
        for j in eachindex(res)
            res[j] = norm(view(Rr, :, j))
        end

        in_contour!(inside, Λ, contour)
        max_res_inside, contour_nonempty = maximum_masked(res, inside)
        debug && iter_debug_print(nit, Λ, res, contour, 1e-5)
        contour_nonempty && max_res_inside < ϵ && break

        nit < iter || break
        Qr .= 0
        Ql .= 0
        for (z, weight) in zip(contour_nodes(contour), contour_weights(contour))
            shifted = Matrix{ComplexF64}(A)
            if B isa UniformScaling
                for i in 1:min(size(shifted)...)
                    shifted[i, i] -= z * B.λ
                end
            else
                shifted .-= z .* B
            end
            right_solve = shifted \ Rr
            left_solve = shifted' \ Rl
            for j in axes(Qr, 2)
                Qr[:, j] .+= (weight / (z - Λ[j])) .* (Xright[:, j] .- right_solve[:, j])
                Ql[:, j] .+= conj(weight / (z - Λ[j])) .* (Xleft[:, j] .- left_solve[:, j])
            end
        end
    end

    inside = in_contour(Λ, contour)
    Λ[inside], Xright[:, inside], Xleft[:, inside], res[inside]
end
