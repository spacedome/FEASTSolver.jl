"""
    reference_nlfeast!(T, X, nodes, iter; kwargs...)

Allocating reference implementation of the canonical nonlinear FEAST/Beyn
hybrid. This keeps the Beyn moment extraction and residual inverse iteration
steps visible, and intentionally uses ordinary Julia linear algebra.
"""
function reference_nlfeast!(T, X::AbstractMatrix{ComplexF64}, nodes::Integer, iter::Integer;
    c=complex(0.0, 0.0),
    r=1.0,
    ϵ=10e-12,
    debug=false,
    spurious=1e-5,
    contour=nothing,
)
    contour = contour === nothing ? circular_contour_trapezoidal(c, r, nodes) : contour
    reference_nlfeast!(T, X, contour, iter; ϵ=ϵ, debug=debug, spurious=spurious)
end

function reference_nlfeast!(
    T,
    X::AbstractMatrix{ComplexF64},
    contour::Contour,
    iter::Integer;
    ϵ=10e-12,
    debug=false,
    spurious=1e-5,
)
    N, m₀ = size(X)
    Q₀ = zeros(ComplexF64, N, m₀)
    Q₁ = zeros(ComplexF64, N, m₀)
    R = zeros(ComplexF64, N, m₀)
    Λ = zeros(ComplexF64, m₀)
    res = zeros(Float64, m₀)
    inside = falses(m₀)

    Q = Matrix(qr(X).Q)[:, 1:m₀]
    copyto!(X, Q)

    for nit in 0:iter
        Q₀ .= 0
        Q₁ .= 0

        for (z, weight) in zip(contour_nodes(contour), contour_weights(contour))
            if nit == 0
                solved = T(z) \ X
                Q₀ .+= weight .* solved
                Q₁ .+= (weight * z) .* solved
            else
                solved = T(z) \ R
                for j in axes(X, 2)
                    update = (weight / (z - Λ[j])) .* (X[:, j] .- solved[:, j])
                    Q₀[:, j] .+= update
                    Q₁[:, j] .+= z .* update
                end
            end
        end

        S = LinearAlgebra.svd(Q₀)
        reduced = S.U' * Q₁ * S.V * Diagonal(1 ./ S.S)
        F = LinearAlgebra.eigen(reduced)
        Λ .= F.values
        X .= S.U * F.vectors

        for j in axes(X, 2)
            X[:, j] ./= norm(view(X, :, j))
            Tλ = T(Λ[j])
            R[:, j] .= Tλ * X[:, j]
            res[j] = norm(view(R, :, j)) / norm(Tλ)
        end

        in_contour!(inside, Λ, contour)
        max_res_inside, contour_nonempty = maximum_masked(res, inside)
        debug && iter_debug_print(nit, Λ, res, contour, spurious)
        contour_nonempty && max_res_inside < ϵ && break

        max_spurious_res_inside, spurious_found = maximum_below_masked(res, inside, spurious)
        nit > 1 && spurious_found && max_spurious_res_inside < ϵ && break
    end

    normalize!(X)
    Λ, X, res
end
