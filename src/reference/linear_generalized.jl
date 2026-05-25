"""
    reference_gen_feast!(X, A, B; kwargs...)

Allocating reference implementation of dense generalized FEAST for
`Ax = λBx`.
"""
function reference_gen_feast!(X::AbstractMatrix, A::AbstractMatrix, B;
    nodes::Integer=8,
    iter::Integer=10,
    c=complex(0.0, 0.0),
    r=1.0,
    ϵ=1e-12,
    debug=false,
    contour=nothing,
)
    contour = contour === nothing ? circular_contour_trapezoidal(c, r, nodes) : contour
    reference_gen_feast!(X, A, B, contour; iter=iter, ϵ=ϵ, debug=debug)
end

function reference_gen_feast!(
    X::AbstractMatrix,
    A::AbstractMatrix,
    B,
    contour::Contour;
    iter::Integer=10,
    ϵ=1e-12,
    debug=false,
)
    size(A, 1) == size(A, 2) || error("Incorrect dimensions of A, must be square")
    size(A, 1) == size(X, 1) || error("Incorrect dimensions of X, must match A")
    B isa UniformScaling || size(A) == size(B) || error("Incorrect dimensions of A and B, must match")

    N, m₀ = size(X)
    Q = Matrix{ComplexF64}(X)
    Xritz = zeros(ComplexF64, N, m₀)
    R = zeros(ComplexF64, N, m₀)
    Λ = zeros(ComplexF64, m₀)
    res = zeros(Float64, m₀)
    inside = falses(m₀)

    for nit in 0:iter
        Q = Matrix(qr(Q).Q)[:, 1:m₀]
        AQ = A * Q
        BQ = B * Q
        F = LinearAlgebra.eigen(Q' * AQ, Q' * BQ)
        Λ .= F.values
        Xritz .= Q * F.vectors
        copyto!(X, Xritz)

        for j in axes(Xritz, 2)
            Xritz[:, j] ./= norm(view(Xritz, :, j))
        end
        BX = B * Xritz
        R .= A * Xritz .- BX * Diagonal(Λ)
        for j in eachindex(res)
            res[j] = norm(view(R, :, j))
        end

        in_contour!(inside, Λ, contour)
        max_res_inside, contour_nonempty = maximum_masked(res, inside)
        debug && iter_debug_print(nit, Λ, res, contour, 1e-5)
        contour_nonempty && max_res_inside < ϵ && break

        nit < iter || break
        Q .= 0
        for (z, weight) in zip(contour_nodes(contour), contour_weights(contour))
            shifted = Matrix{ComplexF64}(A)
            if B isa UniformScaling
                for i in 1:min(size(shifted)...)
                    shifted[i, i] -= z * B.λ
                end
            else
                shifted .-= z .* B
            end
            solved = shifted \ R
            for j in axes(Q, 2)
                Q[:, j] .+= (weight / (z - Λ[j])) .* (Xritz[:, j] .- solved[:, j])
            end
        end
    end

    inside = in_contour(Λ, contour)
    Λ[inside], Xritz[:, inside], res[inside]
end
