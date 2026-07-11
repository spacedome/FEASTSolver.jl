function state_modal_data(T, realization)
    decomposition = eigen(realization.state)
    right = realization.right * decomposition.vectors
    left = realization.left * adjoint(inv(decomposition.vectors))
    residual = maximum(vcat([
        norm(T(decomposition.values[j]) * view(right, :, j)) / norm(view(right, :, j))
        for j in eachindex(decomposition.values)
    ], [
        norm(adjoint(T(decomposition.values[j])) * view(left, :, j)) / norm(view(left, :, j))
        for j in eachindex(decomposition.values)
    ]))
    (values=ComplexF64.(decomposition.values), right=right, left=left, residual=residual)
end

function realization_subspace_gap(actual, expected)
    actual_rank = rank(actual; rtol=1e-10)
    expected_rank = rank(expected; rtol=1e-10)
    actual_rank == expected_rank || return Inf
    actual_basis = Matrix(qr(actual).Q)[:, 1:actual_rank]
    expected_basis = Matrix(qr(expected).Q)[:, 1:expected_rank]
    opnorm(expected_basis - actual_basis * (adjoint(actual_basis) * expected_basis))
end
