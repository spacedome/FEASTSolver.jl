struct NonlocalProjector1D{M<:AbstractMatrix{Float64},K<:AbstractMatrix{Float64}}
    grid::Vector{Float64}
    spacing::Float64
    linear_hamiltonian::M
    kernel::K
    coupling::Float64
    occupied::Int
end

function nonlocal_projector_1d(;
    points=48,
    half_length=7.0,
    coupling=5.0,
    occupied=3,
    kernel_width=1.25,
)
    coupling >= 0 || throw(ArgumentError("coupling must be nonnegative"))
    kernel_width > 0 || throw(ArgumentError("kernel_width must be positive"))
    base = contact_mean_field_1d(
        points=points,
        half_length=half_length,
        coupling=0.0,
        occupied=occupied,
    )
    kernel = [
        exp(-0.5 * ((left - right) / kernel_width)^2) * base.spacing
        for left in base.grid, right in base.grid
    ]
    kernel ./= opnorm(kernel)
    NonlocalProjector1D(
        base.grid,
        base.spacing,
        base.linear_hamiltonian,
        kernel,
        Float64(coupling),
        Int(occupied),
    )
end

function initial_orbitals(problem::NonlocalProjector1D; seed=1729)
    base = ContactMeanField1D(
        problem.grid,
        problem.spacing,
        problem.linear_hamiltonian,
        0.0,
        problem.occupied,
    )
    initial_orbitals(base; seed=seed)
end

projector_state(orbitals) = orbitals * adjoint(orbitals)

function hamiltonian(problem::NonlocalProjector1D, projector)
    size(projector) == (length(problem.grid), length(problem.grid)) || throw(
        DimensionMismatch("projector state has the wrong dimensions"),
    )
    interaction = problem.kernel * projector * problem.kernel
    Hermitian(problem.linear_hamiltonian + problem.coupling .* interaction)
end

function projector_invariant_residual(problem::NonlocalProjector1D, projector, orbitals)
    H = hamiltonian(problem, projector)
    reduced = adjoint(orbitals) * H * orbitals
    residual = H * orbitals - orbitals * reduced
    norm(residual) / max(norm(H * orbitals), eps(Float64))
end

function reference_projector_scf(
    problem::NonlocalProjector1D,
    initial_projector;
    mixing=0.2,
    tolerance=1e-12,
    iterations=2000,
)
    0 < mixing <= 1 || throw(ArgumentError("mixing must lie in (0,1]"))
    projector = Matrix{ComplexF64}(initial_projector)
    orbitals = zeros(ComplexF64, length(problem.grid), problem.occupied)
    values = Float64[]
    for iteration in 1:iterations
        decomposition = eigen(hamiltonian(problem, projector))
        values = Float64.(decomposition.values[1:problem.occupied])
        orbitals = ComplexF64.(decomposition.vectors[:, 1:problem.occupied])
        output = projector_state(orbitals)
        defect = norm(output - projector) / max(norm(projector), eps(Float64))
        projector = (1 - mixing) .* projector .+ mixing .* output
        if defect <= tolerance
            return (
                orbitals=orbitals,
                projector=projector,
                values=values,
                iterations=iteration,
                converged=true,
                residual=projector_invariant_residual(problem, projector, orbitals),
            )
        end
    end
    (
        orbitals=orbitals,
        projector=projector,
        values=values,
        iterations=iterations,
        converged=false,
        residual=projector_invariant_residual(problem, projector, orbitals),
    )
end

function projector_ritz_state(problem::NonlocalProjector1D, basis, projector)
    H = hamiltonian(problem, projector)
    reduced_hamiltonian = Hermitian(adjoint(basis) * H * basis)
    decomposition = eigen(reduced_hamiltonian)
    occupied = 1:problem.occupied
    orbitals = basis * decomposition.vectors[:, occupied]
    output_projector = projector_state(orbitals)
    values = Float64.(decomposition.values[occupied])
    residual = H * orbitals - orbitals * Diagonal(values)
    (
        orbitals=orbitals,
        values=values,
        output_projector=output_projector,
        leakage=norm(residual) / max(norm(H * orbitals), eps(Float64)),
        hamiltonian=H,
    )
end

function projector_anderson_correction(states, residuals, mixing, regularization)
    state_vectors = [vec(state) for state in states]
    residual_vectors = [vec(residual) for residual in residuals]
    vector = anderson_correction(
        state_vectors,
        residual_vectors,
        mixing,
        regularization,
    )
    shape = size(states[end])
    correction = reshape(vector, shape)
    Matrix(Hermitian((correction + adjoint(correction)) / 2))
end

function normalize_projector_trace(problem::NonlocalProjector1D, projector)
    result = Matrix(Hermitian((projector + adjoint(projector)) / 2))
    trace_error = problem.occupied - real(tr(result))
    result[diagind(result)] .+= trace_error / size(result, 1)
    result
end
