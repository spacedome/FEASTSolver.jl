struct CircularChart
    center::ComplexF64
    radius::Float64
    nodes::Vector{ComplexF64}
    weights::Vector{ComplexF64}
end

function CircularChart(center, radius, node_count::Integer)
    radius > 0 || throw(ArgumentError("radius must be positive"))
    node_count >= 4 || throw(ArgumentError("at least four contour nodes are required"))
    angles = ((2 .* (1:node_count) .- 1) .* pi) ./ node_count
    coordinates = ComplexF64[cis(angle) for angle in angles]
    radius = Float64(radius)
    CircularChart(
        ComplexF64(center),
        radius,
        ComplexF64(center) .+ radius .* coordinates,
        (radius / node_count) .* coordinates,
    )
end

struct ContactMeanField1D{M<:AbstractMatrix{Float64}}
    grid::Vector{Float64}
    spacing::Float64
    linear_hamiltonian::M
    coupling::Float64
    occupied::Int
end

function contact_mean_field_1d(;
    points=96,
    half_length=8.0,
    coupling=2.0,
    occupied=4,
    sparse=false,
)
    points > occupied >= 1 || throw(ArgumentError("points must exceed the occupied count"))
    half_length > 0 || throw(ArgumentError("half_length must be positive"))
    coupling >= 0 || throw(ArgumentError("coupling must be nonnegative"))
    spacing = 2half_length / (points + 1)
    grid = collect(range(-half_length + spacing; step=spacing, length=points))
    kinetic = if sparse
        spdiagm(
            -1 => fill(-1 / (2spacing^2), points - 1),
            0 => fill(1 / spacing^2, points),
            1 => fill(-1 / (2spacing^2), points - 1),
        )
    else
        matrix = zeros(Float64, points, points)
        for index in 1:points
            matrix[index, index] = 1 / spacing^2
            index > 1 && (matrix[index, index - 1] = -1 / (2spacing^2))
            index < points && (matrix[index, index + 1] = -1 / (2spacing^2))
        end
        matrix
    end
    potential = 0.5 .* grid .^ 2
    linear_hamiltonian = kinetic + spdiagm(0 => potential)
    !sparse && (linear_hamiltonian = Matrix(linear_hamiltonian))
    ContactMeanField1D(grid, spacing, linear_hamiltonian, Float64(coupling), Int(occupied))
end

function orthonormalize(vectors, count=size(vectors, 2))
    factorization = qr(Matrix{ComplexF64}(vectors))
    Matrix(factorization.Q)[:, 1:count]
end

function initial_orbitals(problem::ContactMeanField1D; seed=1729)
    rng = MersenneTwister(seed)
    envelope = exp.(-0.12 .* problem.grid .^ 2)
    trial = randn(rng, ComplexF64, length(problem.grid), problem.occupied)
    trial .*= envelope
    orthonormalize(trial, problem.occupied)
end

function density(problem::ContactMeanField1D, orbitals)
    size(orbitals, 1) == length(problem.grid) || throw(DimensionMismatch(
        "orbital height must equal the grid size",
    ))
    vec(sum(abs2, orbitals; dims=2)) ./ problem.spacing
end

function hamiltonian(problem::ContactMeanField1D, rho)
    length(rho) == length(problem.grid) || throw(DimensionMismatch(
        "density length must equal the grid size",
    ))
    result = problem.linear_hamiltonian + Diagonal(problem.coupling .* real.(rho))
    Hermitian(result)
end

function contact_energy(problem::ContactMeanField1D, orbitals)
    rho = density(problem, orbitals)
    kinetic_and_trap = real(tr(adjoint(orbitals) * problem.linear_hamiltonian * orbitals))
    kinetic_and_trap + 0.5 * problem.coupling * problem.spacing * sum(abs2, rho)
end

function invariant_residual(problem::ContactMeanField1D, rho, orbitals)
    H = hamiltonian(problem, rho)
    reduced = adjoint(orbitals) * H * orbitals
    residual = H * orbitals - orbitals * reduced
    norm(residual) / max(norm(H * orbitals), eps(Float64))
end

function subspace_gap(left, right)
    qleft = orthonormalize(left)
    qright = orthonormalize(right)
    norm(qleft * adjoint(qleft) - qright * adjoint(qright))
end

function reference_scf(
    problem::ContactMeanField1D,
    initial_density;
    mixing=0.5,
    tolerance=1e-13,
    iterations=1000,
)
    rho = Float64.(initial_density)
    orbitals = zeros(ComplexF64, length(rho), problem.occupied)
    values = Float64[]
    for iteration in 1:iterations
        decomposition = eigen(hamiltonian(problem, rho))
        values = decomposition.values[1:problem.occupied]
        orbitals = ComplexF64.(decomposition.vectors[:, 1:problem.occupied])
        output_density = density(problem, orbitals)
        defect = norm(output_density - rho) / max(norm(rho), eps(Float64))
        rho = (1 - mixing) .* rho .+ mixing .* output_density
        if defect <= tolerance
            return (
                orbitals=orbitals,
                density=rho,
                values=values,
                iterations=iteration,
                converged=true,
                residual=invariant_residual(problem, rho, orbitals),
                energy=contact_energy(problem, orbitals),
            )
        end
    end
    (
        orbitals=orbitals,
        density=rho,
        values=values,
        iterations=iterations,
        converged=false,
        residual=invariant_residual(problem, rho, orbitals),
        energy=contact_energy(problem, orbitals),
    )
end
