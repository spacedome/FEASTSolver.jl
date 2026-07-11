struct SimilarityMeanField1D
    base::ContactMeanField1D
    similarity::Vector{Float64}
end

function similarity_mean_field_1d(; nonnormality=0.35, kwargs...)
    base = contact_mean_field_1d(; kwargs...)
    SimilarityMeanField1D(base, exp.(nonnormality .* base.grid))
end

function hamiltonian(problem::SimilarityMeanField1D, rho)
    diagonal = Diagonal(problem.similarity)
    inverse_diagonal = Diagonal(1.0 ./ problem.similarity)
    Matrix{ComplexF64}(diagonal * Matrix(hamiltonian(problem.base, rho)) * inverse_diagonal)
end

function initial_biorthogonal_orbitals(problem::SimilarityMeanField1D; seed=1729)
    basis = initial_orbitals(problem.base; seed=seed)
    right = problem.similarity .* basis
    left = (1.0 ./ problem.similarity) .* basis
    right, left
end

function oblique_density(problem::SimilarityMeanField1D, right, left)
    overlap = adjoint(left) * right
    projector = right * (overlap \ adjoint(left))
    real.(diag(projector)) ./ problem.base.spacing
end

function biorthogonalize(right, left, count=size(right, 2); ranktol=1e-12)
    overlap = adjoint(left) * right
    factorization = svd(overlap)
    factorization.S[count] >= ranktol * factorization.S[1] || error(
        "right/left filtered spaces have deficient overlap",
    )
    inverse_root = Diagonal(1.0 ./ sqrt.(factorization.S[1:count]))
    right_basis = right * factorization.V[:, 1:count] * inverse_root
    left_basis = left * factorization.U[:, 1:count] * inverse_root
    right_basis, left_basis
end

function dual_projector_step(H, chart::CircularChart, right, left; ranktol=1e-12)
    size(right) == size(left) || throw(DimensionMismatch(
        "right and left probes must have equal dimensions",
    ))
    dimension, state_count = size(right)
    identity_matrix = Matrix{ComplexF64}(I, dimension, dimension)
    right_filtered = zeros(ComplexF64, dimension, state_count)
    left_filtered = zeros(ComplexF64, dimension, state_count)
    for (z, weight) in zip(chart.nodes, chart.weights)
        factorization = lu(z .* identity_matrix .- H)
        right_filtered .+= weight .* (factorization \ right)
        left_filtered .+= conj(weight) .* (adjoint(factorization) \ left)
    end
    right_basis, left_basis = biorthogonalize(
        right_filtered,
        left_filtered,
        state_count;
        ranktol=ranktol,
    )
    state = adjoint(left_basis) * H * right_basis
    values = eigvals(state)
    count_inside = count(value -> abs(value - chart.center) < chart.radius, values)
    count_inside == state_count || error(
        "dual FEAST found $count_inside states inside the contour, expected $state_count",
    )
    (
        right=right_basis,
        left=left_basis,
        state=state,
        values=values,
        overlap_singular_values=svdvals(adjoint(left_filtered) * right_filtered),
        solve_count=length(chart.nodes),
        rhs_count=2length(chart.nodes) * state_count,
    )
end

function dual_invariant_residual(problem::SimilarityMeanField1D, rho, right, left)
    H = hamiltonian(problem, rho)
    overlap = adjoint(left) * right
    state = overlap \ (adjoint(left) * H * right)
    right_residual = H * right - right * state
    left_residual = adjoint(H) * left - left * adjoint(state)
    max(
        norm(right_residual) / max(norm(H * right), eps(Float64)),
        norm(left_residual) / max(norm(adjoint(H) * left), eps(Float64)),
    )
end

Base.@kwdef struct DualSCFConfig
    iterations::Int = 100
    mixing::Float64 = 0.3
    density_tolerance::Float64 = 1e-10
    residual_tolerance::Float64 = 1e-10
    ranktol::Float64 = 1e-12
end

function solve_dual_scf(
    problem::SimilarityMeanField1D,
    chart::CircularChart,
    initial_right,
    initial_left;
    config::DualSCFConfig=DualSCFConfig(),
)
    0 < config.mixing <= 1 || throw(ArgumentError("mixing must lie in (0,1]"))
    right, left = biorthogonalize(initial_right, initial_left)
    rho = oblique_density(problem, right, left)
    history = NamedTuple[]
    converged = false

    for iteration in 1:config.iterations
        step = dual_projector_step(
            hamiltonian(problem, rho),
            chart,
            right,
            left;
            ranktol=config.ranktol,
        )
        output_density = oblique_density(problem, step.right, step.left)
        density_defect = norm(output_density - rho) / max(norm(rho), eps(Float64))
        next_density = (1 - config.mixing) .* rho .+ config.mixing .* output_density
        residual = dual_invariant_residual(
            problem,
            next_density,
            step.right,
            step.left,
        )
        push!(history, (
            iteration=iteration,
            density_defect=density_defect,
            residual=residual,
            overlap_ratio=step.overlap_singular_values[end] /
                step.overlap_singular_values[1],
            solve_count=step.solve_count,
            rhs_count=step.rhs_count,
        ))
        right = step.right
        left = step.left
        rho = next_density
        if density_defect <= config.density_tolerance && residual <= config.residual_tolerance
            converged = true
            break
        end
    end

    (
        right=right,
        left=left,
        density=rho,
        history=history,
        converged=converged,
        residual=dual_invariant_residual(problem, rho, right, left),
        solve_count=sum(record.solve_count for record in history),
        rhs_count=sum(record.rhs_count for record in history),
    )
end
