function contour_density_response(
    problem::ContactMeanField1D,
    orbitals,
    values,
    direction,
    chart::CircularChart;
    factorizations,
)
    length(direction) == length(problem.grid) || throw(DimensionMismatch(
        "density direction length must equal the grid size",
    ))
    length(values) == size(orbitals, 2) || throw(DimensionMismatch(
        "one occupied value is required per orbital",
    ))
    length(factorizations) == length(chart.nodes) || throw(DimensionMismatch(
        "one factorization is required per contour node",
    ))
    forcing = problem.coupling .* direction .* orbitals
    forcing .-= orbitals * (adjoint(orbitals) * forcing)
    response = zeros(ComplexF64, size(orbitals))
    for (z, weight, factorization) in zip(chart.nodes, chart.weights, factorizations)
        solved = factorization \ forcing
        solved ./= reshape(z .- values, 1, :)
        response .+= weight .* solved
    end
    response .-= orbitals * (adjoint(orbitals) * response)
    vec(2 .* real.(sum(conj.(orbitals) .* response; dims=2))) ./ problem.spacing
end

function gmres_action(action, right_hand_side; tolerance, iterations)
    dimension = length(right_hand_side)
    initial_norm = norm(right_hand_side)
    initial_norm == 0 && return (solution=zeros(Float64, dimension), iterations=0, residual=0.0)
    basis = zeros(Float64, dimension, iterations + 1)
    hessenberg = zeros(Float64, iterations + 1, iterations)
    target = zeros(Float64, iterations + 1)
    target[1] = initial_norm
    basis[:, 1] .= right_hand_side ./ initial_norm
    solution = zeros(Float64, dimension)
    relative_residual = 1.0

    for column in 1:iterations
        vector = Float64.(action(view(basis, :, column)))
        for row in 1:column
            hessenberg[row, column] = dot(view(basis, :, row), vector)
            vector .-= hessenberg[row, column] .* view(basis, :, row)
        end
        hessenberg[column + 1, column] = norm(vector)
        if hessenberg[column + 1, column] > eps(Float64)
            basis[:, column + 1] .= vector ./ hessenberg[column + 1, column]
        end
        reduced = hessenberg[1:(column + 1), 1:column]
        coefficients = reduced \ target[1:(column + 1)]
        solution .= basis[:, 1:column] * coefficients
        relative_residual = norm(target[1:(column + 1)] - reduced * coefficients) / initial_norm
        relative_residual <= tolerance && return (
            solution=solution,
            iterations=column,
            residual=relative_residual,
        )
    end
    (solution=solution, iterations=iterations, residual=relative_residual)
end

function cg_action(action, right_hand_side; tolerance, iterations)
    dimension = length(right_hand_side)
    initial_norm = norm(right_hand_side)
    initial_norm == 0 && return (solution=zeros(Float64, dimension), iterations=0, residual=0.0)
    solution = zeros(Float64, dimension)
    residual_vector = Float64.(right_hand_side)
    direction = copy(residual_vector)
    residual_squared = dot(residual_vector, residual_vector)
    relative_residual = 1.0

    for iteration in 1:iterations
        action_direction = Float64.(action(direction))
        curvature = dot(direction, action_direction)
        curvature > 0 || error("conjugate gradient requires a positive-definite response equation")
        step = residual_squared / curvature
        solution .+= step .* direction
        residual_vector .-= step .* action_direction
        next_residual_squared = dot(residual_vector, residual_vector)
        relative_residual = sqrt(next_residual_squared) / initial_norm
        relative_residual <= tolerance && return (
            solution=solution,
            iterations=iteration,
            residual=relative_residual,
        )
        direction .= residual_vector .+ (next_residual_squared / residual_squared) .* direction
        residual_squared = next_residual_squared
    end
    (solution=solution, iterations=iterations, residual=relative_residual)
end

function projector_density_jacobian(problem::ContactMeanField1D, rho)
    decomposition = eigen(hamiltonian(problem, rho))
    vectors = decomposition.vectors
    values = decomposition.values
    jacobian = zeros(Float64, length(rho), length(rho))
    scale = 2problem.coupling / problem.spacing
    for occupied in 1:problem.occupied
        for virtual in (problem.occupied + 1):length(values)
            product = vectors[:, occupied] .* vectors[:, virtual]
            jacobian .+= (scale / (values[occupied] - values[virtual])) .* (product * transpose(product))
        end
    end
    jacobian
end

function contour_density_jacobian(problem::ContactMeanField1D, rho, chart::CircularChart)
    H = hamiltonian(problem, rho)
    dimension = length(rho)
    identity_matrix = Matrix{ComplexF64}(I, dimension, dimension)
    jacobian = zeros(ComplexF64, dimension, dimension)
    for (z, weight) in zip(chart.nodes, chart.weights)
        resolvent = inv(z .* identity_matrix .- H)
        jacobian .+= (weight * problem.coupling / problem.spacing) .* (
            resolvent .* transpose(resolvent)
        )
    end
    real.(jacobian)
end
