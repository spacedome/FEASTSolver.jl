function polynomial_matrix(coefficients, z)
    matrix = complex.(coefficients[end])
    for index in (length(coefficients) - 1):-1:1
        @. matrix = z * matrix + coefficients[index]
    end
    matrix
end

function polynomial_divided_difference(coefficients, eta, theta)
    result = zeros(ComplexF64, size(coefficients[1]))
    for degree in 1:(length(coefficients) - 1)
        scalar = zero(ComplexF64)
        for left_power in 0:(degree - 1)
            scalar += eta^left_power * theta^(degree - 1 - left_power)
        end
        result .+= scalar .* coefficients[degree + 1]
    end
    result
end

function polynomial_divided_overlap(
    coefficients,
    left_values,
    left,
    right_values,
    right,
)
    count = length(right_values)
    result = zeros(ComplexF64, count, count)
    for degree in 1:(length(coefficients) - 1)
        projected = adjoint(left) * coefficients[degree + 1] * right
        for i in eachindex(left_values), j in eachindex(right_values)
            scalar = zero(ComplexF64)
            for left_power in 0:(degree - 1)
                scalar += left_values[i]^left_power * right_values[j]^(degree - 1 - left_power)
            end
            result[i, j] += scalar * projected[i, j]
        end
    end
    result
end

function diagonal_divided_overlap(
    divided_functions,
    left_coordinates,
    right_coordinates,
    left_values,
    right_values,
)
    count = length(right_values)
    result = zeros(ComplexF64, count, count)
    for i in eachindex(left_values), j in eachindex(right_values)
        for direction in eachindex(divided_functions)
            result[i, j] += conj(left_coordinates[direction, i]) *
                divided_functions[direction](left_values[i], right_values[j]) *
                right_coordinates[direction, j]
        end
    end
    result
end

function cardinal_sine(z)
    iszero(z) ? one(ComplexF64) : sin(z) / z
end

sine_divided_difference(eta, theta) =
    cos((eta + theta) / 2) * cardinal_sine((eta - theta) / 2)

cosine_divided_difference(eta, theta) =
    -sin((eta + theta) / 2) * cardinal_sine((eta - theta) / 2)

function exponential_divided_difference(eta, theta)
    half_difference = (eta - theta) / 2
    ratio = iszero(half_difference) ? one(ComplexF64) : sinh(half_difference) / half_difference
    exp((eta + theta) / 2) * ratio
end

function linear_case(A)
    A = Matrix{ComplexF64}(A)
    n = size(A, 1)
    identity_matrix = Matrix{ComplexF64}(I, n, n)
    T = z -> z .* identity_matrix .- A
    divided_difference = (eta, theta) -> copy(identity_matrix)
    divided_overlap = (left_values, left, right_values, right) -> adjoint(left) * right
    right_solve = (z, B) -> T(z) \ B
    left_solve = (z, B) -> adjoint(T(z)) \ B
    (
        T=T,
        divided_difference=divided_difference,
        divided_overlap=divided_overlap,
        right_solve=right_solve,
        left_solve=left_solve,
        dimension=n,
    )
end

function polynomial_case(coefficients)
    coefficients = Matrix{ComplexF64}.(coefficients)
    n = size(coefficients[1], 1)
    T = z -> polynomial_matrix(coefficients, z)
    divided_difference = (eta, theta) -> polynomial_divided_difference(coefficients, eta, theta)
    divided_overlap = (left_values, left, right_values, right) -> polynomial_divided_overlap(
        coefficients,
        left_values,
        left,
        right_values,
        right,
    )
    right_solve = (z, B) -> T(z) \ B
    left_solve = (z, B) -> adjoint(T(z)) \ B
    (
        T=T,
        divided_difference=divided_difference,
        divided_overlap=divided_overlap,
        right_solve=right_solve,
        left_solve=left_solve,
        dimension=n,
        coefficients=coefficients,
    )
end

function scalar_polynomial_coefficients(roots)
    coefficients = ComplexF64[1]
    for root in roots
        next = zeros(ComplexF64, length(coefficients) + 1)
        for index in eachindex(coefficients)
            next[index] -= root * coefficients[index]
            next[index + 1] += coefficients[index]
        end
        coefficients = next
    end
    coefficients
end

function many_root_polynomial_case()
    roots_by_direction = (
        ComplexF64[-2.0, -1.2, -0.2, 0.8, 1.7, 6.0, 7.0, 8.0],
        ComplexF64[-1.8, -1.0, 0.55, 1.0, 1.9, 6.5, 7.5, 8.5],
        ComplexF64[-1.6, -0.8, 0.2, 1.2, 2.1, 7.0, 8.0, 9.0],
        ComplexF64[-1.4, -0.6, 0.4, 1.4, 2.3, 7.5, 8.5, 9.5],
    )
    scalar_coefficients = scalar_polynomial_coefficients.(roots_by_direction)
    similarity = ComplexF64[
        1.0 0.7 -0.4 0.2
        0.2 1.0 0.8 -0.3
        -0.5 0.1 1.0 0.6
        0.4 -0.2 0.3 1.0
    ]
    inverse_similarity = inv(similarity)
    coefficients = Matrix{ComplexF64}[]
    for degree in 1:9
        diagonal = Diagonal(ComplexF64[direction[degree] for direction in scalar_coefficients])
        push!(coefficients, similarity * diagonal * inverse_similarity)
    end
    center = 0.0 + 0.0im
    radius = 2.5
    expected = sort(
        reduce(vcat, [roots[abs.(roots .- center) .< radius] for roots in roots_by_direction]);
        by=value -> (real(value), imag(value)),
    )
    merge(polynomial_case(coefficients), (center=center, radius=radius, expected=expected))
end

function scalar_sine_case()
    T = z -> fill(ComplexF64(sin(z)), 1, 1)
    divided_difference = (eta, theta) -> fill(sine_divided_difference(eta, theta), 1, 1)
    divided_overlap = function (left_values, left, right_values, right)
        ComplexF64[
            conj(left[1, i]) * sine_divided_difference(left_values[i], right_values[j]) * right[1, j]
            for i in eachindex(left_values), j in eachindex(right_values)
        ]
    end
    right_solve = (z, B) -> B ./ sin(z)
    left_solve = (z, B) -> B ./ conj(sin(z))
    (
        T=T,
        divided_difference=divided_difference,
        divided_overlap=divided_overlap,
        right_solve=right_solve,
        left_solve=left_solve,
        dimension=1,
    )
end

function canonical_one_root_case()
    roots = ComplexF64[-0.5, 0.2 + 0.3im, 0.6 - 0.2im]
    similarity = ComplexF64[1 0.7 -0.2; 0.1 1 0.5; -0.3 0.2 1]
    inverse_similarity = inv(similarity)
    T = z -> similarity * Diagonal(sin.(z .- roots)) * inverse_similarity
    divided_difference = (eta, theta) -> similarity *
        Diagonal(sine_divided_difference.(eta .- roots, theta .- roots)) * inverse_similarity
    divided_functions = Tuple(
        (eta, theta) -> sine_divided_difference(eta - root, theta - root)
        for root in roots
    )
    divided_overlap = (left_values, left, right_values, right) -> diagonal_divided_overlap(
        divided_functions,
        adjoint(similarity) * left,
        inverse_similarity * right,
        left_values,
        right_values,
    )
    right_solve = (z, B) -> T(z) \ B
    left_solve = (z, B) -> adjoint(T(z)) \ B
    (
        T=T,
        divided_difference=divided_difference,
        divided_overlap=divided_overlap,
        right_solve=right_solve,
        left_solve=left_solve,
        dimension=3,
        expected=roots,
    )
end

function nonnormal_analytic_case(; coupling=8.0)
    similarity = ComplexF64[1 coupling 0; 0 1 coupling / 2; 0 0 1]
    inverse_similarity = inv(similarity)
    functions = (sin, cos, z -> exp(z) - one(z))
    T = z -> similarity * Diagonal(ComplexF64[f(z) for f in functions]) * inverse_similarity
    structured_coefficients = Matrix{ComplexF64}[
        similarity[:, index] * transpose(inverse_similarity[index, :])
        for index in eachindex(functions)
    ]
    divided_differences = (sine_divided_difference, cosine_divided_difference, exponential_divided_difference)
    divided_difference = (eta, theta) -> similarity *
        Diagonal(ComplexF64[f(eta, theta) for f in divided_differences]) * inverse_similarity
    divided_overlap = (left_values, left, right_values, right) -> diagonal_divided_overlap(
        divided_differences,
        adjoint(similarity) * left,
        inverse_similarity * right,
        left_values,
        right_values,
    )
    right_solve = (z, B) -> T(z) \ B
    left_solve = (z, B) -> adjoint(T(z)) \ B
    (
        T=T,
        divided_difference=divided_difference,
        divided_overlap=divided_overlap,
        right_solve=right_solve,
        left_solve=left_solve,
        dimension=3,
        structured_coefficients=structured_coefficients,
        structured_functions=functions,
    )
end

function shared_eigenvector_case()
    coefficients = Matrix{ComplexF64}[
        ComplexF64[0 12 0; -2 14 0; 0 0 0],
        ComplexF64[-1 -6 0; 2 -9 0; 0 0 0],
        Matrix{ComplexF64}(I, 3, 3),
    ]
    merge(
        polynomial_case(coefficients),
        (
            contours=(
                (
                    center=1.5 + 0.0im,
                    radius=1.0,
                    expected=ComplexF64[1, 2],
                    expected_right=ComplexF64[1 0; 0 1; 0 0],
                    expected_left=ComplexF64[1 1; -1 -1; 0 0],
                    shared=:left,
                ),
                (
                    center=2.5 + 0.0im,
                    radius=1.0,
                    expected=ComplexF64[2, 3],
                    expected_right=ComplexF64[0 1; 1 1; 0 0],
                    expected_left=ComplexF64[1 2; -1 -3; 0 0],
                    shared=:none,
                ),
                (
                    center=3.5 + 0.0im,
                    radius=1.0,
                    expected=ComplexF64[3, 4],
                    expected_right=ComplexF64[1 1; 1 1; 0 0],
                    expected_left=ComplexF64[2 1; -3 -2; 0 0],
                    shared=:right,
                ),
            ),
        ),
    )
end
