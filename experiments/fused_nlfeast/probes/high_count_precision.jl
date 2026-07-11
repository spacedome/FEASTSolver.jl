using LinearAlgebra
using Printf

function sine_moments(root_count, radius, moment_count)
    half_count = (root_count - 1) ÷ 2
    indices = -half_count:half_count
    roots = BigFloat[index * big(pi) / radius for index in indices]
    residues = BigFloat[isodd(abs(index)) ? -1 : 1 for index in indices]
    moments = BigFloat[
        sum(residues[j] * roots[j]^degree for j in eachindex(roots))
        for degree in 0:(moment_count - 1)
    ]
    roots, moments
end

function polynomial_from_roots(roots)
    coefficients = BigFloat[1]
    for root in roots
        updated = zeros(BigFloat, length(coefficients) + 1)
        updated[1:(end - 1)] .-= root .* coefficients
        updated[2:end] .+= coefficients
        coefficients = updated
    end
    coefficients
end

function recurrence_diagnostic(root_count, precision_bits)
    setprecision(BigFloat, precision_bits) do
        roots, moments = sine_moments(root_count, big(50), 2 * root_count)
        hankel = BigFloat[
            moments[row + column - 1]
            for row in 1:root_count, column in 1:root_count
        ]
        shifted = BigFloat[moments[root_count + row] for row in 1:root_count]
        recovered = vcat(-(hankel \ shifted), big(1))
        exact = polynomial_from_roots(roots)
        condition_number = opnorm(hankel, Inf) * opnorm(inv(hankel), Inf)
        coefficient_error = norm(recovered - exact, Inf) / norm(exact, Inf)
        recurrence_error = norm(hankel * recovered[1:end-1] + shifted, Inf) /
            norm(shifted, Inf)
        (
            log_condition=Float64(log10(condition_number)),
            coefficient_error=Float64(coefficient_error),
            recurrence_error=Float64(recurrence_error),
        )
    end
end

println("exact scalar sine power-Hankel conditioning")
println("roots  precision bits  log₁₀ κ∞(H₀)  coefficient error  recurrence error")
configurations = vec([
    (root_count, precision_bits)
    for root_count in (7, 15, 31, 63, 127), precision_bits in (128, 256, 512)
])
push!(configurations, (127, 1024))
for (root_count, precision_bits) in configurations
    outcome = recurrence_diagnostic(root_count, precision_bits)
    @printf(
        "%-6d %-15d %-14.2f %-18.3e %.3e\n",
        root_count,
        precision_bits,
        outcome.log_condition,
        outcome.coefficient_error,
        outcome.recurrence_error,
    )
end
