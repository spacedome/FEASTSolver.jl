# Small problem gallery for moment-RII/NLFEAST experiments.
#
# These problems are intentionally compact and analytically controlled.  They
# are used by showcases, torture checks, and lower-rung FEAST diagnostics.

function diagonal_linear_problem()
    n = 10
    A = Diagonal(ComplexF64.(1:n))
    coeffs = [-Matrix(A), Matrix{ComplexF64}(I, n, n)]
    coeffs, 2.5 + 0im, 1.6, 4
end

function deficient_quadratic_problem()
    A0 = Matrix{ComplexF64}(Matrix(mmread(joinpath(REPO_ROOT, "data", "quadraticM0.mtx"))))
    A1 = Matrix{ComplexF64}(Matrix(mmread(joinpath(REPO_ROOT, "data", "quadraticM1.mtx"))))
    coeffs = [A0 - 0.02 * A1, 0.1 * A1, A1]
    coeffs, 0.0 + 0im, 0.25, 3
end

function butterfly_problem()
    N = diagm(-1 => ones(7))
    Mh0 = (4I + N + N') / 6
    Mh1 = N - N'
    Mh2 = -(2I - N - N')
    Mh3 = Mh1
    Mh4 = -Mh2
    c = [0.6 1.3; 1.3 0.1; 0.1 1.2; 1.0 1.0; 1.2 1.0]
    I8 = Matrix(I, 8, 8)
    coeffs = [
        c[1, 1] * kron(I8, Mh0) + c[1, 2] * kron(Mh0, I8),
        c[2, 1] * kron(I8, Mh1) + c[2, 2] * kron(Mh1, I8),
        c[3, 1] * kron(I8, Mh2) + c[3, 2] * kron(Mh2, I8),
        c[4, 1] * kron(I8, Mh3) + c[4, 2] * kron(Mh3, I8),
        c[5, 1] * kron(I8, Mh4) + c[5, 2] * kron(Mh4, I8),
    ]
    complex.(coeffs), 1.0 + 1.0im, 0.5, 16
end

function many_eigenvalue_diagonal_matrix()
    n = 20
    A = Matrix(Diagonal(ComplexF64.(1:n)))
    A, 5.5 + 0im, 5.1, 4
end

function grcar_matrix(n, bands=3)
    A = zeros(ComplexF64, n, n)
    for i in 1:n
        A[i, i] = 1
        i > 1 && (A[i, i - 1] = -1)
        for j in 1:bands
            i + j <= n && (A[i, i + j] = 1)
        end
    end
    A
end

function grcar_linear_matrix()
    A = grcar_matrix(32)
    A, 0.2 + 1.6im, 1.05, 8
end

function linear_reference_inside(A, center, radius)
    lambdas = ComplexF64.(eigvals(A))
    inside = FEASTSolver.in_contour(lambdas, center, radius)
    lambdas[inside]
end

function many_eigenvalue_quadratic_problem()
    n = 12
    inside_roots = ComplexF64[1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 6.0, 6.5, 7.0, 7.5]
    outside_roots = ComplexF64[16 + i for i in 1:n]
    coeffs = [
        Matrix(Diagonal(inside_roots .* outside_roots)),
        Matrix(Diagonal(-(inside_roots .+ outside_roots))),
        Matrix{ComplexF64}(I, n, n),
    ]
    coeffs, 4.25 + 0im, 3.75, 4
end

function polynomial_coefficients_from_roots(roots)
    coeffs = ComplexF64[1]
    for root in roots
        next = zeros(ComplexF64, length(coeffs) + 1)
        for (j, coeff) in pairs(coeffs)
            next[j] -= root * coeff
            next[j + 1] += coeff
        end
        coeffs = next
    end
    coeffs
end

function many_eigenvalue_nonnormal_polynomial_problem()
    n = 4
    degree = 8
    center = 0.0 + 0.0im
    radius = 2.5
    roots_by_direction = [
        ComplexF64[-2.0, -1.2, -0.2, 0.8, 1.7, 6.0, 7.0, 8.0],
        ComplexF64[-1.8, -1.0, 0.55, 1.0, 1.9, 6.5, 7.5, 8.5],
        ComplexF64[-1.6, -0.8, 0.2, 1.2, 2.1, 7.0, 8.0, 9.0],
        ComplexF64[-1.4, -0.6, 0.4, 1.4, 2.3, 7.5, 8.5, 9.5],
    ]
    scalar_coeffs = [polynomial_coefficients_from_roots(roots) for roots in roots_by_direction]

    # A single non-unitary similarity keeps the roots exact but makes the
    # physical eigenvectors strongly reused across many nonlinear eigenvalues.
    V = ComplexF64[
        1.0 0.7 -0.4 0.2
        0.2 1.0 0.8 -0.3
        -0.5 0.1 1.0 0.6
        0.4 -0.2 0.3 1.0
    ]
    Vinv = inv(V)
    coeffs = Matrix{ComplexF64}[]
    for j in 1:(degree + 1)
        push!(coeffs, V * Diagonal([scalar_coeffs[i][j] for i in 1:n]) * Vinv)
    end
    coeffs, center, radius, n
end

function dual_sensitive_polynomial_problem()
    n = 6
    degree = 6
    center = 0.0 + 0.0im
    radius = 2.0
    inside_sets = [
        ComplexF64[-1.5, -0.6, 0.3, 1.2],
        ComplexF64[-1.2, -0.3, 0.6, 1.5],
        ComplexF64[-0.9, -0.1, 0.9, 1.7],
    ]
    outside_sets = [
        ComplexF64[4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        ComplexF64[4.3, 5.3, 6.3, 7.3, 8.3, 9.3],
        ComplexF64[4.6, 5.6, 6.6, 7.6, 8.6, 9.6],
    ]
    roots_by_direction = [
        vcat(inside_sets[1], ComplexF64[6.0, 8.0]),
        vcat(inside_sets[2], ComplexF64[6.5, 8.5]),
        vcat(inside_sets[3], ComplexF64[7.0, 9.0]),
        outside_sets[1],
        outside_sets[2],
        outside_sets[3],
    ]
    scalar_coeffs = [polynomial_coefficients_from_roots(roots) for roots in roots_by_direction]
    # A moderately ill-conditioned Vandermonde similarity keeps the exact roots
    # known while making one-sided Galerkin extraction admit spurious roots.
    nodes = 1.0 .+ 0.2 .* (0:n - 1)
    V = ComplexF64[nodes[i]^(j - 1) for i in 1:n, j in 1:n]
    V[:, 2] .+= 0.05im .* V[:, 1]
    Vinv = inv(V)
    coeffs = Matrix{ComplexF64}[]
    for j in 1:(degree + 1)
        push!(coeffs, V * Diagonal([scalar_coeffs[i][j] for i in 1:n]) * Vinv)
    end
    expected = sort(vcat(inside_sets...); by=z -> (real(z), imag(z)))
    coeffs, center, radius, n, expected
end

function scalar_roots_in_contour(candidates, center, radius)
    roots = ComplexF64.(candidates)
    roots[FEASTSolver.in_contour(roots, center, radius)]
end

function real_periodic_roots_in_contour(base, period, center, radius)
    lower = (real(center) - radius - base) / period
    upper = (real(center) + radius - base) / period
    candidates = [base + period * k for k in floor(Int, lower)-2:ceil(Int, upper)+2]
    scalar_roots_in_contour(candidates, center, radius)
end

function imaginary_periodic_roots_in_contour(base, period, center, radius)
    lower = (imag(center) - radius - base) / period
    upper = (imag(center) + radius - base) / period
    candidates = [im * (base + period * k) for k in floor(Int, lower)-2:ceil(Int, upper)+2]
    scalar_roots_in_contour(candidates, center, radius)
end

function scalar_sine_case()
    (
        name="sin",
        f=z -> sin(z),
        df=z -> cos(z),
        fmat=S -> sin(S),
        roots=(center, radius) -> real_periodic_roots_in_contour(0.0, pi, center, radius),
    )
end

function scalar_cosine_case()
    (
        name="cos",
        f=z -> cos(z),
        df=z -> -sin(z),
        fmat=S -> cos(S),
        roots=(center, radius) -> real_periodic_roots_in_contour(0.5pi, pi, center, radius),
    )
end

function scalar_shifted_sine_case(alpha=0.3)
    a = asin(alpha)
    (
        name="sin_minus_$alpha",
        f=z -> sin(z) - alpha,
        df=z -> cos(z),
        fmat=S -> sin(S) .- alpha .* Matrix{ComplexF64}(I, size(S, 1), size(S, 2)),
        roots=(center, radius) -> vcat(
            real_periodic_roots_in_contour(a, 2pi, center, radius),
            real_periodic_roots_in_contour(pi - a, 2pi, center, radius),
        ),
    )
end

function scalar_squared_sine_case()
    (
        name="sin_squared",
        f=z -> sin(z)^2,
        df=z -> 2sin(z) * cos(z),
        fmat=S -> sin(S)^2,
        roots=(center, radius) -> real_periodic_roots_in_contour(0.0, pi, center, radius),
        multiplicity=2,
    )
end

function scalar_expm1_case()
    (
        name="exp_minus_1",
        f=z -> exp(z) - 1,
        df=z -> exp(z),
        fmat=S -> exp(S) .- Matrix{ComplexF64}(I, size(S, 1), size(S, 2)),
        roots=(center, radius) -> imaginary_periodic_roots_in_contour(0.0, 2pi, center, radius),
    )
end

function scalar_delay_case(; a=0.4, b=2.0, tau=1.0)
    (
        name="delay_a$(a)_b$(b)_tau$(tau)",
        f=z -> z + a - b * exp(-tau * z),
        df=z -> 1 + b * tau * exp(-tau * z),
        fmat=S -> S .+ a .* Matrix{ComplexF64}(I, size(S, 1), size(S, 2)) .-
                  b .* exp((-tau) * S),
        # Deliberately no closed-form oracle: this case validates the
        # count-driven policy against the full-operator argument-principle
        # count instead of exact root matching.
        roots=(center, radius) -> ComplexF64[],
    )
end

function scalar_two_delay_case(; a=0.15, b=1.6, tau=0.7, c=0.85, sigma=1.4)
    (
        name="two_delay_a$(a)_b$(b)_tau$(tau)_c$(c)_sigma$(sigma)",
        f=z -> z + a - b * exp(-tau * z) - c * exp(-sigma * z),
        df=z -> 1 + b * tau * exp(-tau * z) + c * sigma * exp(-sigma * z),
        fmat=S -> begin
            Ired = Matrix{ComplexF64}(I, size(S, 1), size(S, 2))
            S .+ a .* Ired .- b .* exp((-tau) * S) .- c .* exp((-sigma) * S)
        end,
        # No closed-form oracle: this tests count-driven stopping on a
        # quasipolynomial component with more than one delay scale.
        roots=(center, radius) -> ComplexF64[],
    )
end

function case_root_multiplicity(case)
    hasproperty(case, :multiplicity) ? case.multiplicity : 1
end

function expected_roots_counting_multiplicity(cases, center, radius)
    roots = ComplexF64[]
    for case in cases
        for root in case.roots(center, radius)
            for _ in 1:case_root_multiplicity(case)
                push!(roots, root)
            end
        end
    end
    sort(roots; by=z -> (real(z), imag(z)))
end
