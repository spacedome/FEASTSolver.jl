using LinearAlgebra
using Printf
using Random
using SparseArrays
using Statistics

side = parse(Int, get(ENV, "CACHE_PROBE_SIDE", "48"))
repetitions = parse(Int, get(ENV, "CACHE_PROBE_REPETITIONS", "5"))
n = side^2
line = spdiagm(
    -1 => fill(-1.0, side - 1),
    0 => fill(2.0, side),
    1 => fill(-1.0, side - 1),
)
identity_line = sparse(I, side, side)
operator = kron(identity_line, line) + kron(line, identity_line)
coordinates = collect(range(-1.0, 1.0; length=side))
potential = [0.2 * (x^2 + y^2) for x in coordinates, y in coordinates]
operator += spdiagm(0 => vec(potential))
node_matrix = 3.0im * sparse(I, n, n) - operator
rng = MersenneTwister(902)

lu(node_matrix)
factor_times = [@elapsed lu(node_matrix) for _ in 1:repetitions]
factor = lu(node_matrix)

println("sparse two-dimensional shifted-solve kernel probe")
@printf(
    "n=%d nnz=%d factor median=%.6f s stored≈%.2f MiB\n",
    n,
    nnz(node_matrix),
    median(factor_times),
    Base.summarysize(factor) / 2.0^20,
)
println("width   block time   time/column   columns/s   factor equivalent")
for width in (1, 2, 4, 8, 16, 32)
    right_hand_side = randn(rng, ComplexF64, n, width)
    factor \ right_hand_side
    times = [@elapsed factor \ right_hand_side for _ in 1:repetitions]
    block_time = median(times)
    column_time = block_time / width
    @printf(
        "%5d   %10.6f   %11.6f   %9.1f   %17.1f\n",
        width,
        block_time,
        column_time,
        width / block_time,
        median(factor_times) / column_time,
    )
end
