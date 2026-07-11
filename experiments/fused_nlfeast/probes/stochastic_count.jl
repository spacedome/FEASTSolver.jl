using LinearAlgebra
using Printf
using Random
using Statistics

function oblique_projector(block_size, coupling, rng)
    signs = rand(rng, (-1.0, 1.0), block_size, block_size)
    coupling_block = (coupling / sqrt(block_size)) .* signs
    identity_block = Matrix{Float64}(I, block_size, block_size)
    [zeros(block_size, block_size) coupling_block; zeros(block_size, block_size) identity_block]
end

function complex_rademacher(rng, rows, columns)
    phases = ComplexF64[1, -1, im, -im]
    phases[rand(rng, 1:4, rows, columns)]
end

function trace_estimate(projector, samples, rng)
    probes = complex_rademacher(rng, size(projector, 1), samples)
    values = ComplexF64[dot(probes[:, j], projector * probes[:, j]) for j in 1:samples]
    mean(values)
end

rng = MersenneTwister(20260709)
block_size = 20
target = block_size
trials = 200

println("coupling  samples  ‖P‖F         median error  rounded-correct")
for coupling in (0.0, 1.0, 10.0, 100.0)
    projector = oblique_projector(block_size, coupling, rng)
    for samples in (4, 16, 64, 256)
        errors = Float64[]
        correct = 0
        for _ in 1:trials
            estimate = trace_estimate(projector, samples, rng)
            push!(errors, abs(estimate - target))
            correct += abs(imag(estimate)) <= 0.5 && round(Int, real(estimate)) == target
        end
        @printf(
            "%-9.1f %-8d %-13.3e %-13.3e %d/%d\n",
            coupling,
            samples,
            norm(projector),
            median(errors),
            correct,
            trials,
        )
    end
end
