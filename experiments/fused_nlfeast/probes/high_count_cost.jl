using LinearAlgebra
using Printf

include(joinpath(@__DIR__, "..", "src", "FusedNLFEAST.jl"))
using .FusedNLFEAST

function modal_sine_overlap(values, left, right)
    ComplexF64[
        conj(left[i]) * FusedNLFEAST.sine_divided_difference(values[i], values[j]) * right[j]
        for i in eachindex(values), j in eachindex(values)
    ]
end

println("states  block action  modal action  ratio      relative error")
for states in (7, 15, 31, 63, 127, 255, 511)
    half = (states - 1) ÷ 2
    values = ComplexF64[k * pi for k in -half:half]
    state = Matrix(Diagonal(values))
    right = reshape(ComplexF64[1 + 0.01k for k in 1:states], 1, :)
    left = reshape(ComplexF64[1 - 0.005k for k in 1:states], 1, :)

    block_value = nothing
    block_seconds = @elapsed block_value = structured_state_divided_overlap(
        (ones(ComplexF64, 1, 1),),
        (sin,),
        state,
        left,
        state,
        right,
    )
    modal_value = nothing
    modal_seconds = @elapsed modal_value = modal_sine_overlap(values, vec(left), vec(right))
    relative_error = norm(block_value - modal_value) / norm(modal_value)
    @printf(
        "%-7d %.3e     %.3e     %-10.1f %.3e\n",
        states,
        block_seconds,
        modal_seconds,
        block_seconds / modal_seconds,
        relative_error,
    )
end
