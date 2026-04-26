@testitem "gallery operators: materialize and apply consistently" setup=[FEASTTestSetup] begin
    using FEASTSolver
    using LinearAlgebra
    using Random
    using SparseArrays

    rng = MersenneTwister(7781)
    operators = Any[
        matrix_operator(z -> z * Matrix{ComplexF64}(I, 8, 8) - Diagonal(complex.(1.0:8.0)), zeros(ComplexF64, 8, 8)),
        feast_gallery(
            "polynomial",
            [spdiagm(0 => complex.(-1.0:-1.0:-8)), sparse(I, 8, 8)],
        ),
        feast_gallery("nlevp_native_hadeler", 100, 8),
        feast_gallery("nlevp_native_loaded_string", 8, 1, 1),
    ]

    for op in operators
        z = 1.2 + 0.3im
        M = operator_prototype(op)
        materialize!(M, op, z)
        M2 = similar(M)
        matrix_materializer(op)(M2, z)
        V = rand(rng, ComplexF64, size(op, 2), 3)
        Y = zeros(ComplexF64, size(op, 1), 3)

        mul!(Y, op, z, V)

        @test M2 ≈ M
        @test Y ≈ M * V
    end
end

@testitem "gallery operators: Schrodinger sparse materializer is in-place" setup=[FEASTTestSetup] begin
    using FEASTSolver
    using LinearAlgebra
    using Random

    rng = MersenneTwister(7782)
    T = feast_gallery("schrodinger_movebc", 12)
    M = operator_prototype(T)
    pattern = (copy(M.colptr), copy(M.rowval))
    z = -35.0 + 2.0im
    V = rand(rng, ComplexF64, size(T, 2), 2)
    Y = zeros(ComplexF64, size(T, 1), 2)

    materialize!(M, T, z)
    mul!(Y, T, z, V)

    @test M.colptr == pattern[1]
    @test M.rowval == pattern[2]
    @test Y ≈ M * V
end
