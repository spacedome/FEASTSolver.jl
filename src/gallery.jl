# FEAST-native nonlinear operator interface and small gallery operators.
#
# These mirror the NLEVP/NEP-PACK gallery problems we use for tests and
# experiments, but expose the contract FEAST wants: a top-level callable
# object, an explicit prototype, an in-place materializer, and an action path.

abstract type AbstractFeastOperator end
abstract type AbstractFeastGalleryOperator <: AbstractFeastOperator end

struct MaterializedMatrixOperator{F,P} <: AbstractFeastOperator
    T::F
    prototype::P
end

struct SparseCombinationPattern{Tv,Ti}
    m::Int
    n::Int
    colptr::Vector{Ti}
    rowval::Vector{Ti}
    values::Vector{Vector{Tv}}
end

struct PolynomialGalleryOperator{A<:AbstractVector,P} <: AbstractFeastGalleryOperator
    matrices::A
    sparse_pattern::P
end

struct HadelerGalleryOperator{T<:AbstractMatrix} <: AbstractFeastGalleryOperator
    A0::T
    A2::T
    B::T
end

struct LoadedStringGalleryOperator{T<:AbstractMatrix} <: AbstractFeastGalleryOperator
    A0::T
    A1::T
    B0::T
    B1::T
    C::T
    sigma::Float64
end

struct GunGalleryOperator{T<:SparseMatrixCSC{Float64,Int},P} <: AbstractFeastGalleryOperator
    K::T
    M::T
    W1::T
    W2::T
    sparse_pattern::P
end

struct SchrodingerMoveBCGalleryOperator{T<:SparseMatrixCSC{Float64,Int},P} <: AbstractFeastGalleryOperator
    A0::T
    I0::T
    G::T
    F::T
    L0::Float64
    L1::Float64
    V0::Float64
    sparse_pattern::P
end

struct GalleryMatrixMaterializer{O<:AbstractFeastOperator}
    operator::O
end

Base.size(op::MaterializedMatrixOperator) = size(op.prototype)
Base.size(op::MaterializedMatrixOperator, dim::Integer) = size(op.prototype, dim)
Base.size(op::PolynomialGalleryOperator) = size(op.matrices[1])
Base.size(op::PolynomialGalleryOperator, dim::Integer) = size(op.matrices[1], dim)
Base.size(op::HadelerGalleryOperator) = size(op.A0)
Base.size(op::HadelerGalleryOperator, dim::Integer) = size(op.A0, dim)
Base.size(op::LoadedStringGalleryOperator) = size(op.A0)
Base.size(op::LoadedStringGalleryOperator, dim::Integer) = size(op.A0, dim)
Base.size(op::GunGalleryOperator) = size(op.K)
Base.size(op::GunGalleryOperator, dim::Integer) = size(op.K, dim)
Base.size(op::SchrodingerMoveBCGalleryOperator) = size(op.A0)
Base.size(op::SchrodingerMoveBCGalleryOperator, dim::Integer) = size(op.A0, dim)
(op::AbstractFeastOperator)(z) = operator_matrix(op, z)
(materializer::GalleryMatrixMaterializer)(M, z) = materialize!(M, materializer.operator, z)

matrix_operator(T, prototype::AbstractMatrix) = MaterializedMatrixOperator(T, prototype)
matrix_materializer(op::AbstractFeastOperator) = GalleryMatrixMaterializer(op)

operator_prototype(op::MaterializedMatrixOperator) = similar(op.prototype, ComplexF64)

function operator_matrix(op::AbstractFeastOperator, z)
    M = operator_prototype(op)
    materialize!(M, op, z)
    M
end

function materialize!(M, op::MaterializedMatrixOperator, z)
    copyto!(M, op.T(z))
    M
end

function operator_action_workspace(op::AbstractFeastOperator, Y::AbstractVecOrMat, V::AbstractVecOrMat)
    similar(Y)
end

function mul!(Y::AbstractVecOrMat, op::MaterializedMatrixOperator, z, V::AbstractVecOrMat)
    mul!(Y, op.T(z), V)
end

function mul!(Y::AbstractVecOrMat, op::AbstractFeastGalleryOperator, z, V::AbstractVecOrMat)
    mul!(Y, op, z, V, operator_action_workspace(op, Y, V))
end

function sparse_combination_pattern(matrices)
    isempty(matrices) && error("sparse combination requires at least one matrix")
    m, n = size(matrices[1])
    rows = Int[]
    cols = Int[]
    for A in matrices
        size(A) == (m, n) || error("all sparse combination matrices must have matching dimensions")
        for col in axes(A, 2)
            for p in A.colptr[col]:(A.colptr[col+1]-1)
                push!(rows, A.rowval[p])
                push!(cols, col)
            end
        end
    end

    pattern = sparse(rows, cols, ones(Float64, length(rows)), m, n)
    lookups = [Dict{Int,Int}() for _ in 1:n]
    for col in axes(pattern, 2)
        for p in pattern.colptr[col]:(pattern.colptr[col+1]-1)
            lookups[col][pattern.rowval[p]] = p
        end
    end

    Tv = promote_type(map(eltype, matrices)...)
    values = [zeros(Tv, length(pattern.nzval)) for _ in matrices]
    for (matrix_index, A) in pairs(matrices)
        aligned_values = values[matrix_index]
        for col in axes(A, 2)
            lookup = lookups[col]
            for p in A.colptr[col]:(A.colptr[col+1]-1)
                aligned_values[lookup[A.rowval[p]]] = A.nzval[p]
            end
        end
    end
    SparseCombinationPattern(m, n, copy(pattern.colptr), copy(pattern.rowval), values)
end

function sparse_pattern_prototype(pattern::SparseCombinationPattern)
    SparseMatrixCSC(pattern.m, pattern.n, copy(pattern.colptr), copy(pattern.rowval), zeros(ComplexF64, length(pattern.rowval)))
end

function materialize_sparse_combination!(M::SparseMatrixCSC, pattern::SparseCombinationPattern, coefficients)
    M.m == pattern.m && M.n == pattern.n || error("sparse output matrix has incorrect dimensions")
    M.colptr == pattern.colptr && M.rowval == pattern.rowval || error("sparse output matrix does not match gallery sparsity pattern")
    fill!(M.nzval, zero(eltype(M.nzval)))
    @inbounds for (values, coefficient) in zip(pattern.values, coefficients)
        for i in eachindex(M.nzval)
            M.nzval[i] += coefficient * values[i]
        end
    end
    M
end

function materialize_sparse_polynomial!(M::SparseMatrixCSC, pattern::SparseCombinationPattern, z)
    M.m == pattern.m && M.n == pattern.n || error("sparse output matrix has incorrect dimensions")
    M.colptr == pattern.colptr && M.rowval == pattern.rowval || error("sparse output matrix does not match gallery sparsity pattern")
    fill!(M.nzval, zero(eltype(M.nzval)))
    power = one(z)
    @inbounds for values in pattern.values
        for i in eachindex(M.nzval)
            M.nzval[i] += power * values[i]
        end
        power *= z
    end
    M
end

function PolynomialGalleryOperator(matrices::AbstractVector)
    sparse_pattern = all(A -> A isa SparseMatrixCSC, matrices) ? sparse_combination_pattern(matrices) : nothing
    PolynomialGalleryOperator(matrices, sparse_pattern)
end

function operator_prototype(op::PolynomialGalleryOperator)
    op.sparse_pattern !== nothing && return sparse_pattern_prototype(op.sparse_pattern)
    similar(op.matrices[1], ComplexF64)
end

operator_prototype(op::HadelerGalleryOperator) = similar(op.A0, ComplexF64)
operator_prototype(op::LoadedStringGalleryOperator) = similar(op.A0, ComplexF64)
operator_prototype(op::GunGalleryOperator) = sparse_pattern_prototype(op.sparse_pattern)
operator_prototype(op::SchrodingerMoveBCGalleryOperator) = sparse_pattern_prototype(op.sparse_pattern)

function materialize!(M, op::PolynomialGalleryOperator, z)
    copyto!(M, op.matrices[end])
    for i in (length(op.matrices)-1):-1:1
        Ai = op.matrices[i]
        @. M = z * M + Ai
    end
    M
end

function materialize!(M::SparseMatrixCSC, op::PolynomialGalleryOperator, z)
    if op.sparse_pattern === nothing
        copyto!(M, op.matrices[end])
        for i in (length(op.matrices)-1):-1:1
            Ai = op.matrices[i]
            @. M = z * M + Ai
        end
        return M
    end
    materialize_sparse_polynomial!(M, op.sparse_pattern, z)
end

function mul!(Y::AbstractVecOrMat, op::PolynomialGalleryOperator, z, V::AbstractVecOrMat, workspace::AbstractVecOrMat)
    mul!(Y, op.matrices[end], V)
    for i in (length(op.matrices)-1):-1:1
        mul!(workspace, op.matrices[i], V)
        @. Y = z * Y + workspace
    end
    Y
end

function materialize!(M, op::HadelerGalleryOperator, z)
    z2 = z^2
    ez = exp(z) - one(z)
    @. M = -op.A0 + z2 * op.A2 + ez * op.B
    M
end

function mul!(Y::AbstractVecOrMat, op::HadelerGalleryOperator, z, V::AbstractVecOrMat, workspace::AbstractVecOrMat)
    z2 = z^2
    ez = exp(z) - one(z)
    mul!(Y, op.A0, V)
    rmul!(Y, -one(z))
    mul!(workspace, op.A2, V)
    axpy!(z2, workspace, Y)
    mul!(workspace, op.B, V)
    axpy!(ez, workspace, Y)
    Y
end

function materialize!(M, op::LoadedStringGalleryOperator, z)
    rational = z / (z - op.sigma)
    @. M = op.A0 + op.A1 - z * (op.B0 + op.B1) + rational * op.C
    M
end

function mul!(Y::AbstractVecOrMat, op::LoadedStringGalleryOperator, z, V::AbstractVecOrMat, workspace::AbstractVecOrMat)
    rational = z / (z - op.sigma)
    mul!(Y, op.A0, V)
    mul!(workspace, op.A1, V)
    axpy!(one(z), workspace, Y)
    mul!(workspace, op.B0, V)
    axpy!(-z, workspace, Y)
    mul!(workspace, op.B1, V)
    axpy!(-z, workspace, Y)
    mul!(workspace, op.C, V)
    axpy!(rational, workspace, Y)
    Y
end

function materialize!(M::SparseMatrixCSC, op::GunGalleryOperator, z)
    materialize_sparse_combination!(
        M,
        op.sparse_pattern,
        (one(z), -z, 1im * sqrt(z), 1im * sqrt(z - 108.8774^2)),
    )
end

function mul!(Y::AbstractVecOrMat, op::GunGalleryOperator, z, V::AbstractVecOrMat, workspace::AbstractVecOrMat)
    mul!(Y, op.K, V)
    mul!(workspace, op.M, V)
    axpy!(-z, workspace, Y)
    mul!(workspace, op.W1, V)
    axpy!(1im * sqrt(z), workspace, Y)
    mul!(workspace, op.W2, V)
    axpy!(1im * sqrt(z - 108.8774^2), workspace, Y)
    Y
end

function schrodinger_movebc_coefficients(op::SchrodingerMoveBCGalleryOperator, z)
    root = sqrt(z + op.V0)
    span = op.L1 - op.L0
    g = cosh(span * root)
    f = sinh(span * root) / root
    (one(z), -z, g, f)
end

function materialize!(M::SparseMatrixCSC, op::SchrodingerMoveBCGalleryOperator, z)
    materialize_sparse_combination!(M, op.sparse_pattern, schrodinger_movebc_coefficients(op, z))
end

function mul!(Y::AbstractVecOrMat, op::SchrodingerMoveBCGalleryOperator, z, V::AbstractVecOrMat, workspace::AbstractVecOrMat)
    _, mz, g, f = schrodinger_movebc_coefficients(op, z)
    mul!(Y, op.A0, V)
    mul!(workspace, op.I0, V)
    axpy!(mz, workspace, Y)
    mul!(workspace, op.G, V)
    axpy!(g, workspace, Y)
    mul!(workspace, op.F, V)
    axpy!(f, workspace, Y)
    Y
end

function feast_gallery(name::AbstractString, args...; kwargs...)
    if name == "polynomial"
        return PolynomialGalleryOperator(args[1])
    elseif name == "nlevp_native_hadeler"
        return hadeler_gallery(args...; kwargs...)
    elseif name == "nlevp_native_loaded_string"
        return loaded_string_gallery(args...; kwargs...)
    elseif name == "nlevp_native_gun"
        return gun_gallery()
    elseif name == "schrodinger_movebc"
        return schrodinger_movebc_gallery(args...; kwargs...)
    end
    error("unknown FEAST gallery problem '$name'")
end

function hadeler_gallery(alpha=100, n=8)
    i = collect(1:n)
    columns = ones(n) * i'
    A0 = Matrix{Float64}(alpha * I, n, n)
    A2 = n * Matrix{Float64}(I, n, n) .+ 1.0 ./ (columns .+ columns')
    B = ((n + 1) .- max.(columns', columns)) .* (i * i')
    HadelerGalleryOperator(A0, A2, B)
end

function gallery_toeplitz(v::AbstractVector)
    n = length(v)
    T = zeros(eltype(v), n, n)
    for i in 1:n
        for j in 1:(n-i+1)
            T[i, j+i-1] = v[j]
            T[j+i-1, i] = v[j]
        end
    end
    T
end

function loaded_string_gallery(n=20, kappa=1, m=1)
    A0 = gallery_toeplitz(SparseVector([2.0 * n; -n; zeros(n - 2)]))
    A1 = zeros(n, n)
    A1[n, n] = n - A0[n, n]
    B0 = gallery_toeplitz(SparseVector([4 / (6 * n); 1 / (6 * n); zeros(n - 2)]))
    B1 = zeros(n, n)
    B1[n, n] = 2 / (6 * n) - B0[n, n]
    C = zeros(n, n)
    C[n, n] = kappa
    LoadedStringGalleryOperator(A0, A1, B0, B1, C, Float64(kappa / m))
end

function read_gallery_sparse_matrix(filename, ::Type{T}=Int64) where {T}
    data = open(filename) do io
        readlines(io)
    end
    m = parse(T, data[1])
    n = parse(T, data[2])
    nnz = Int((length(data) - 2) ÷ 3)
    I = map(x -> parse(T, x), data[3:(2+nnz)])
    J = map(x -> parse(T, x), data[(3+nnz):(2+2nnz)])
    V = map(x -> parse(Float64, x), data[(3+2nnz):(2+3nnz)])
    sparse(I, J, V, m, n)
end

function gun_gallery_data_dir()
    local_data = normpath(joinpath(@__DIR__, "..", "data", "nlevp"))
    if isfile(joinpath(local_data, "gun_K.txt"))
        return local_data
    end

    nep_path = Base.find_package("NonlinearEigenproblems")
    nep_path === nothing && error("gun gallery data not found; expected data/nlevp/gun_*.txt or NonlinearEigenproblems gallery data")
    joinpath(dirname(nep_path), "gallery_extra", "converted_nlevp")
end

function gun_gallery()
    base = joinpath(gun_gallery_data_dir(), "gun_")
    K = read_gallery_sparse_matrix(base * "K.txt")
    M = read_gallery_sparse_matrix(base * "M.txt")
    W1 = read_gallery_sparse_matrix(base * "W1.txt")
    W2 = read_gallery_sparse_matrix(base * "W2.txt")
    GunGalleryOperator(K, M, W1, W2, sparse_combination_pattern((K, M, W1, W2)))
end

function schrodinger_movebc_gallery(n::Integer=1000, L0=1, L1=8, α=25 * π / 2, V0=10.0)
    n >= 3 || error("schrodinger_movebc requires n >= 3")
    L0, L1, α, V0 = Float64(L0), Float64(L1), Float64(α), Float64(V0)
    xv = collect(range(0.0, stop=L0, length=n))
    h = xv[2] - xv[1]
    potential = 1 .+ sin.(α .* xv[1:end-1])

    Dn = spdiagm(
        -1 => [ones(n - 2); 0.0] ./ h^2,
        0 => [-2 .* ones(n - 1); 0.0] ./ h^2,
        1 => ones(n - 1) ./ h^2,
    )
    Vn = spdiagm(0 => [potential; 0.0])
    I0 = spdiagm(0 => [ones(n - 1); 0.0])

    G = sparse([n], [n], [1.0], n, n)
    F = sparse(
        [n, n, n],
        [n - 2, n - 1, n],
        [1 / (2h), -2 / h, 3 / (2h)],
        n,
        n,
    )
    A0 = Dn - Vn
    SchrodingerMoveBCGalleryOperator(A0, I0, G, F, L0, L1, V0, sparse_combination_pattern((A0, I0, G, F)))
end
