# FEAST-native nonlinear operator interface.
#
# The solver wants an explicit prototype, an in-place materializer, and an
# action path. Gallery problems and user wrappers implement this contract.

abstract type AbstractFeastOperator end
abstract type AbstractFeastGalleryOperator <: AbstractFeastOperator end

struct MaterializedMatrixOperator{F,P} <: AbstractFeastOperator
    T::F
    prototype::P
end

struct GalleryMatrixMaterializer{O<:AbstractFeastOperator}
    operator::O
end

Base.size(op::MaterializedMatrixOperator) = size(op.prototype)
Base.size(op::MaterializedMatrixOperator, dim::Integer) = size(op.prototype, dim)
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
