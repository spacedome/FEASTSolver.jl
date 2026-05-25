_default_feast_worker_ids() = _normalize_feast_worker_ids(workers())

function _normalize_feast_worker_ids(worker_ids)
    ids = unique(collect(worker_ids))
    filter!(!=(myid()), ids)
    ids
end

function _local_complex_matrix(A::AbstractMatrix, dims)
    size(A) == dims || error("matrix dimensions must match A")
    Matrix{ComplexF64}(A)
end

function _local_complex_matrix(B::UniformScaling, dims)
    matrix = zeros(ComplexF64, dims)
    n = min(dims...)
    @inbounds for i in 1:n
        matrix[i, i] = B.λ
    end
    matrix
end

function _dense_feast_worker_assignments(worker_ids::Vector{Int}, node_count::Int)
    active_count = min(length(worker_ids), node_count)
    active_workers = worker_ids[1:active_count]
    assignments = [Int[] for _ in active_workers]
    for node_index in 1:node_count
        push!(assignments[mod1(node_index, active_count)], node_index)
    end
    active_workers, assignments
end
