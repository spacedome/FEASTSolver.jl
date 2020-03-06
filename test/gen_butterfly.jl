using FEASTSolver
using LinearAlgebra
# using MatrixMarket
using SparseArrays

function mmwrite(filename, matrix :: SparseMatrixCSC)
  open(filename, "w") do file
    elem = eltype(matrix) <: Bool ? "pattern" :
           eltype(matrix) <: Integer ?  "integer" :
           eltype(matrix) <: AbstractFloat ? "real" :
           eltype(matrix) <: Complex ? "complex" :
           error("Invalid matrix type")
      sym = "general"

      # write mm header
      write(file, "%%MatrixMarket matrix coordinate $elem $sym\n")

      # only use lower triangular part of symmetric and Hermitian matrices
      #if issymmetric(matrix) || ishermitian(matrix)
      #    matrix = tril(matrix)
      #end

      # write matrix size and number of nonzeros
      write(file, "$(size(matrix, 1)) $(size(matrix, 2)) $(nnz(matrix))\n")

      rows = rowvals(matrix)
      vals = nonzeros(matrix)
      for i in 1:size(matrix, 2)
          for j in nzrange(matrix, i)
              write(file, "$(rows[j]) $i")
              if elem == "pattern" # omit values on pattern matrices
              elseif elem == "complex"
                  write(file, " $(real(vals[j])) $(imag(vals[j]))")
              else
                  write(file, " $(vals[j])")
              end
              write(file, "\n")
          end
      end
  end
end

    N = diagm(-1 => [1,1,1,1,1,1,1])
    Mh0 = 1/6*(4*I + N + N')
    Mh1 = N - N'
    Mh2 = -1*(2*I - N - N')
    Mh3 = Mh1
    Mh4 = -Mh2
    c = [0.6 1.3; 1.3 0.1; 0.1 1.2; 1.0 1.0; 1.2 1.0]
    I8 = Matrix(I, 8, 8)
    M0 = c[1,1] * kron(I8, Mh0) + c[1,2] * kron(Mh0, I8)
    M1 = c[2,1] * kron(I8, Mh1) + c[2,2] * kron(Mh1, I8)
    M2 = c[3,1] * kron(I8, Mh2) + c[3,2] * kron(Mh2, I8)
    M3 = c[4,1] * kron(I8, Mh3) + c[4,2] * kron(Mh3, I8)
    M4 = c[5,1] * kron(I8, Mh4) + c[5,2] * kron(Mh4, I8)
    function butterfly(A0, A1, A2, A3, A4)
        f(z) =  z^4*A4 + z^3*A3 + z^2*A2 + z*A1 + A0
    end

    mmwrite("data/butterflyM0.mtx", sparse(M0))
    mmwrite("data/butterflyM1.mtx", sparse(M1))
    mmwrite("data/butterflyM2.mtx", sparse(M2))
    mmwrite("data/butterflyM3.mtx", sparse(M3))
    mmwrite("data/butterflyM4.mtx", sparse(M4))
