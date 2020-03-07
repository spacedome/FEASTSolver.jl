using FEASTSolver
using LinearAlgebra

function bf()
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

    return butterfly(M0, M1, M2, M3, M4)
end

function inside(z)
	return ( abs(z - complex(1.0,1.0)) <= 0.5) 
end


T = bf()
e, v, res = nlfeast!(T, rand(ComplexF64,64,30), 2^1, 0, c=complex(1.0,1.0), r=0.5)
e, v, res = @timev nlfeast!(T, rand(ComplexF64,64,30), 2^5, 20, c=complex(1.0,1.0), r=0.5)
# display(e)
print("\nmax res inside: ")
println(maximum(res[inside.(e)]))
print("number inside : ")
print(size(e[inside.(e)])[1])
