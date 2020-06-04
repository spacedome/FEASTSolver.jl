using NonlinearEigenproblems: nep_gallery, compute_Mder
using FEASTSolver
nep = nep_gallery("nlevp_native_gun")
T(x) = compute_Mder(nep, x)
n = size(nep, 1)
e, v, res = nlfeast!(T, rand(ComplexF64,n,10), 2^2, 5, c=complex(61330.0,63185.0), r=50000.0)
display(e)
print("\nmax res inside: ")
println(maximum(res[in_contour.(e, complex(61330.0, 63185.0), 50000.0)]))
print("number inside : ")
print(size(e[in_contour.(e, complex(61330.0, 63185.0), 50000.0)])[1])
#
