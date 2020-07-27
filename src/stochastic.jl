
function contour_estimate_eig(A::AbstractMatrix, contour::Contour, B=I;
                samples::Integer=min(100, size(A,1)), ϵ=1e-12, debug=false,
                mixed_prec=false, factorizer=lu, left_divider=ldiv!)


    N, m₀ = size(A,1), samples
    Ctype = if mixed_prec ComplexF32 else ComplexF64 end
    # X = if typeof(A) <: AbstractSparseMatrix
    #     sprandn(Ctype, N, m₀, min(1.0, m₀^2/N))
    # else
    #     randn(Ctype, N, m₀)
    # end
    ### Using sparse X does not work, possibly due to not being i.i.d. mean zero RV
    X = randn(Ctype, N, m₀)

    temp = zeros(Ctype, N, m₀)
    P = zeros(ComplexF64, m₀, m₀)
    ZmA = similar(A, Ctype)
    nodes = size(contour.nodes, 1)
    est = 0.0

    for i=1:nodes
        ZmA .= Ctype.(B*contour.nodes[i] - A)
        left_divider(temp, factorizer(ZmA), X)
        mul!(P, X', temp)
        est += tr(P)*contour.weights[i]/samples
        if debug print(".") end
    end
    if debug println() end

    return real(est)
end
