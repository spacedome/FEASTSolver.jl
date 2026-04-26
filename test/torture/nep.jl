@testitem "torture NEP: sparse NLFEAST on gun cavity problem" tags=[:slow, :torture, :nep] setup=[FEASTTestSetup] begin
    using FEASTSolver
    using .FEASTTestSetup: initial_subspace, assert_converged

    T = feast_gallery("nlevp_native_gun")
    c, r = 140000.0 + 0.0im, 30000.0

    λ, _, res = nlfeast!(
        T,
        initial_subspace(size(T, 1), 36, 9901),
        8,
        4;
        c=c,
        r=r,
        ϵ=1e-8,
        store=false,
        spurious=1e-5,
    )

    inside = in_contour(λ, c, r)
    residuals_inside = res[inside]
    @test count(inside) >= 17
    @test count(residuals_inside .< 1e-6) >= 17
    assert_converged(residuals_inside[residuals_inside .< 1e-5]; atol=1e-6)
end
