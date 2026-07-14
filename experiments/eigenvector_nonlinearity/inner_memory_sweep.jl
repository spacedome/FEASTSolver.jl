using Printf

include(joinpath(@__DIR__, "src", "ProjectorMomentFEAST.jl"))
using .ProjectorMomentFEAST

function sweep_case(; coupling, occupied, points, mixing, inner_counts, windows)
    problem = contact_mean_field_1d(
        points=points,
        half_length=7.0,
        coupling=coupling,
        occupied=occupied,
    )
    initial = initial_orbitals(problem; seed=occupied == 1 ? 77 : 33)
    chart = OccupiedChartPolicy(lower_bound=0.0, upper_bound=20.0, node_count=48)
    width = min(2, occupied)
    depth = cld(occupied, width)
    for window in windows, inner in inner_counts
        result = solve_raw_windowed_nlfeast(
            problem,
            chart,
            initial;
            config=WindowedNLFEASTConfig(
                moment_depth=depth,
                probe_width=width,
                window_blocks=window,
                refresh_iterations=40,
                inner_schedule=:fixed,
                fixed_inner_iterations=inner,
                inner_mixing=mixing,
                density_tolerance=1e-7,
                residual_tolerance=1e-7,
            ),
        )
        @printf(
            "g=%2.0f p=%d h=%-4s inner=%2d  ok=%-5s refresh=%2d q=%2d residual=%.2e\n",
            coupling,
            occupied,
            window == 0 ? "grow" : string(window),
            inner,
            string(result.converged),
            result.refresh_count,
            size(result.basis, 2),
            result.residual,
        )
    end
end

println("inner-depth and subspace-memory sweep")
sweep_case(
    coupling=5.0,
    occupied=3,
    points=48,
    mixing=0.3,
    inner_counts=(1, 2, 3, 5, 8),
    windows=(1, 2, 3, 4, 0),
)
sweep_case(
    coupling=10.0,
    occupied=1,
    points=48,
    mixing=0.15,
    inner_counts=(2, 3, 5),
    windows=(1, 2, 3, 4, 0),
)
