struct ContourCountError <: Exception
    expected::Int
    observed::Int
    winding::Float64
    maximum_phase_step::Float64
end

function Base.showerror(io::IO, error::ContourCountError)
    print(
        io,
        "contour contains ",
        error.observed,
        " states, expected ",
        error.expected,
        " (determinant winding ",
        error.winding,
        ", maximum sampled phase step ",
        error.maximum_phase_step / pi,
        "π)",
    )
end

function determinant_count(factorizations)
    phases = [angle(last(logabsdet(factorization))) for factorization in factorizations]
    increments = [
        angle(cis(phases[mod1(index + 1, length(phases))] - phases[index]))
        for index in eachindex(phases)
    ]
    winding = sum(increments) / (2pi)
    (
        count=round(Int, winding),
        winding=winding,
        integer_error=abs(winding - round(winding)),
        maximum_phase_step=maximum(abs, increments),
    )
end

struct InertiaBoundaryError <: Exception
    shift::Float64
end

Base.showerror(io::IO, error::InertiaBoundaryError) = print(
    io,
    "inertia shift ",
    error.shift,
    " is numerically indistinguishable from an eigenvalue",
)

function negative_inertia(D::Tridiagonal, shift; tolerance=1e-13)
    count = 0
    index = 1
    while index <= length(D.d)
        if index < length(D.d) && abs(D.dl[index]) > tolerance * max(
            1.0,
            abs(D.d[index]),
            abs(D.d[index + 1]),
        )
            determinant = real(D.d[index] * D.d[index + 1] - abs2(D.dl[index]))
            scale = max(abs(D.d[index] * D.d[index + 1]), abs2(D.dl[index]), 1.0)
            abs(determinant) > tolerance * scale || throw(InertiaBoundaryError(shift))
            if determinant < 0
                count += 1
            elseif real(D.d[index] + D.d[index + 1]) < 0
                count += 2
            end
            index += 2
        else
            value = real(D.d[index])
            abs(value) > tolerance * max(abs(value), 1.0) || throw(InertiaBoundaryError(shift))
            value < 0 && (count += 1)
            index += 1
        end
    end
    count
end

function hermitian_count_below(H, shift)
    shifted = Matrix(H)
    shifted[diagind(shifted)] .-= shift
    factorization = bunchkaufman(Hermitian(shifted))
    negative_inertia(factorization.D, shift)
end

Base.@kwdef struct OccupiedChartPolicy
    lower_bound::Float64
    upper_bound::Float64
    node_count::Int = 64
    relative_tolerance::Float64 = 1e-8
    gap_resolution::Float64 = 0.1
    padding::Float64 = 0.5
    expansion::Float64 = 2.0
end

function occupied_chart(policy::OccupiedChartPolicy, H, occupied)
    occupied >= 1 || throw(ArgumentError("occupied count must be positive"))
    policy.lower_bound < policy.upper_bound || throw(ArgumentError(
        "chart-policy lower bound must be below its upper bound",
    ))
    0 < policy.padding <= 1 || throw(ArgumentError("padding must lie in (0,1]"))
    policy.expansion > 1 || throw(ArgumentError("expansion must exceed one"))
    0 < policy.gap_resolution < 0.5 || throw(ArgumentError(
        "gap_resolution must lie in (0,0.5)",
    ))
    counts = Dict{Float64, Int}()
    count_at = function (shift)
        get!(counts, Float64(shift)) do
            hermitian_count_below(H, shift)
        end
    end

    lower = policy.lower_bound
    upper = policy.upper_bound
    width = upper - lower
    while count_at(lower) >= 1
        lower -= width
        width *= policy.expansion
    end
    while count_at(upper) < occupied + 1
        upper += width
        width *= policy.expansion
    end

    indices = unique((1, occupied, occupied + 1))
    brackets = Dict(index => (lower, upper) for index in indices)
    refine! = function (index)
        left, right = brackets[index]
        middle = (left + right) / 2
        brackets[index] = count_at(middle) < index ? (middle, right) : (left, middle)
    end

    refinement_count = 0
    while brackets[occupied][2] >= brackets[occupied + 1][1]
        index = (brackets[occupied][2] - brackets[occupied][1]) >=
            (brackets[occupied + 1][2] - brackets[occupied + 1][1]) ?
            occupied : occupied + 1
        refine!(index)
        refinement_count += 1
        refinement_count <= 256 || error("inertia slicing could not separate the occupied gap")
    end

    while true
        gap_lower_bound = brackets[occupied + 1][1] - brackets[occupied][2]
        scale = max(1.0, abs(lower), abs(upper))
        target_width = max(
            policy.relative_tolerance * scale,
            policy.gap_resolution * gap_lower_bound,
        )
        widths = Dict(index => brackets[index][2] - brackets[index][1] for index in indices)
        maximum(values(widths)) <= target_width && break
        refine!(argmax(index -> widths[index], indices))
        refinement_count += 1
        refinement_count <= 512 || error("inertia slicing did not reach gap-relative resolution")
    end

    locate(index) = sum(brackets[index]) / 2

    first_value = locate(1)
    occupied_value = locate(occupied)
    exterior_value = locate(occupied + 1)
    gap = exterior_value - occupied_value
    gap > 0 || error("occupied chart policy did not resolve a positive spectral gap")
    left_boundary = first_value - policy.padding * gap
    right_boundary = (occupied_value + exterior_value) / 2
    chart = CircularChart(
        (left_boundary + right_boundary) / 2,
        (right_boundary - left_boundary) / 2,
        policy.node_count,
    )
    (
        chart=chart,
        gap=gap,
        first_value=first_value,
        occupied_value=occupied_value,
        exterior_value=exterior_value,
        factorization_count=length(counts),
    )
end

resolve_chart(chart::CircularChart, H, occupied) = (
    chart=chart,
    gap=NaN,
    factorization_count=0,
)

resolve_chart(policy::OccupiedChartPolicy, H, occupied) = occupied_chart(policy, H, occupied)
