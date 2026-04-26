using TestItemRunner

const TEST_FILTER = isempty(ARGS) ? nothing : Regex(ARGS[1])
const RUN_SLOW_TESTS = get(ENV, "FEAST_TEST_SLOW", "0") == "1"
const RUN_TORTURE_TESTS = get(ENV, "FEAST_TEST_TORTURE", "0") == "1"
const ONLY_TORTURE_TESTS = get(ENV, "FEAST_TEST_ONLY_TORTURE", "0") == "1"
const TEST_ROOT = normpath(@__DIR__)

function testitem_filter(ti)
    startswith(normpath(ti.filename), TEST_ROOT) || return false
    name_ok = TEST_FILTER === nothing || occursin(TEST_FILTER, ti.name)
    slow_ok = RUN_SLOW_TESTS || TEST_FILTER !== nothing || !(:slow in ti.tags)
    torture_ok = RUN_TORTURE_TESTS || TEST_FILTER !== nothing || !(:torture in ti.tags)
    torture_only_ok = !ONLY_TORTURE_TESTS || (:torture in ti.tags)
    benchmark_ok = TEST_FILTER !== nothing || !(:benchmark in ti.tags)
    name_ok && slow_ok && torture_ok && torture_only_ok && benchmark_ok
end

include("support.jl")

include("fast/standard.jl")
include("fast/generalized.jl")
include("fast/nonlinear.jl")
include("fast/distributed.jl")
include("fast/gallery.jl")

include("torture/generated_linear.jl")
include("torture/nep.jl")

@run_package_tests filter=testitem_filter verbose=true
