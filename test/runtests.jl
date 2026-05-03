using TestItemRunner

function parse_test_tags(text)
    isempty(strip(text)) && return Set{Symbol}()
    Set(Symbol(strip(part)) for part in split(text, ",") if !isempty(strip(part)))
end

const TEST_FILTER = isempty(ARGS) || isempty(ARGS[1]) ? nothing : Regex(ARGS[1])
const INCLUDE_TAGS = length(ARGS) >= 2 ? parse_test_tags(ARGS[2]) : parse_test_tags(get(ENV, "FEAST_TEST_TAGS", ""))
const EXCLUDE_TAGS = length(ARGS) >= 3 ? parse_test_tags(ARGS[3]) : parse_test_tags(get(ENV, "FEAST_TEST_EXCLUDE_TAGS", ""))
const RUN_SLOW_TESTS = get(ENV, "FEAST_TEST_SLOW", "0") == "1"
const RUN_TORTURE_TESTS = get(ENV, "FEAST_TEST_TORTURE", "0") == "1"
const ONLY_TORTURE_TESTS = get(ENV, "FEAST_TEST_ONLY_TORTURE", "0") == "1"
const TEST_ROOT = normpath(@__DIR__)
const HAS_INCLUDE_TAGS = !isempty(INCLUDE_TAGS)

function testitem_filter(ti)
    startswith(normpath(ti.filename), TEST_ROOT) || return false
    name_ok = TEST_FILTER === nothing || occursin(TEST_FILTER, ti.name)
    include_tags_ok = isempty(INCLUDE_TAGS) || all(tag -> tag in ti.tags, INCLUDE_TAGS)
    exclude_tags_ok = isempty(EXCLUDE_TAGS) || all(tag -> !(tag in ti.tags), EXCLUDE_TAGS)
    slow_ok = RUN_SLOW_TESTS || HAS_INCLUDE_TAGS || !(:slow in ti.tags)
    torture_ok = RUN_TORTURE_TESTS || HAS_INCLUDE_TAGS || !(:torture in ti.tags)
    torture_only_ok = !ONLY_TORTURE_TESTS || (:torture in ti.tags)
    benchmark_ok = HAS_INCLUDE_TAGS || !(:benchmark in ti.tags)
    name_ok && include_tags_ok && exclude_tags_ok && slow_ok && torture_ok && torture_only_ok && benchmark_ok
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
