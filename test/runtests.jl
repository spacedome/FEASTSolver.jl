using TestItemRunner

include("options.jl")

function maybe_handle_test_info_args(args)
    normalized = normalize_test_args(args)
    handle_test_info_args(normalized)
    normalized
end

const TEST_OPTIONS = parse_test_options(maybe_handle_test_info_args(ARGS))
const TEST_FILTER = TEST_OPTIONS.filter
const INCLUDE_TAGS = TEST_OPTIONS.include_tags
const EXCLUDE_TAGS = TEST_OPTIONS.exclude_tags
const RUN_SLOW_TESTS = TEST_OPTIONS.run_slow
const RUN_TORTURE_TESTS = TEST_OPTIONS.run_torture
const ONLY_TORTURE_TESTS = TEST_OPTIONS.only_torture
const TEST_ROOT = normpath(@__DIR__)

if TEST_OPTIONS.list_tests
    print_matching_test_items(TEST_OPTIONS; test_root=TEST_ROOT)
    exit(0)
end

function testitem_filter(ti)
    testitem_matches(ti.name, ti.tags, ti.filename, TEST_OPTIONS; test_root=TEST_ROOT)
end

include("support.jl")

include("fast/standard.jl")
include("fast/generalized.jl")
include("fast/nonlinear.jl")
include("fast/distributed.jl")
include("fast/gallery.jl")

include("torture/generated_linear.jl")
include("torture/nep.jl")
include("torture/moment_rii.jl")

@run_package_tests filter=testitem_filter verbose=true
