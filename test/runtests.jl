using TestItemRunner

function parse_test_tags(text)
    isempty(strip(text)) && return Set{Symbol}()
    Set(Symbol(strip(part)) for part in split(text, ",") if !isempty(strip(part)))
end

function union_test_tags(left, right)
    result = copy(left)
    union!(result, right)
    result
end

function apply_test_preset(
    name,
    filter,
    include_tags,
    exclude_tags,
    run_slow,
    run_torture,
    only_torture,
)
    if name == "moment-core"
        filter = Regex(
            "linear SS-FEAST|dual linear RII|polynomial bridge agrees|dual reduced extraction|" *
            "agrees across extractors without root oracle|canonical NLFEAST limit|" *
            "residual Laurent update",
        )
        exclude_tags = union_test_tags(exclude_tags, Set([:moment_heavy]))
        run_slow = true
    elseif name == "moment-heavy"
        filter === nothing && (filter = Regex("moment RII"))
        include_tags = union_test_tags(include_tags, Set([:moment_heavy]))
        run_slow = true
    elseif name == "moment-count"
        filter === nothing && (filter = Regex("count-driven refinement"))
        run_slow = true
    elseif name == "torture"
        only_torture = true
        run_torture = true
        run_slow = true
    else
        error("unknown test preset: $name")
    end
    filter, include_tags, exclude_tags, run_slow, run_torture, only_torture
end

function normalize_test_args(args)
    length(args) == 1 || return args
    text = strip(only(args))
    startswith(text, "--") || return args

    tokens = split(text)
    normalized = String[]
    index = firstindex(tokens)
    while index <= lastindex(tokens)
        token = tokens[index]
        if token in ("--slow", "--torture", "--only-torture") || startswith(token, "--tags=") ||
                startswith(token, "--exclude=")
            push!(normalized, String(token))
        elseif startswith(token, "--preset=")
            push!(normalized, String(token))
        elseif token in ("--tags", "--exclude", "--preset")
            push!(normalized, String(token))
            index == lastindex(tokens) && error("$token requires a comma-separated value")
            index += 1
            push!(normalized, String(tokens[index]))
        elseif token == "--"
            index < lastindex(tokens) && push!(normalized, join(tokens[(index + 1):end], " "))
            break
        elseif startswith(token, "--")
            push!(normalized, String(token))
        else
            push!(normalized, join(tokens[index:end], " "))
            break
        end
        index += 1
    end
    normalized
end

function parse_test_options(args)
    args = normalize_test_args(args)
    filter = nothing
    include_tags = parse_test_tags(get(ENV, "FEAST_TEST_TAGS", ""))
    exclude_tags = parse_test_tags(get(ENV, "FEAST_TEST_EXCLUDE_TAGS", ""))
    run_slow = get(ENV, "FEAST_TEST_SLOW", "0") == "1"
    run_torture = get(ENV, "FEAST_TEST_TORTURE", "0") == "1"
    only_torture = get(ENV, "FEAST_TEST_ONLY_TORTURE", "0") == "1"
    positionals = String[]

    index = firstindex(args)
    while index <= lastindex(args)
        arg = args[index]
        if arg == "--slow"
            run_slow = true
        elseif arg == "--torture"
            run_torture = true
        elseif arg == "--only-torture"
            only_torture = true
            run_torture = true
            run_slow = true
        elseif arg == "--tags"
            index == lastindex(args) && error("--tags requires a comma-separated value")
            index += 1
            include_tags = parse_test_tags(args[index])
        elseif startswith(arg, "--tags=")
            include_tags = parse_test_tags(arg[8:end])
        elseif arg == "--preset"
            index == lastindex(args) && error("--preset requires a preset name")
            index += 1
            filter, include_tags, exclude_tags, run_slow, run_torture, only_torture =
                apply_test_preset(args[index], filter, include_tags, exclude_tags, run_slow, run_torture, only_torture)
        elseif startswith(arg, "--preset=")
            filter, include_tags, exclude_tags, run_slow, run_torture, only_torture =
                apply_test_preset(arg[10:end], filter, include_tags, exclude_tags, run_slow, run_torture, only_torture)
        elseif arg == "--exclude"
            index == lastindex(args) && error("--exclude requires a comma-separated value")
            index += 1
            exclude_tags = parse_test_tags(args[index])
        elseif startswith(arg, "--exclude=")
            exclude_tags = parse_test_tags(arg[11:end])
        elseif arg == "--"
            append!(positionals, args[(index + 1):end])
            break
        elseif startswith(arg, "--")
            error("unknown test option: $arg")
        else
            push!(positionals, arg)
        end
        index += 1
    end

    # Preserve the old positional API: REGEX [INCLUDE_TAGS] [EXCLUDE_TAGS].
    if !isempty(positionals) && !isempty(positionals[1])
        filter = Regex(positionals[1])
    end
    if length(positionals) >= 2 && !isempty(positionals[2])
        include_tags = parse_test_tags(positionals[2])
    end
    if length(positionals) >= 3 && !isempty(positionals[3])
        exclude_tags = parse_test_tags(positionals[3])
    end

    (
        filter=filter,
        include_tags=include_tags,
        exclude_tags=exclude_tags,
        run_slow=run_slow,
        run_torture=run_torture,
        only_torture=only_torture,
    )
end

const TEST_OPTIONS = parse_test_options(ARGS)
const TEST_FILTER = TEST_OPTIONS.filter
const INCLUDE_TAGS = TEST_OPTIONS.include_tags
const EXCLUDE_TAGS = TEST_OPTIONS.exclude_tags
const RUN_SLOW_TESTS = TEST_OPTIONS.run_slow
const RUN_TORTURE_TESTS = TEST_OPTIONS.run_torture
const ONLY_TORTURE_TESTS = TEST_OPTIONS.only_torture
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
