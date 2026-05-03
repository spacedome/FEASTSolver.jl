const TEST_PRESETS = (
    (
        name="moment-core",
        description="focused higher-moment FEAST/SS/Beyn/NLFEAST reduction checks; excludes :moment_heavy and :distributed",
    ),
    (
        name="moment-heavy",
        description="expensive higher-moment diagnostics tagged :moment_heavy",
    ),
    (
        name="moment-count",
        description="count-driven moment-RII policy diagnostics",
    ),
    (
        name="torture",
        description="flagged numerical torture tests only",
    ),
)

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
            "fused contour samples|fused cache augmentation|" *
            "agrees across extractors without root oracle|canonical NLFEAST limit|" *
            "residual Laurent update|low-rank compression preserves update|" *
            "scalar Laurent truncation alone|positive moments expose",
        )
        exclude_tags = union_test_tags(exclude_tags, Set([:moment_heavy, :distributed]))
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
        valid = join((preset.name for preset in TEST_PRESETS), ", ")
        error("unknown test preset: $name. Available presets: $valid")
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
        if token in ("--slow", "--torture", "--only-torture", "--list-tests", "--dry-run") ||
                startswith(token, "--tags=") ||
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
    list_tests = false
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
        elseif arg in ("--list-tests", "--dry-run")
            list_tests = true
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
        list_tests=list_tests,
    )
end

function testitem_matches(name, tags, filename, options; test_root=nothing)
    if test_root !== nothing && !startswith(normpath(filename), normpath(test_root))
        return false
    end
    has_include_tags = !isempty(options.include_tags)
    name_ok = options.filter === nothing || occursin(options.filter, name)
    include_tags_ok = isempty(options.include_tags) || all(tag -> tag in tags, options.include_tags)
    exclude_tags_ok = isempty(options.exclude_tags) || all(tag -> !(tag in tags), options.exclude_tags)
    slow_ok = options.run_slow || has_include_tags || !(:slow in tags)
    torture_ok = options.run_torture || has_include_tags || !(:torture in tags)
    torture_only_ok = !options.only_torture || (:torture in tags)
    benchmark_ok = has_include_tags || !(:benchmark in tags)
    name_ok && include_tags_ok && exclude_tags_ok && slow_ok && torture_ok && torture_only_ok && benchmark_ok
end

function print_test_presets(io=stdout)
    println(io, "Available test presets:")
    for preset in TEST_PRESETS
        println(io, "  ", preset.name, ": ", preset.description)
    end
end

function print_test_help(io=stdout)
    println(io, "Usage:")
    println(io, "  just test [OPTIONS] [REGEX]")
    println(io, "  julia --project=. test/pkgtest.jl [OPTIONS] [REGEX]")
    println(io)
    println(io, "Options:")
    println(io, "  --slow                 Include tests tagged :slow.")
    println(io, "  --torture              Include tests tagged :torture.")
    println(io, "  --only-torture         Run only tests tagged :torture; implies --slow.")
    println(io, "  --tags TAGS            Require comma-separated tags.")
    println(io, "  --exclude TAGS         Exclude comma-separated tags.")
    println(io, "  --preset NAME          Apply a maintained test loop.")
    println(io, "  --list-tests           List matching test items and exit.")
    println(io, "  --dry-run              Alias for --list-tests.")
    println(io, "  --list-presets         Show maintained preset names.")
    println(io, "  --help                 Show this help.")
    println(io)
    print_test_presets(io)
end

function source_test_items(test_root)
    items = NamedTuple[]
    for (root, _, files) in walkdir(test_root)
        for file in sort(files)
            endswith(file, ".jl") || continue
            path = joinpath(root, file)
            path == joinpath(test_root, "options.jl") && continue
            for (line_number, line) in enumerate(eachline(path))
                occursin("@testitem", line) || continue
                name_match = match(r"@testitem\s+\"((?:\\.|[^\"])*)\"", line)
                name_match === nothing && continue
                name = replace(name_match.captures[1], "\\\"" => "\"")
                tags = Set{Symbol}()
                tag_match = match(r"tags\s*=\s*\[([^\]]*)\]", line)
                if tag_match !== nothing
                    for tag in eachmatch(r":([A-Za-z_][A-Za-z0-9_]*)", tag_match.captures[1])
                        push!(tags, Symbol(tag.captures[1]))
                    end
                end
                push!(items, (name=name, tags=tags, filename=path, line=line_number))
            end
        end
    end
    sort!(items; by=item -> (item.filename, item.line))
end

function print_matching_test_items(options; test_root=@__DIR__, io=stdout)
    items = source_test_items(test_root)
    matched = [
        item for item in items
        if testitem_matches(item.name, item.tags, item.filename, options; test_root=test_root)
    ]

    println(io, "Matching test items: ", length(matched), " / ", length(items))
    for item in matched
        relative = relpath(item.filename, test_root)
        tag_text = isempty(item.tags) ? "" : " tags=" * join(sort!(collect(string.(item.tags))), ",")
        println(io, "  ", relative, ":", item.line, "  ", item.name, tag_text)
    end
    nothing
end

function handle_test_info_args(args)
    if any(arg -> arg in ("--help", "-h"), args)
        print_test_help()
        exit(0)
    elseif any(arg -> arg == "--list-presets", args)
        print_test_presets()
        exit(0)
    end
    nothing
end
