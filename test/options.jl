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
    println(io, "  --list-presets         Show maintained preset names.")
    println(io, "  --help                 Show this help.")
    println(io)
    print_test_presets(io)
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
