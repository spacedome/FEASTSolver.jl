args = copy(ARGS)
while !isempty(args) && isempty(args[end])
    pop!(args)
end

include(joinpath(@__DIR__, "options.jl"))
handle_test_info_args(args)

test_options = parse_test_options(args)
if test_options.list_tests
    print_matching_test_items(test_options; test_root=@__DIR__)
    exit(0)
end

using Pkg

Pkg.test(test_args=args)
