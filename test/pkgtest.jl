args = copy(ARGS)
while !isempty(args) && isempty(args[end])
    pop!(args)
end

include(joinpath(@__DIR__, "options.jl"))
handle_test_info_args(args)

using Pkg

Pkg.test(test_args=args)
