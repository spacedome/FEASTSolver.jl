using Pkg

args = copy(ARGS)
while !isempty(args) && isempty(args[end])
    pop!(args)
end

Pkg.test(test_args=args)
