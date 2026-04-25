set shell := ["bash", "-cu"]

julia := env_var_or_default("JULIA", "julia")

default:
    @just --list

# Load the package. Use this after environment or module-layout changes.
smoke:
    {{julia}} --project=. --startup-file=no -e 'using FEASTSolver; println("FEASTSolver loaded")'

# Run the package test harness. Optionally pass a regex matching testset names.
test filter='':
    {{julia}} --project=. --startup-file=no -e 'using Pkg; filter = ARGS[1]; Pkg.test(test_args=isempty(filter) ? String[] : [filter])' {{quote(filter)}}

# Run the package test harness including test items tagged as slow.
test-slow filter='':
    FEAST_TEST_SLOW=1 {{julia}} --project=. --startup-file=no -e 'using Pkg; filter = ARGS[1]; Pkg.test(test_args=isempty(filter) ? String[] : [filter])' {{quote(filter)}}

# Run a simple dense FEAST contour-parallel scaling benchmark.
bench-parallel:
    {{julia}} --project=. --startup-file=no benchmark/dense_parallel_scaling.jl

# Build local documentation.
docs:
    {{julia}} --project=docs --startup-file=no docs/make.jl

# Instantiate package and docs environments.
instantiate:
    {{julia}} --project=. --startup-file=no -e 'using Pkg; Pkg.instantiate()'
    {{julia}} --project=docs --startup-file=no -e 'using Pkg; Pkg.instantiate()'

# Update package and docs environments.
update:
    {{julia}} --project=. --startup-file=no -e 'using Pkg; Pkg.update()'
    {{julia}} --project=docs --startup-file=no -e 'using Pkg; Pkg.update()'

# Show package and docs dependency status.
status:
    {{julia}} --project=. --startup-file=no -e 'using Pkg; Pkg.status()'
    {{julia}} --project=docs --startup-file=no -e 'using Pkg; Pkg.status()'

# Show the repo assumptions future agents should keep in mind.
notes:
    @printf '%s\n' \
      'Use nix develop for Julia work.' \
      'The shell sets JULIA_PROJECT=@. and uses ./.julia as the first depot.' \
      'Run just smoke, just test, and just docs for the normal local loop.' \
      'Run just test REGEX to run only matching TestItems through Pkg.test(test_args=...).' \
      'Run just test-slow to include TestItems tagged :slow.' \
      'test/runtests.jl is the automated TestItemRunner entrypoint.' \
      'just test uses Pkg.test(); test-only dependencies live in Project.toml extras/targets.' \
      'Historical research scripts live under experiments/legacy_tests/ until curated.' \
      'Julia 1.10-1.12 compat should eventually be checked in CI, not by expanding this dev shell.' \
      'Optional plotting, benchmarking, and upstream FEAST binding work should use separate environments or package extensions.'
