set shell := ["bash", "-cu"]

julia := env_var_or_default("JULIA", "julia")

default:
    @just --list

# Load the package. Use this after environment or module-layout changes.
smoke:
    {{julia}} --project=. --startup-file=no -e 'using FEASTSolver; println("FEASTSolver loaded")'

# Run the current automated test entrypoint.
test:
    {{julia}} --project=. --startup-file=no test/runtests.jl

# Run Julia's package test harness.
pkg-test:
    {{julia}} --project=. --startup-file=no -e 'using Pkg; Pkg.test()'

# Build local documentation.
docs:
    {{julia}} --project=docs --startup-file=no docs/make.jl

# Instantiate both package and docs environments.
instantiate:
    {{julia}} --project=. --startup-file=no -e 'using Pkg; Pkg.instantiate()'
    {{julia}} --project=docs --startup-file=no -e 'using Pkg; Pkg.instantiate()'

# Update both package and docs environments.
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
      'test/runtests.jl is the only automated test entrypoint right now.' \
      'Most other files in test/ are research scripts or experiments.' \
      'Julia 1.10-1.12 compat should eventually be checked in CI, not by expanding this dev shell.' \
      'Optional plotting, benchmarking, and upstream FEAST binding work should use separate environments or package extensions.'
