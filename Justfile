set shell := ["bash", "-cu"]

julia := env_var_or_default("JULIA", "julia")

default:
    @just --list

# Load the package. Use this after environment or module-layout changes.
smoke:
    {{julia}} --project=. --startup-file=no -e 'using FEASTSolver; println("FEASTSolver loaded")'

# Run the package test harness. Args: regex filter, required tags, excluded tags.
# Tags are comma-separated, e.g. `just test 'moment RII' '' moment_heavy`.
test filter='' tags='' exclude='':
    {{julia}} --project=. --startup-file=no test/pkgtest.jl {{quote(filter)}} {{quote(tags)}} {{quote(exclude)}}

# Run the package test harness including test items tagged as slow.
test-slow filter='' tags='' exclude='':
    FEAST_TEST_SLOW=1 {{julia}} --project=. --startup-file=no test/pkgtest.jl {{quote(filter)}} {{quote(tags)}} {{quote(exclude)}}

# Run flagged numerical torture tests. These are correctness tests, not benchmarks.
test-torture filter='':
    FEAST_TEST_ONLY_TORTURE=1 FEAST_TEST_TORTURE=1 FEAST_TEST_SLOW=1 {{julia}} --project=. --startup-file=no test/pkgtest.jl {{quote(filter)}}

# Run the fast moment-RII iteration smoke loop. Use test-slow for full moment research checks.
test-moment filter='moment RII':
    @just test {{quote(filter)}}

# Run the expensive moment-RII agreement/policy diagnostics explicitly.
test-moment-heavy filter='moment RII':
    @just test-slow {{quote(filter)}} moment_heavy ''

# Run only the count-driven moment-RII policy tests.
test-moment-count:
    @just test-slow 'count-driven refinement'

# Run the older simple dense FEAST contour-parallel scaling benchmark.
bench-parallel:
    {{julia}} --project=. --startup-file=no benchmark/dense_parallel_scaling.jl

# Run BenchmarkTools-backed dense serial/distributed FEAST benchmarks.
bench-dense:
    {{julia}} --project=benchmark --startup-file=no benchmark/run_dense.jl

# Run BenchmarkTools-backed sparse FEAST/UMFPACK profiling.
bench-sparse:
    {{julia}} --project=benchmark --startup-file=no benchmark/run_sparse.jl

# Run the research experiment comparing nonlinear FEAST against NEP-PACK NLEIGS.
experiment-nleigs:
    {{julia}} --project=. --startup-file=no experiments/nleigs_comparison/run.jl

# Run the higher-moment invariant-pair RII research prototype.
experiment-moment-rii:
    {{julia}} --project=. --startup-file=no experiments/moment_rii/run.jl

# Run a cheap NLEIGS comparison smoke check without the large dense problem.
experiment-nleigs-smoke:
    FEAST_EXPERIMENT_PROBLEMS=butterfly FEAST_EXPERIMENT_METHODS=feast,nleigs FEAST_EXPERIMENT_PROCS=0 FEAST_EXPERIMENT_WARMUP=false {{julia}} --project=. --startup-file=no experiments/nleigs_comparison/run.jl

# Build local documentation.
docs:
    {{julia}} --project=docs --startup-file=no docs/make.jl

# Instantiate package and docs environments.
instantiate:
    {{julia}} --project=. --startup-file=no -e 'using Pkg; Pkg.instantiate()'
    {{julia}} --project=docs --startup-file=no -e 'using Pkg; Pkg.instantiate()'
    {{julia}} --project=benchmark --startup-file=no -e 'using Pkg; Pkg.instantiate()'

# Update package and docs environments.
update:
    {{julia}} --project=. --startup-file=no -e 'using Pkg; Pkg.update()'
    {{julia}} --project=docs --startup-file=no -e 'using Pkg; Pkg.update()'
    {{julia}} --project=benchmark --startup-file=no -e 'using Pkg; Pkg.update()'

# Show package and docs dependency status.
status:
    {{julia}} --project=. --startup-file=no -e 'using Pkg; Pkg.status()'
    {{julia}} --project=docs --startup-file=no -e 'using Pkg; Pkg.status()'
    {{julia}} --project=benchmark --startup-file=no -e 'using Pkg; Pkg.status()'

# Show the repo assumptions future agents should keep in mind.
notes:
    @printf '%s\n' \
      'Use nix develop for Julia work.' \
      'The shell sets JULIA_PROJECT=@. and uses ./.julia as the first depot.' \
      'Run just smoke, just test, and just docs for the normal local loop.' \
      'Run just test REGEX [TAGS] [EXCLUDE_TAGS] to run matching TestItems through Pkg.test(test_args=...).' \
      'Tags are comma-separated; for example: just test "moment RII" "" moment_heavy.' \
      'Run just test-slow to include TestItems tagged :slow.' \
      'Regex filters do not include :slow tests; required tags are treated as explicit and may select slow tests.' \
      'Run just test-moment for the fast moment-RII loop, and just test-moment-heavy for expensive moment diagnostics.' \
      'Run just test-moment-count for count-driven moment-RII policy tests.' \
      'Run just test-torture to include flagged generated/NLEVP numerical stress tests.' \
      'test/runtests.jl is the automated TestItemRunner entrypoint.' \
      'just test uses Pkg.test(); test-only dependencies live in Project.toml extras/targets.' \
      'Run just bench-dense for BenchmarkTools-backed dense FEAST benchmarks.' \
      'Run just bench-sparse for sparse FEAST/UMFPACK profiling.' \
      'Run just experiment-nleigs for the research comparison against NEP-PACK NLEIGS.' \
      'Run just experiment-moment-rii for the higher-moment invariant-pair RII prototype.' \
      'Run just experiment-nleigs-smoke for a cheap experiment script sanity check.' \
      'Historical research scripts live under experiments/legacy_tests/ until curated.' \
      'Julia 1.10-1.12 compat should eventually be checked in CI, not by expanding this dev shell.' \
      'Optional plotting and upstream FEAST binding work should use separate environments or package extensions.'
