set shell := ["bash", "-cu"]

julia := env_var_or_default("JULIA", "julia")

default:
    @just --list

# Load the package. Use this after environment or module-layout changes.
smoke:
    {{ julia }} --project=. --startup-file=no -e 'using FEASTSolver; println("FEASTSolver loaded")'

# Examples:
#   just test
#   just test 'moment RII'
#   just test --slow 'count-driven refinement'
#   just test --tags moment_heavy 'moment RII'
#   just test --exclude moment_heavy 'moment RII'
#   just test --preset moment-core
#   just test --preset moment-heavy 'residual Laurent update'
#   just test --list-tests --slow 'moment RII'
#   just test --list-presets

# Run the package test harness.
test *args:
    {{ julia }} --project=. --startup-file=no test/pkgtest.jl {{ quote(args) }}

# Include test items tagged :slow.
test-slow *args:
    @just test --slow {{ quote(args) }}

# Run only flagged numerical torture tests. These are correctness tests, not benchmarks.
test-torture *args:
    @just test --only-torture {{ quote(args) }}

# Run the older simple dense FEAST contour-parallel scaling benchmark.
bench-parallel:
    {{ julia }} --project=. --startup-file=no benchmark/dense_parallel_scaling.jl

# Run BenchmarkTools-backed dense serial/distributed FEAST benchmarks.
bench-dense:
    {{ julia }} --project=benchmark --startup-file=no benchmark/run_dense.jl

# Run BenchmarkTools-backed sparse FEAST/UMFPACK profiling.
bench-sparse:
    {{ julia }} --project=benchmark --startup-file=no benchmark/run_sparse.jl

# Run BenchmarkTools-backed moment-NLFEAST experiment benchmarks.
bench-moment:
    {{ julia }} --project=benchmark --startup-file=no benchmark/run_moment.jl

# Run the research experiment comparing nonlinear FEAST against NEP-PACK NLEIGS.
experiment-nleigs:
    {{ julia }} --project=. --startup-file=no experiments/nleigs_comparison/run.jl

# Run the higher-moment invariant-pair RII research prototype.
experiment-moment-rii:
    {{ julia }} --project=. --startup-file=no experiments/moment_rii/run.jl

# Run the focused cache-native two-sided NLFEAST experiment tests.
test-fused-nlfeast:
    {{ julia }} --project=. --startup-file=no experiments/fused_nlfeast/test/runtests.jl

# Run a cheap NLEIGS comparison smoke check without the large dense problem.
experiment-nleigs-smoke:
    FEAST_EXPERIMENT_PROBLEMS=butterfly FEAST_EXPERIMENT_METHODS=feast,nleigs FEAST_EXPERIMENT_PROCS=0 FEAST_EXPERIMENT_WARMUP=false {{ julia }} --project=. --startup-file=no experiments/nleigs_comparison/run.jl

# Build local documentation.
docs:
    {{ julia }} --project=docs --startup-file=no docs/make.jl

# Instantiate package and docs environments.
instantiate:
    {{ julia }} --project=. --startup-file=no -e 'using Pkg; Pkg.instantiate()'
    {{ julia }} --project=docs --startup-file=no -e 'using Pkg; Pkg.instantiate()'
    {{ julia }} --project=benchmark --startup-file=no -e 'using Pkg; Pkg.instantiate()'

# Update package and docs environments.
update:
    {{ julia }} --project=. --startup-file=no -e 'using Pkg; Pkg.update()'
    {{ julia }} --project=docs --startup-file=no -e 'using Pkg; Pkg.update()'
    {{ julia }} --project=benchmark --startup-file=no -e 'using Pkg; Pkg.update()'

# Show package and docs dependency status.
status:
    {{ julia }} --project=. --startup-file=no -e 'using Pkg; Pkg.status()'
    {{ julia }} --project=docs --startup-file=no -e 'using Pkg; Pkg.status()'
    {{ julia }} --project=benchmark --startup-file=no -e 'using Pkg; Pkg.status()'

# Show the repo assumptions future agents should keep in mind.
notes:
    @printf '%s\n' \
      'Use nix develop for Julia work.' \
      'The shell sets JULIA_PROJECT=@. and uses ./.julia as the first depot.' \
      'Run just smoke, just test, and just docs for the normal local loop.' \
      'Run just test [OPTIONS] [REGEX] to run matching TestItems through Pkg.test(test_args=...).' \
      'Run just test --help for the full test-harness help.' \
      'Supported options: --slow, --torture, --only-torture, --tags TAGS, --exclude TAGS, --preset PRESET, --list-tests, --list-presets.' \
      'Tags are comma-separated; for example: just test --tags moment_heavy "moment RII".' \
      'Presets are moment-core, moment-heavy, moment-count, and torture.' \
      'Run just test-slow to include TestItems tagged :slow.' \
      'Regex filters do not include :slow tests unless --slow or explicit --tags are passed.' \
      'Run just test --preset moment-core for focused FEAST/SS/Beyn/NLFEAST reduction checks; it excludes :moment_heavy and :distributed.' \
      'Run just test --preset moment-heavy for expensive moment diagnostics.' \
      'Run just test --preset moment-count for count-driven moment-RII policy tests.' \
      'Run just test-torture to include flagged generated/NLEVP numerical stress tests.' \
      'test/runtests.jl is the automated TestItemRunner entrypoint.' \
      'just test uses Pkg.test(); test-only dependencies live in Project.toml extras/targets.' \
      'Run just bench-dense for BenchmarkTools-backed dense FEAST benchmarks.' \
      'Run just bench-sparse for sparse FEAST/UMFPACK profiling.' \
      'Run just bench-moment for BenchmarkTools-backed moment-NLFEAST experiment benchmarks.' \
      'Run just experiment-nleigs for the research comparison against NEP-PACK NLEIGS.' \
      'Run just experiment-moment-rii for the higher-moment invariant-pair RII prototype.' \
      'Run just experiment-nleigs-smoke for a cheap experiment script sanity check.' \
      'Historical research scripts live under experiments/legacy_tests/ until curated.' \
      'Julia 1.10-1.12 compat should eventually be checked in CI, not by expanding this dev shell.' \
      'Optional plotting and upstream FEAST binding work should use separate environments or package extensions.'
