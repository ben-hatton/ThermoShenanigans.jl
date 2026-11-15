# ThermoShenanigans.jl — Agent Rules

## Project Overview

ThermoShenanigans.jl is an extension to Oceananigans.jl that implements generic flux and production terms for the active tracers that may be constrained by the first and second laws of thermodynamics. It also adds a generic thermodynamics, where the user can specify not only the equation of state, but also specific heat capacity, entropy, etc.

## Extension practices

- Keep style and structure in line with Oceananigans.jl, falling back on the standard Julia style guide if not applicable
- Keep new implementations to a minimum, relying as much as possible on the existing Oceananigans.jl code
- Ensure code is suitable for CPU and GPU usage

## Testing
- After a change, run only the test groups that cover the source files touched, as listed in
  `test/test_groups.jl` (e.g. `run_tests("energetics")`). Do not run the full suite after every change.
- Run the full suite (`TEST_GROUP=all`) once, when a task is finished and before giving the commit
  command. Report its result and wall time.
- Run tests in one persistent session started with `julia --project -i test/session.jl` (in a Julia
  REPL tool if one is available), calling `run_tests(...)` and `run_doctests_and_imports()`, so that
  the compiled tests are reused between runs. Start a new session only after a change to a struct or
  to the test environment.
- Doctests and the explicit-imports check run only in the full suite, or when a docstring, export
  or import changed (`TEST_DOCTESTS_AND_IMPORTS=true`, or `run_doctests_and_imports()` in the session).
- Keep test grids between 4³ and 8³ and reuse the shared test models in
  `test/utils_for_runtests.jl`, so that all test files compile one model type. A test that needs a
  new grid, float type or closure combination must say why in a comment.
- State in the reply which tests were run and which were skipped.
- Which test files cover which source files is defined in `test/test_groups.jl`, not here. When a
  source or test file is added, split, renamed or removed, update `test/test_groups.jl` in the same
  change. A test in that file checks that every file in `src/` is covered by at least one group.
- Tests that import internal (unexported) names break when those names change. When renaming or
  removing an internal function, search `test/` and `examples/` for it and update every use in the
  same change, keeping assertions and tolerances unchanged.

## Kernel Functions (GPU compatibility)

- Use `@kernel` / `@index` (KernelAbstractions.jl)
- Kernels must be **type-stable** and **allocation-free**
- Use `ifelse` — never short-circuiting `if`/`else` in kernels
- No error messages, no Models inside kernels
- Mark functions called inside kernels with `@inline`
- **Never loop over grid points outside kernels** — use `launch!`

## Type Stability & Memory

- All structs must be concretely typed
- Type annotations are for **dispatch**, not documentation
- Minimize allocation; favor inline computation
- **Never hardcode Float64**: no literal `0.0` or `1.0` in kernels or constructors.
  Use `zero(grid)`, `one(grid)`, `convert(FT, 1//2)`, or rational literals

## Imports

- Source code: explicit imports (checked by tests)
- Examples/docs: rely on `using Oceananigans`; never explicitly import exported names

## Docstrings

- Use DocStringExtensions.jl with `$(TYPEDSIGNATURES)` when the signature does not include
  default values for args and/or kwargs.
- **ALWAYS `jldoctest` blocks, NEVER plain `julia` blocks** — doctests are tested; plain blocks rot
- Include `# output` with verifiable output; prefer `show` methods over boolean comparisons
- Use unicode for math (`Δt`, `η`, `ρ`), not LaTeX — LaTeX doesn't render in the REPL

## Comments
- Comments describe the code as it is now, for a reader who has never seen an earlier version. Never refer to 
  a change, a previous implementation, or the request that prompted it: no “now”, “no longer”, “previously”, 
  “instead of”, “replaces”, “was”, “new”, “fixed”, “refactored”, “simplified”, “as requested”.
- The history of a change belongs in the commit message and the reply, not in the source.
- If a comment explains why the code is written one way and not another, state the reason as a property of the current 
  code (“cₚ/π is kept explicit so the heat flux holds for thermodynamics that specify cₚ independently”), not as a 
  comparison with what was there before.
- When editing a function, reread its existing comments and docstring and correct any that the edit made untrue.

## Model Constructors

- `grid` is positional: `NonhydrostaticModel(grid; closure=nothing)`
- `ShallowWaterModel(grid, gravitational_acceleration; ...)` — both positional
- Omit semicolon when there are no keyword arguments: `NonhydrostaticModel(grid)` not `NonhydrostaticModel(grid;)`

## Naming Conventions

- **Files**: snake_case matching the type they define — `nonhydrostatic_model.jl`
- **Types/Constructors**: PascalCase **only for true constructors** — `NonhydrostaticModel`
- **Functions**: snake_case — `time_step!`; functions that return values are never PascalCase
- **Kernels**: may prefix with underscore — `_compute_tendency_kernel`
- **Variables**: English long name or readable unicode math notation — never mix abbreviated and
  full forms (e.g., `cond` vs `condition`) to imply a difference; be specific
  
## Agent Behavior

- Prioritize type stability and GPU compatibility
- Follow established patterns in existing code
- Add tests for new functionality; update exports when adding public API
- Reference physics equations in comments when implementing dynamics
- After making a significant change, give the user a commit command with a short message,
  preceded by the `cd` into the repo root, e.g.
  `cd path/to/ThermoShenanigans.jl && git commit -am "Short message"`
- Never add co-authorship or AI attribution (e.g. `Co-Authored-By:` trailers, "Generated with Claude Code")
  to commit messages, pull requests, issues or comments
  
  
## Physics
- The package targets DNS with constant molecular coefficients; do not add support for LES or other closures with closure fields.
- Keep fluxes and production terms in their general thermodynamic form (e.g. j_θ = -κ_θ (cₚ/π) ∇T). 
  Do not substitute an identity that holds only for the current thermodynamics or definitions; if one 
  exists, state it in a comment and leave the code general.
- All thermodynamic functions take (θ, S, z, thermodynamics).

## Performance and GPU (no local GPU)
- No GPU is available during development. Never state or imply that code was run, tested or benchmarked on a GPU.
- Every performance change comes with a before/after CPU benchmark, reported in the reply. Results must match to round-off.
- Every kernel launches over the largest index set it writes; no loop over grid-sized ranges inside a kernel.
- Every public diagnostic has a GPU test in the architecture tests, gated by TEST_ARCHITECTURE=GPU, comparing against the CPU. Write it even though it cannot be run locally.
- On the CPU, every public diagnostic is checked with @inferred and zero allocations per cell, in both Float64 and Float32.
- Any code path not yet run on a GPU is listed in GPU_UNVERIFIED.md with the test that will cover it. Remove an entry only when the user reports the cluster result.
- Use only array operations already used on GPU arrays in Oceananigans or Oceanostics, and cite the file. Anything else goes in GPU_UNVERIFIED.md as a specific risk.

## Readability and structure
- One name, one function: never bind the same local name to different generated functions.
- @eval only for x/y/z triplets, with the generated names listed in a comment above the loop.
- Kernel function, docstring and constructor for a diagnostic sit together.
- Diagnostics do not validate their model; do not add model checks unless asked.
- Before writing a kernel, check whether Oceananigans or Oceanostics has it, and cite the file and version ported from.
  
