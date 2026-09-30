# ThermoShenanigans.jl — Agent Rules

## Project Overview

ThermoShenanigans.jl is an extension to Oceananigans.jl that implements generic flux and production terms for the active tracers that may be constrained by the first and second laws of thermodynamics. It also adds a generic thermodynamics, where the user can specify not only the equation of state, but also specific heat capacity, entropy, etc.

## Extension practices

- Keep style and structure in line with Oceananigans.jl
- Keep new implementations to a minimum, relying as much as possible on the existing Oceananigans.jl code
- Ensure code is suitable for CPU and GPU usage

## Critical Rules

### Kernel Functions (GPU compatibility)

- Use `@kernel` / `@index` (KernelAbstractions.jl)
- Kernels must be **type-stable** and **allocation-free**
- Use `ifelse` — never short-circuiting `if`/`else` in kernels
- No error messages, no Models inside kernels
- Mark functions called inside kernels with `@inline`
- **Never loop over grid points outside kernels** — use `launch!`

### Type Stability & Memory

- All structs must be concretely typed
- Type annotations are for **dispatch**, not documentation
- Minimize allocation; favor inline computation
- **Never hardcode Float64**: no literal `0.0` or `1.0` in kernels or constructors.
  Use `zero(grid)`, `one(grid)`, `convert(FT, 1//2)`, or rational literals

### Imports

- Source code: explicit imports (checked by tests)
- Examples/docs: rely on `using Oceananigans`; never explicitly import exported names

### Docstrings

- Use DocStringExtensions.jl with `$(TYPEDSIGNATURES)` when the signature does not include
  default values for args and/or kwargs.
- **ALWAYS `jldoctest` blocks, NEVER plain `julia` blocks** — doctests are tested; plain blocks rot
- Include `# output` with verifiable output; prefer `show` methods over boolean comparisons
- Use unicode for math (`Δt`, `η`, `ρ`), not LaTeX — LaTeX doesn't render in the REPL

### Model Constructors

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
