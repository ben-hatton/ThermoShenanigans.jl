# Code paths not yet run on a GPU

No GPU is available during development, and no GPU test result has been recorded here yet, so every code path below
is unverified. An entry is removed only when a cluster result that covers it is reported. `benchmark/gpu.slurm` runs
the GPU tests (`TEST_ARCHITECTURE=GPU TEST_GROUP=all`) and `benchmark/benchmark.jl GPU`.

Citations of Oceananigans and Oceanostics refer to Oceananigans 0.116.1 and Oceanostics 0.17.2.

## 1. Reached by the GPU tests, awaiting the first cluster run

| Code path | Test in `test/test_architectures.jl` |
|---|---|
| One time step of a consistent model: the energy-consistent buoyancy b̃ in the w equation and the hydrostatic pressure anomaly (`z_dot_g_bᶜᶜᶠ`, `thermodynamic_buoyancy.jl`), the `ThermodynamicDiffusivity` fluxes with their wall masks (`diffusive_flux_x/y/z`, `thermodynamic_diffusivity.jl`), and the θ forcing σ_θ (`θ_productionᶜᶜᶜ`, `model_components.jl`) | `Architectures [GPU, Float64]` and `[GPU, Float32]`: θ and S against the CPU after a step from the smooth state; `GPU diagnostics match the CPU [Float64]`: u, v, w, θ and S after a step from a random state |
| Every exported diagnostic in Float64, compared with the CPU: `reference_height`; `thermodynamic_operation` of `potential_energy`, `temperature`, `exner`, `chemical_potential_analogue`, `chemical_potential` and `HeightIndependent(potential_enthalpy)`; `viscous_dissipation`, `θ_production`, `buoyancy_production`, `heating_buoyancy_production`, `nonlinear_buoyancy_production`, `entropy_production`, `potential_energy_conversion`, `buoyancy_work`, `buoyancy_tendency`, `advective_buoyancy_tendency`; `θ_flux`, `salt_flux`, `buoyancy_flux` (x, y and z); the reference state `reference_displacement`, `reference_potential`, `reference_buoyancy`, `reference_work`, `available_potential_energy`, `background_potential_energy`, `background_gravitational_energy`; the APE budget `ape_production`, `heating_ape_production`, `nonlinear_ape_production`, `ape_dissipation`, `ape_flux` (x, y, z), `ape_advection`, `available_buoyancy_work`, `spurious_ape_production`, `reference_tendency_work`; the KE budget `kinetic_energy`, `ke_advection`, `ke_flux` (x, y, z); the BPE budget `bpe_flux` (x, y, z), `bpe_advection`, `spurious_bpe_production` | `GPU diagnostics match the CPU [Float64]` (`test_architectures_match`), one testset per diagnostic, with `≈` at the default `rtol = √eps`. `spurious_bpe_production` (𝒩_ℬ), about 10⁶ times smaller than the terms it is the residual of, also has `atol = √eps ‖∇·(u ℬ)‖`. Against a CPU model whose state was perturbed by up to two ulps before the step, every difference was below 3 × 10⁻⁵ of its tolerance. `Every exported diagnostic is compared between architectures` (run on the CPU) fails if an exported diagnostic is missing from this list |
| Every exported diagnostic above in Float32, checked to be finite only. In Float32, π ≈ cₚ = 4000 J kg⁻¹ K⁻¹ is resolved to 2.4 × 10⁻⁴ J kg⁻¹ K⁻¹, larger than its horizontal differences between cells (1.5 × 10⁻⁵ in the test model), so that σ_θ, j_θ·∇π and the diagnostics built on them are dominated by round-off that differs between architectures: on the CPU, σ_θ in Float32 differs from σ_θ in Float64 by 44 % (root sum of squares) in the same state | `Architectures [GPU, Float32]` |
| The `Sorting` reference state on the device: `compute_reference_state!` and `compute_reference_tendency!` (`sorting.jl`), with the halo fill of its data arrays through Oceananigans' `fill_halo_regions!(::OffsetArray, boundary_conditions, indices, loc, grid)` (`reference_state.jl`, `fill_center_halo_regions!`; the method Oceananigans uses for every Field, `src/BoundaryConditions/fill_halo_regions.jl:20`) | the `reference_height`, `reference_*`, `available_potential_energy` and `reference_tendency_work` testsets of `GPU diagnostics match the CPU [Float64]` |

## 2. Array operations not used on GPU arrays in Oceananigans or Oceanostics

The `src/` and `ext/` directories of Oceananigans 0.116.1 and the `src/` directory of Oceanostics 0.17.2 contain no
`sort!` or `Base.cumsum!` of a device array: Oceananigans' `cumsum!` is a `Field` method that launches its own kernel
(`src/Fields/scans.jl`, `directional_accumulate!`), and its `sort!`/`sortperm` calls sort host collections
(`src/OutputReaders/combining_field_time_series.jl:479`, `src/OutputWriters/averaged_specified_times.jl:69`). These
operations in `sorting.jl` therefore rely on CUDA.jl alone. The 4³ comparison tests reach each of them, but not at
the sizes where a device sort or scan takes its multi-block path; the benchmark reaches those sizes without checking
values.

| Operation | Where | Risk |
|---|---|---|
| `sort!(b★)` on a `CuVector{FT}` | `compute_reference_state!`, `sorting.jl:209` | CUDA.jl's device sort, run on N = Nx Ny Nz values at every computation of z★ |
| `sort!(pairs; by = first)` on a `CuVector{Tuple{FT, FT}}` | `compute_reference_tendency!`, `sorting.jl:280` | CUDA.jl's sort with a `by` keyword on a tuple element type. The sort need not be stable: the order of tied buoyancies does not change Ṗ★ |
| `cumsum!(P★, b★)` and `cumsum!(Ċ, Ḃ★)` on `CuVector`s | `sorting.jl:212` and `sorting.jl:282` | CUDA.jl's parallel scan sums in a different order from the CPU's sequential sum. 𝒲★ = p★ᶜᶜᶜ - p★(z★) and R★ = Ṗ★ᶜᶜᶜ - Ṗ★(z★) are small differences of these sums, so their round-off differs between architectures, more so on large grids than in the 4³ test |
| `P★ .= h .* (P★ .- b★)` and `Ḃ★ .= last.(pairs)` | `sorting.jl:213` and `sorting.jl:281` | Broadcasts on device vectors, the second over a tuple-valued `CuVector` |
| `on_architecture(arch, fill((zero(FT), zero(FT)), N))` | `reference_tendency_workspace`, `sorting.jl:110` | A device array of tuples. Oceananigans moves `zeros(FT, …)` to the device this way (`src/OrthogonalSphericalShellGrids/distributed_zipper.jl:140`), but not arrays of tuples |
| `CartesianIndices(size(grid))` passed to a kernel, which indexes an `OffsetArray` with its `CartesianIndex`es | `_gather_buoyancy!` and `_gather_buoyancy_and_tendency!`, launched at `sorting.jl:208` and `sorting.jl:278` | Oceananigans builds `CartesianIndices` inside its launch machinery (`src/Utils/kernel_launching.jl:444, 535`) but does not pass them to kernels as arguments |

## 3. Performance

| Code path | Risk | Measured by |
|---|---|---|
| `_compute_level_reference_potential!` (`sorting.jl:176`) and `_compute_level_reference_tendency!` (`sorting.jl:247`) | Each launches Nz work-items that loop over the Nx Ny slots of their level, the second with two bisections per slot: a loop over a grid-sized range inside a kernel. At 128³ that is 128 work-items of 16384 iterations each, which leaves most of a GPU idle | `reference_height` and `reference_tendency_work` rows of `benchmark/benchmark.jl GPU` |
| Device `sort!` of N = 2²¹ values (128³) at every `compute!` of z★, R★ or 𝒩 | Its cost relative to a time step | the same rows |

## 4. Not reached by any GPU test

| Code path | Test that would cover it |
|---|---|
| Grids with Periodic directions: the wall masks are no-ops there, and the halos of z★, of the buoyancy and of p★(z★) (`reference_state.jl`, `compute!` of z★) are filled periodically. Reached by the benchmark, on (Periodic, Periodic, Bounded), without checking values | `comparison_models` on a (Periodic, Periodic, Bounded) grid |
| Grids with a Flat direction (2D models, as in the doctests and `examples/two_layer_cabbeling/two_layer_cabbeling.jl`) | `comparison_models` on a (Periodic, Flat, Bounded) grid |
| Grids larger than 4³ for any value check: device sort and scan beyond a single block, bisection (`count_sorted_slots`) over large sorted arrays. Reached by the benchmark at 64³ and 128³ without checking values | z★, `reference_work` and `reference_tendency_work` compared between GPU and CPU on a 32³ random state |
| `ConstantSalinityThermodynamicBuoyancy` (`constant_salinity = S₀`): `get_temperature_and_salinity` (`thermodynamic_buoyancy.jl:63`), the zero `salt_gradient_*` (`thermodynamic_diffusivity.jl:102`), `salt_entropy_production_*` (`production.jl:74`), `salt_advection_buoyancy_tendencyᶜᶜᶜ` (`ape_budget.jl:125`) and `salt_advection_potential_energy_tendencyᶜᶜᶜ` (`bpe_budget.jl:127`) | `GPU diagnostics match the CPU` for a θ-only model built with `constant_salinity` |
| Advection schemes other than `Centered(order = 2)` in the diagnostics that use the model's advection: `advective_buoyancy_tendency`, `ape_advection`, `bpe_advection`, `spurious_ape_production`, `spurious_bpe_production`, `ke_advection`, and Ḃ in `reference_tendency_work` | `GPU diagnostics match the CPU` for a model with `WENO()` tracer and momentum advection |
| Flux boundary conditions on θ and S, and passive tracers, whose `ThermodynamicDiffusivity` flux is the zero `tracer_flux_*(…, ::Val, …)` (`thermodynamic_diffusivity.jl`) | a time step of `flux_bc_model` (tracers θ, S, c) on the GPU, compared with the CPU |
| `consistent = false` model components (`ScalarDiffusivity` κ with `ThermodynamicBuoyancy`) | a time step of a `consistent = false` model on the GPU, compared with the CPU |
| `cell_diffusion_timescale` of `ThermodynamicDiffusivity`, used by `TimeStepWizard(; diffusive_cfl)` and `DiffusiveCFL` (`thermodynamic_diffusivity.jl:55`). Called by `examples/two_layer_cabbeling/gpu`, which is not a test | `DiffusiveCFL(Δt)(model)` on the GPU, compared with the CPU |
| `thermodynamic_operation` of `∂T∂θ`, `∂T∂S`, `heat_capacity`, `isothermal_∂b∂S`, `isothermal_∂μ∂S` and of `HeightIndependent` wrappers other than `potential_enthalpy`. Each compiles its own kernel; several are evaluated inside the compared diagnostics (cₚ, T_θ and T_S in j_θ; (∂b/∂S)_T and (∂μ/∂S)_T in Γ_S and σ_η) | add them to `operation_functions` in `test_architectures.jl` |
| Thermodynamics other than `CabbelingBoussinesqThermodynamics`. The diagnostics are generic, but a user subtype of `AbstractBoussinesqThermodynamics` must be an isbits type to be passed to a kernel | `GPU diagnostics match the CPU` for an isbits test thermodynamics such as `CoupledTestThermodynamics` |
| Float32 values on the GPU (only finiteness is checked; see section 1) | a comparison of Float32 GPU values with Float64 CPU values, with tolerances set by the round-off of π and m |
| Reductions and output of the diagnostics on the GPU: `Integral`, `Average` and `Field` reductions of the reference-state and budget diagnostics, and output writers of the `reference_potential` and `reference_buoyancy` Fields, whose data is shared with the `ReferenceState`. `examples/two_layer_cabbeling/gpu` integrates ε and σ_η only, and is not a test | `Integral` of `available_potential_energy` and `reference_tendency_work` compared between GPU and CPU |
| Distributed architectures. `Sorting` would sort each rank's cells only, and `validate_reference_state_grid` does not reject a distributed grid, since `XYZRegularRG` does not restrict the architecture (`src/Grids/rectilinear_grid.jl:53`). This is unsupported on any architecture, rather than unverified on GPUs | a test that `reference_height` throws an `ArgumentError` on a distributed grid |
