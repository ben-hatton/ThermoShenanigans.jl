# Plan: energy tendencies ∂ₜ𝒦, ∂ₜ𝒫, ∂ₜ𝒜, ∂ₜℬ

**Goal.** Add four exported diagnostics, `ke_tendency`, `potential_energy_tendency`, `ape_tendency` and `bpe_tendency`.
They give the left-hand sides of the existing local budgets, built from the model's prognostic tendencies. Add tests
showing that every budget closes in every cell.

Work through the steps in order. Do not start a step until the tests listed for the previous step pass. If a test
fails, do not loosen its tolerance and do not change the identity it checks. Stop and report the failure, with the
largest residual and the cell where it occurs.

## Rules

- Follow `CLAUDE.md`, in particular the Comments section. Every comment and docstring describes the code as it is.
  None of them mentions this change. That includes the `ReferenceTendency` docstring after step 4.
- Copy the pattern of the neighbouring diagnostics:
  - the kernel function, docstring and constructor sit together;
  - build operations with `center_operation`;
  - pass field data (`field_data`, `.data`), not `Field`s. Pass a `Field` only where computing the operation must
    compute it: the reference-tendency `Field` of steps 5 and 6 and the pressure `Field` of step 3. Pass z★ as
    `z★.data` wherever another argument already computes it, as `ape_data_operation` does.
  - Refuse Fickian models by dispatch on `ThermodynamicDiffusivity`, as `potential_energy(model)` does. The
    `MethodError` must then have `f` equal to the exported function and `args[1]` a `ScalarDiffusivity`, as
    `test_refused` expects.
- Docstrings: `$(TYPEDSIGNATURES)`, then one sentence giving the definition, then the budget identity as an indented
  block, as in `spurious_ape_production`. No doctests. Each of the four docstrings also says:

  > It reads the model's tendencies `Gⁿ`, which `time_step!`, `set!` and `update_state!` compute for the current
  > state; it is valid after any of them.

  The `ke_tendency` docstring adds:

  > p_NHS is that of the model's last pressure correction, as in [`ke_flux`](@ref): after `set!`, the potential of
  > the projection of the velocity, and after `time_step!`, the pressure of its last substep. It differs from the
  > pressure of `Gⁿ` by a flux divergence, so that ∫ ∂ₜ𝒦 dV is exact and the budget closes in every cell.

- Use only the names in the "New names" table. If another helper seems necessary, stop and ask.
- Do not:
  - add types, keyword arguments or model checks;
  - change examples, the README or Oceananigans' handling of boundary conditions;
  - call `fill_halo_regions!` or `compute_tendencies!` on model fields inside a diagnostic;
  - add momentum `FluxBoundaryCondition` tendencies (F_u). The `ke_tendency` docstring says, in one sentence, that
    they are not included;
  - share R★ between `ape_tendency` and `bpe_tendency`: each builds its own reference tendency, as
    `reference_tendency_work` does.
- No benchmark. The `CLAUDE.md` rule covers performance changes, whose results must match to round-off. The pressure
  `Field` is a correctness change, and it changes the results of `ke_flux` next to a periodic boundary. In the final
  reply, say that it adds one kernel and one halo fill per computation of `ke_flux` and `ke_tendency`.

## Facts this plan relies on (checked in Oceananigans 0.116.1)

Cite these in comments where the code depends on them. Do not re-derive them. Oceanostics citations refer to v0.21.7,
as in `GPU_UNVERIFIED.md`, and were checked against its source; the installed Oceanostics may be older, so do not
re-check them against it.

| Fact | Where |
|---|---|
| `update_state!` ends with `compute_tendencies!`, so `Gⁿ` is current after it | `src/Models/NonhydrostaticModels/update_nonhydrostatic_model_state.jl:60` |
| `time_step!` (RK3) ends with `update_state!` | `src/TimeSteppers/runge_kutta_3.jl:172` |
| `set!` projects the velocity with Δt = 1, then calls `update_state!` | `src/Models/NonhydrostaticModels/set_nonhydrostatic_model.jl:63-70` |
| The flux of a `FluxBoundaryCondition` (F) is added to `Gⁿ` only at the start of the next substep, so `Gⁿ` after `update_state!` does not contain F | `src/Models/NonhydrostaticModels/nonhydrostatic_rk3_substep.jl:35` |
| After a step, `pNHS` holds p in the interior but p·Δτ in the halos, because `_divide_by!` runs over the interior only | `src/Models/NonhydrostaticModels/pressure_correction.jl:103-106` |
| Computing a `KernelFunctionOperation` computes its arguments. A `Tuple` or `NamedTuple` argument falls through the generic `compute_at!` to `compute!`, which computes the `Field`s inside it | `src/AbstractOperations/kernel_function_operation.jl:112`, `src/Fields/field.jl:521` and `:480` |

## Definitions

- G = `model.timestepper.Gⁿ`.
- F = the tendency of θ or S from its `FluxBoundaryCondition`s only. It is **not** `wall_θ_flux_*`, which also
  contains a Fickian closure's Value/Gradient wall flux. That flux is already in G, so using it would count it twice.
- The prognostic tendencies:
  - ∂ₜθ = G_θ + F_θ
  - ∂ₜS = G_S + F_S, which is zero for `ConstantSalinityThermodynamicBuoyancy`
  - ∂ₜ𝐮 = G_𝐮 − ∇p_NHS, with `∂xᶠᶜᶜ`, `∂yᶜᶠᶜ`, `∂zᶜᶜᶠ`. G already holds −∇p_HY′ in u and v.
  - ḃ = b_θ ∂ₜθ + b_S ∂ₜS

| Diagnostic | Expression | Fickian |
|---|---|---|
| `ke_tendency(model)` | ∂ₜ𝒦 = ℑx(u ∂ₜu) + ℑy(v ∂ₜv) + ℑz(w ∂ₜw) | supported |
| `potential_energy_tendency(model)` | ∂ₜ𝒫 = π ∂ₜθ + m ∂ₜS | refused |
| `ape_tendency(model, z★)` | ∂ₜ𝒜 = Υ ḃ + R★[ḃ] | supported |
| `bpe_tendency(model, z★)` | ∂ₜℬ = (π − Υ b_θ) ∂ₜθ + (m − Υ b_S) ∂ₜS − R★[ḃ] | refused |

Here R★[ḃ] is the reference term of `reference_tendency_work`, formed from ḃ instead of the diagnosed Ḃ.

## New names (complete list)

| File | Name | Signature / body |
|---|---|---|
| `thermodynamic_diffusivity.jl` | `area_boundary_condition_flux_{x,y,z}` | `(i, j, k, grid, bcs, c, clock, model_fields)` = face area (`Axᶠᶜᶜ`, `Ayᶜᶠᶜ`, `Azᶜᶜᶠ`) × `boundary_condition_flux_{x,y,z}`. A separate `@eval` loop directly after `boundary_condition_flux_z`, with the generated names listed in a comment above it, written like the `area_buoyancy_flux_{x,y,z}` loop (`ape_budget.jl:158-164`). Do not add them to the large loop of that file. |
| `diagnostics.jl` | `wall_flux_tendencyᶜᶜᶜ` | `(i, j, k, grid, bcs, c, clock, model_fields)` = −(δxᶜᵃᵃ + δyᵃᶜᵃ + δzᵃᵃᶜ of the area fluxes) / `Vᶜᶜᶜ`. Cite `compute_flux_bcs.jl:113-165`. |
| `diagnostics.jl` | `model_tendencies(model)` | `field_data(model.timestepper.Gⁿ)` |
| `diagnostics.jl` | `θ_tendencyᶜᶜᶜ` | `(i, j, k, grid, tendencies, walls)` = `tendencies.θ[i, j, k]` + `wall_flux_tendencyᶜᶜᶜ(…, walls.θ, walls.fields.θ, walls.clock, walls.fields)`. The test helper `θ_tendency(model)` (`utils_for_runtests.jl:265`), which returns G_θ without F, stays as it is. |
| `diagnostics.jl` | `salt_tendencyᶜᶜᶜ` | `(i, j, k, grid, buoyancy_formulation, tendencies, walls)`, as above for S; a `::ConstantSalinityThermodynamicBuoyancy` method returns `zero(grid)` |
| `diagnostics.jl` | `prognostic_buoyancy_tendencyᶜᶜᶜ` | `(i, j, k, grid, buoyancy_formulation, fields, tendencies, walls)` = ḃ, with b_θ and b_S from `thermodynamic_functionᶜᶜᶜ` |
| `diagnostics.jl` | `potential_energy_tendencyᶜᶜᶜ`, `potential_energy_tendency` | same arguments as ḃ; π and m from `thermodynamic_functionᶜᶜᶜ(…, exner/chemical_potential_analogue, …)`. Place them directly after `potential_energy`. |
| `ke_budget.jl` | `nonhydrostatic_pressureᶜᶜᶜ`, `nonhydrostatic_pressure_field(model)` | `Field(center_operation(nonhydrostatic_pressureᶜᶜᶜ, grid, model.pressures.pNHS.data))`. Cite Oceanostics v0.21.7 `src/KineticEnergyEquation.jl:451-459`. The `_field` suffix keeps the function apart from the argument name `nonhydrostatic_pressure` of `pressureᶜᶜᶜ`. |
| `ke_budget.jl` | `u_tendency_workᶠᶜᶜ`, `v_tendency_workᶜᶠᶜ`, `w_tendency_workᶜᶜᶠ` | `(i, j, k, grid, velocities, tendencies, nonhydrostatic_pressure)` = u (G_u − ∂xᶠᶜᶜ p), written like `u_advective_workᶠᶜᶜ`. Cite Oceanostics v0.21.7 `uᵢGᵢᶜᶜᶜ`, `src/KineticEnergyEquation.jl:103-123`, which omits the pressure term. |
| `ke_budget.jl` | `ke_tendencyᶜᶜᶜ`, `ke_tendency(model)` | the sum of the three works, each interpolated to the cell centre, written like `ke_advectionᶜᶜᶜ` |
| `reference_state.jl` | `reference_tendency(z★, buoyancy_tendency_function, arguments)` | new method; see step 4 |
| `ape_budget.jl` | `prognostic_reference_tendency(model, z★)` | `reference_tendency(z★, prognostic_buoyancy_tendencyᶜᶜᶜ, (model.buoyancy.formulation, field_data(model.tracers), model_tendencies(model), wall_conditions(model)))` |
| `ape_budget.jl` | `ape_tendencyᶜᶜᶜ`, `ape_tendency` | see step 5 |
| `bpe_budget.jl` | `bpe_tendencyᶜᶜᶜ`, `bpe_tendency` | see step 6 |

All kernel functions are `@inline`. Read array data with `@inbounds`.

## Steps, each with its tests

The test commands below follow `CLAUDE.md` (one persistent session, `revise()` before each run).

### 1. Wall-flux tendency

- Add `area_boundary_condition_flux_*` and `wall_flux_tendencyᶜᶜᶜ` as in the table above.
- Add `θ_tendencyᶜᶜᶜ`, `salt_tendencyᶜᶜᶜ`, `prognostic_buoyancy_tendencyᶜᶜᶜ` and `model_tendencies`. Put them in
  `diagnostics.jl`, after `wall_conditions`.
- **Test, in `test_production.jl`.** For `flux_bc_model` and `fickian_wall_model`:
  1. Call `set_random_state!(model)`.
  2. Compute `center_operation(θ_tendencyᶜᶜᶜ, grid, model_tendencies(model), wall_conditions(model))`, and the same for
     S, into arrays.
  3. Call `compute_flux_bc_tendencies!(model)`.
  4. Assert that the arrays equal `interior(Gⁿ.θ)` and `interior(Gⁿ.S)`, with rtol = 1e-14.

  This is the only direct test of F, so do not skip it. Import the internal names in the test file. After step 3,
  `Gⁿ` holds F, so do not reuse the model without calling `update_state!`.
- Run `run_tests("diffusivity", "production", "model_components")`.

### 2. ∂ₜ𝒫

- Add `potential_energy_tendency` with its Fickian dispatch.
- Rename the local `potential_energy_tendency` in `test_production.jl:32` to `integrated_pe_tendency`, since it would
  shadow the export. Keep its assertion unchanged.
- No new test yet; steps 6 and 7 test it.
- Run `run_tests("production", "model_components")`.

### 3. ∂ₜ𝒦 and the pressure halos

- Add `nonhydrostatic_pressure_field`.
- Change `pressure_data(model)` to `(nonhydrostatic_pressure_field(model), maybe_data(model.pressures.pHY′))`. The
  kernels of `ke_flux` stay as they are: `pressureᶜᶜᶜ` indexes the `Field` like data. A comment above `pressure_data`
  explains:
  - p_NHS is read through a computed `Field` because the halos of `pNHS` hold p·Δτ after a time step (cite);
  - computing the diagnostic computes the `Field` (cite `kernel_function_operation.jl:112` and `field.jl:521`, `:480`).
- Add the tendency works and `ke_tendency(model) = center_operation(ke_tendencyᶜᶜᶜ, model.grid,
  field_data(model.velocities), model_tendencies(model), nonhydrostatic_pressure_field(model))`.
- **Write the halo test first, with `ke_flux` only, and check that it fails before the `pressure_data` change.** Add
  `ke_tendency` to it once `ke_tendency` exists.
  - Put it in `test_ke_budget.jl`. Use a 6³ grid with topology `(Periodic, Bounded, Bounded)`. Add a comment that it is
    the only periodic grid because only a periodic direction reads the halos of p_NHS.
  - Build `component_model`, call `set_random_state!` and take one `time_step!(model, 1e-3)`.
  - Assert that the x halo of `pNHS` differs from the periodic copy of the interior. This proves the test is not
    vacuous.
  - Compute `ke_tendency` and `ke_flux(model, z★).x`.
  - Then call `fill_halo_regions!(model.pressures.pNHS)` and recompute both.
  - Assert equality, with rtol = 1e-14.
- **Update `test_ke_budget.jl`.**
  - Rename the local `ke_tendency` array to `expected_ke_tendency`, because it would otherwise shadow the export.
  - Keep it as an independent reference, including the random-p_NHS case.
  - Add `@test computed(ke_tendency(model)) ≈ expected_ke_tendency rtol = 1e-12`.
  - Keep the existing budget assertion, with `budget` compared with `computed(ke_tendency(model))`.
- **One-step check for ∂ₜ𝒦**, in `test_ke_budget.jl`. 𝒦 is the only energy without a finite-difference check of its
  budget (𝒜 and ℬ have one, in `test_budgets_close`), and its pressure term needs a real time step.
  1. Build `component_model(test_grid(), cabbeling)` and call `set_random_state!`.
  2. Take one warm-up `time_step!(model, Δt)` with Δt = 1e-5, so that `pNHS` holds the pressure of a time step and
     not the projection potential of `set!`. If a test comment explains this, it says only that `pNHS` holds the
     pressure of the last substep of the warm-up step.
  3. Compute 𝒦 and ∂ₜ𝒦 into arrays, take one more `time_step!(model, Δt)`, and compute 𝒦 again.
  4. Assert max |(𝒦ⁿ⁺¹ − 𝒦ⁿ)/Δt − ∂ₜ𝒦| < 1e-3 max |∂ₜ𝒦|, as in Oceanostics v0.21.7
     `test/test_kinetic_energy_equation.jl:381-415`.
- Run `run_tests("ke_budget")`.

### 4. Generalised reference tendency

- In `ReferenceTendency`, replace the fields `advection`, `clock`, `fields`, `parameters` and `walls` with
  `buoyancy_tendency_function` and `arguments`. Keep `reference_height`, `buoyancy_tendency`, `reference_tendency` and
  `workspace`. `sorting.jl` reads only those four, at lines 277-296.
- `_compute_buoyancy_tendency!(buoyancy_tendency, grid, buoyancy_tendency_function, arguments)` sets the value to
  `buoyancy_tendency_function(i, j, k, grid, arguments...)`.
- Add `reference_tendency(z★, buoyancy_tendency_function, arguments)`, which builds the `Field`.
  `reference_tendency(model, z★)` calls it with `total_buoyancy_tendencyᶜᶜᶜ` and `(model.advection, model.clock,
  velocities_and_tracers(model), production_parameters(model), wall_conditions(model))`. R★ of the existing budget is
  then unchanged.
- Update the docstrings of `ReferenceTendency` and `reference_tendency`. Search `test/` and `examples/` for the
  removed field names and update every use.
- **Start a new test session**, because a struct changed. Run `run_tests("reference_state", "ape_budget")`. The results
  must be unchanged.

### 5. ∂ₜ𝒜

- Add `prognostic_reference_tendency`.
- Add `ape_tendency(model, z★)`:
  - `parcel_tendency = prognostic_reference_tendency(model, z★)`.
  - Return `center_operation(ape_tendencyᶜᶜᶜ, model.grid, z★.data, parcel_tendency.operand.buoyancy_tendency,
    parcel_tendency.operand.reference_tendency, parcel_tendency)`. Computing `parcel_tendency` computes z★, so z★
    itself is not an argument: it would be sorted twice.
  - `ape_tendencyᶜᶜᶜ` = `reference_displacementᶜᶜᶜ` × `buoyancy_tendency[i, j, k]` + `reference_workᶜᶜᶜ(i, j, k, grid,
    reference_tendency, parcel_tendency)`.

  It reads ḃ from the buffer that computing `parcel_tendency` fills. Υ ḃ and R★ then use the same ḃ, and ḃ is
  evaluated once per cell. Say so in a comment.
- **Update `test_budgets_close` in `test_bpe_budget.jl`.** It already builds the APE budget (`budget`) and checks it
  against a finite difference of 𝒜 along the model's tendencies, for every model: consistent, Fickian, Flux
  conditions, Fickian wall conditions and equal heat fluxes. Asserting the new diagnostic against `budget` there ties it
  to that finite difference.
  - Directly after `z★ = reference_height(model)` (line 16), add `ape_tendency_values = computed(ape_tendency(model, z★))`.
  - Move lines 12-14 (`compute_flux_bc_tendencies!(model)` and the reading of `G`, `Gθ` and `GS`) together, to just
    after the new diagnostics of steps 5 and 6. The diagnostics then read `Gⁿ` before F is added (they add F
    themselves), while the finite difference still uses `Gθ` and `GS` with F.
  - Add `@test ape_tendency_values ≈ budget rtol = 1e-8` next to the existing assertions.
  - Apart from these lines and those of step 6, leave the function unchanged.
- Run `run_tests("ape_budget")`.

### 6. ∂ₜℬ

- Add `bpe_tendency(model, z★)` with its Fickian dispatch. It takes the same arguments as `ape_tendency`, `z★.data`
  included, plus the buoyancy formulation, the tracer data, `model_tendencies(model)` and `wall_conditions(model)`.
- Above `bpe_tendencyᶜᶜᶜ`, add a comment in the words of `bpe_flux_*`:

  > π − Υ b_θ = π(θ, S, z★) because b does not depend on z; the general form is kept for thermodynamics in which b
  > depends on z.

- **Update `test_budgets_close`.**
  - Directly after `ape_tendency_values`, and before the moved lines 12-14, add a separate block:

    ```julia
    if bpe
        bpe_tendency_values = computed(bpe_tendency(model, z★))
        pe_tendency_values = computed(potential_energy_tendency(model))
    end
    ```
  - In the existing `if bpe` block of the assertions (line 52), assert that `bpe_tendency_values` equals `bpe_budget`,
    with atol = 1e-12 × `maximum(abs, pe_tendency_values)`. π ≈ cₚ in ∂ₜ𝒫 sets the round-off scale. This ties
    `bpe_tendency` to the existing finite difference of ℬ.
- **Update the "PE budget splits" testset** (`test_bpe_budget.jl:131-160`).
  - Rename its locals `ape_tendency` and `bpe_tendency` to `ape_budget` and `bpe_budget`, since they would shadow the
    exports. Keep its assertions unchanged.
  - Move `z★ = reference_height(model)` (line 141) up, to just before `compute_flux_bc_tendencies!(model)` (line 139).
    Between the two, compute `potential_energy_tendency`, `ape_tendency` and `bpe_tendency` into arrays.
  - Assert that ∂ₜ𝒫 ≈ `pe_tendency` (rtol = 1e-12) and that ∂ₜ𝒫 = ∂ₜ𝒜 + ∂ₜℬ, with the atol above.
- **Refusals.** Add `test_refused(potential_energy_tendency, () -> potential_energy_tendency(model))` to the
  "Fickian model: zero θ production, and refusals" testset (`test_production.jl:78-82`), and
  `test_refused(bpe_tendency, () -> bpe_tendency(model, z★))` to "Fickian model refuses the consistent-only
  diagnostics" (`test_bpe_budget.jl:113-116`). Adding a name to `fickian_refused` only skips it in the loops; these
  calls check that it throws.
- Run `run_tests("bpe_budget")`.

### 7. Lists, exports, documentation and energy conservation

- **Exports** in `ThermoShenanigans.jl`:
  - add `potential_energy_tendency` after `internal_energy` under "Thermodynamic diagnostics";
  - add `ape_tendency` at the end of "Local APE budget";
  - add `ke_tendency` and `bpe_tendency` at the end of "Local KE and BPE budgets".
- **Docstring of `thermodynamic_model_components`:**
  - add `ke_tendency` and `ape_tendency` to "supported";
  - add `potential_energy_tendency` and `bpe_tendency` to "refused".
- **`test/utils_for_runtests.jl`:**
  - add `ke_tendency` and `potential_energy_tendency` to `model_diagnostics`;
  - add `ape_tendency` and `bpe_tendency` to `reference_diagnostics`;
  - add `"potential_energy_tendency"` and `"bpe_tendency"` to `fickian_refused`.

  The existing loops then cover the CPU/GPU comparison, the Fickian status test, and `@inferred` with zero
  allocations in Float64 and Float32 (`test_bpe_budget.jl:248-262`).
- **`test_energy_conservation.jl`.** In "Semi-discrete energy identity", for every case, and in the boundary-flux
  testset:
  - compute `ke_tendency` and `potential_energy_tendency` into arrays **before** calling `energy_tendency`, which adds
    F into `Gⁿ`;
  - then assert that Σ V (∂ₜ𝒦 + ∂ₜ𝒫) equals `dEdt`, and so equals 0 or `wall_input(π, m, grid)`, with atol =
    1e-13 × `round_off`.
- **`GPU_UNVERIFIED.md`**, section 1 ("Reached by the GPU tests"):
  - add the four diagnostics to the "every exported diagnostic" row;
  - add one row each for the paths below, with these covering tests:
    - `Gⁿ` data read inside a `KernelFunctionOperation`: `GPU diagnostics match the CPU [no flux, Float64]`;
    - `wall_flux_tendencyᶜᶜᶜ`: `GPU diagnostics match the CPU [Flux conditions, Float64]` and
      `GPU Fickian diagnostics match the CPU [wall conditions, Float64]`;
    - the computed pressure `Field` inside the arguments of `ke_flux` and `ke_tendency`: the `ke_flux` and
      `ke_tendency` testsets of `GPU diagnostics match the CPU [no flux, Float64]`, on the Bounded test grid. State that
      no GPU test reaches a periodic grid, where the halo fill is Oceananigans' own `fill_halo_regions!`;
    - the function-and-arguments `_compute_buoyancy_tendency!`: the `reference_tendency_work`, `ape_tendency` and
      `bpe_tendency` testsets.
- Run `run_tests("buoyancy", "diffusivity", "model_components", "production")` and `run_doctests_and_imports()`.

## Finish

- Run the full suite once (`run_tests("all")`). Report its result and wall time.
- In the reply:
  - list the tests that were run and the tests that were skipped;
  - give the largest residual of each new closure test and of the one-step check, as a fraction of its tolerance;
  - state the cost of the pressure `Field` (see Rules);
  - give the commit command, with no attribution.
