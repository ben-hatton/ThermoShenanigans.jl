# KernelFunctionOperation at (Center, Center, Center) of f(i, j, k, grid, args...)
center_operation(f, grid, args...) = KernelFunctionOperation{Center, Center, Center}(f, grid, args...)

# x, y and z components of a face flux, at (Face, Center, Center), (Center, Face, Center) and (Center, Center, Face)
face_operations(f_x, f_y, f_z, grid, args...) = (x = KernelFunctionOperation{Face, Center, Center}(f_x, grid, args...),
                                                 y = KernelFunctionOperation{Center, Face, Center}(f_y, grid, args...),
                                                 z = KernelFunctionOperation{Center, Center, Face}(f_z, grid, args...))

#=
The diagnostics read the data of the model's fields, rather than the Fields themselves: a Field's type carries its
grid and boundary conditions, about 2000 characters, and the type of every diagnostic, and of every output writer or
reduction built on it, carries the types of its arguments. Compile times grow with the size of these types; within
a kernel, a Field is reduced to its data anyway.
=#
field_data(fields) = map(field -> field.data, fields)

# The data of the model fields read by the diagnostics
velocities_and_tracers(model) = field_data(merge(model.velocities, model.tracers))

#=
The arguments from which the diagnostics evaluate the wall fluxes of θ and S as `compute_flux_bc_tendencies!` does
(Oceananigans 0.116.1, `src/Models/NonhydrostaticModels/compute_nonhydrostatic_tendencies.jl:177-195`): the boundary
conditions of θ and S on the sides of the domain, as the model has regularised them, and the model's clock and the data
of `fields(model)`. Field-dependent boundary functions index the model fields by position
(`field_dependencies_indices`, `src/BoundaryConditions/continuous_boundary_function.jl:44`), so that these are all of
`fields(model)`, auxiliary fields included, in its order. The sides are held in a `NamedTuple`, which a kernel reads by
name: a `FieldBoundaryConditions` adapts to the GPU as a tuple of its ordered conditions. The constant-salinity model
has no S.
=#
wall_conditions(model) = (θ = side_boundary_conditions(model.tracers.θ), S = side_boundary_conditions(get(model.tracers, :S, nothing)),
                          clock = model.clock, fields = field_data(Oceananigans.fields(model)))

function side_boundary_conditions(field)
    bcs = field.boundary_conditions
    return (; bcs.west, bcs.east, bcs.south, bcs.north, bcs.bottom, bcs.top)
end
side_boundary_conditions(::Nothing) = nothing

#=
F, the tendency of a tracer with data c by the fluxes of its `FluxBoundaryCondition`s `bcs`, minus the divergence of the
wall fluxes, as `compute_flux_bc_tendencies!` adds it to the boundary cell (Oceananigans 0.116.1,
`src/BoundaryConditions/compute_flux_bcs.jl:113-165`): + A Q / V at the west, south and bottom walls and - A Q / V at the
east, north and top walls. It is zero away from walls.
=#
@inline wall_flux_tendencyᶜᶜᶜ(i, j, k, grid, bcs, c, clock, model_fields) =
    - (δxᶜᵃᵃ(i, j, k, grid, area_boundary_condition_flux_x, bcs, c, clock, model_fields) +
       δyᵃᶜᵃ(i, j, k, grid, area_boundary_condition_flux_y, bcs, c, clock, model_fields) +
       δzᵃᵃᶜ(i, j, k, grid, area_boundary_condition_flux_z, bcs, c, clock, model_fields)) / Vᶜᶜᶜ(i, j, k, grid)

#=
The model's tendencies Gⁿ, which `update_state!` computes for the current state, as it ends with `compute_tendencies!`
(Oceananigans 0.116.1, `src/Models/NonhydrostaticModels/update_nonhydrostatic_model_state.jl:60`). They do not
contain F: the flux of a `FluxBoundaryCondition` is added to Gⁿ only at the start of the next substep
(`src/Models/NonhydrostaticModels/nonhydrostatic_rk3_substep.jl:35`). `time_step!` ends with `update_state!`
(`src/TimeSteppers/runge_kutta_3.jl:172`) and `set!` projects with Δt = 1
(`src/Models/NonhydrostaticModels/set_nonhydrostatic_model.jl:63-70`), so Gⁿ holds these tendencies after either.
=#
model_tendencies(model) = field_data(model.timestepper.Gⁿ)

# Reads Gⁿ instead of recomputing Oceananigans' `tracer_tendency`
# ∂ₜθ = G_θ + F_θ, with `walls` = `wall_conditions(model)`
@inline θ_tendencyᶜᶜᶜ(i, j, k, grid, tendencies, walls) =
    @inbounds tendencies.θ[i, j, k] + wall_flux_tendencyᶜᶜᶜ(i, j, k, grid, walls.θ, walls.fields.θ, walls.clock, walls.fields)

# Reads Gⁿ instead of recomputing Oceananigans' `tracer_tendency`
# ∂ₜS = G_S + F_S, zero for constant salinity
@inline salt_tendencyᶜᶜᶜ(i, j, k, grid, buoyancy_formulation, tendencies, walls) =
    @inbounds tendencies.S[i, j, k] + wall_flux_tendencyᶜᶜᶜ(i, j, k, grid, walls.S, walls.fields.S, walls.clock, walls.fields)

@inline salt_tendencyᶜᶜᶜ(i, j, k, grid, ::ConstantSalinityThermodynamicBuoyancy, tendencies, walls) = zero(grid)

# No equivalent in Oceananigans or Oceanostics
# ḃ = b_θ ∂ₜθ + b_S ∂ₜS, the buoyancy tendency of the prognostic tendencies of θ and S
@inline prognostic_buoyancy_tendencyᶜᶜᶜ(i, j, k, grid, buoyancy_formulation, fields, tendencies, walls) =
    thermodynamic_functionᶜᶜᶜ(i, j, k, grid, ∂b∂θ, buoyancy_formulation, fields) *
    θ_tendencyᶜᶜᶜ(i, j, k, grid, tendencies, walls) +
    thermodynamic_functionᶜᶜᶜ(i, j, k, grid, ∂b∂S, buoyancy_formulation, fields) *
    salt_tendencyᶜᶜᶜ(i, j, k, grid, buoyancy_formulation, tendencies, walls)

# The parameters of the production terms in `production.jl` for the closure and buoyancy of a model, built as for its
# θ forcing σ_θ
production_parameters(model) = production_parameters(model.closure, model.buoyancy.formulation)

#=
One function per diagnostic, after Oceanostics.jl: each returns a `KernelFunctionOperation` (or a `NamedTuple` of
the x, y and z components of a face flux), so that only the diagnostics asked for are compiled. Wrap them in `Field`
to compute them. All take a consistent model, built with `thermodynamic_model_components` (`consistent = true`) or by
hand with the θ forcing of `θ_production_forcing`, and all but those that depend on h⁰ and z₀ or on the consistent fluxes
also take a Fickian model (`consistent = false`), whose closure is a single `ScalarDiffusivity`. The refused ones
forward on the host to a method typed on the model's diffusivity, which exists only for a `ThermodynamicDiffusivity`.
=#

"""
$(TYPEDSIGNATURES)

Return the `KernelFunctionOperation` at (Center, Center, Center) of the thermodynamic function `f(θ, S, z,
thermodynamics)` of `model`, such as `potential_energy` (𝒫 = h⁰ - b (z - z₀) + g₀ z₀), `temperature` (T), `exner` (π),
`chemical_potential_analogue` (m), `chemical_potential` (μ) or `potential_enthalpy`, the internal energy ℐ = h⁰.

```jldoctest
julia> using Oceananigans, ThermoShenanigans

julia> grid = RectilinearGrid(size = (4, 4), x = (0, 1), z = (-1, 0), topology = (Periodic, Flat, Bounded));

julia> thermodynamics = CabbelingBoussinesqThermodynamics(thermal_expansion = 2e-4, haline_contraction = 0.8,
                                                          reference_temperature = 283, reference_salinity = 0.035,
                                                          reference_heat_capacity = 4000);

julia> components = thermodynamic_model_components(grid, thermodynamics; viscosity = 1e-6, thermal_diffusivity = 1e-7,
                                                   salt_diffusivity = 1e-9);

julia> model = NonhydrostaticModel(grid; tracers = (:θ, :S), components...);

julia> set!(model, θ = 283, S = 0.035);

julia> T = Field(thermodynamic_operation(model, temperature));

julia> round(T[1, 1, 4], digits = 6) # T = θ at z₀ = 0, slightly warmer at the top cell centre, z = -0.125
283.000017
```

`potential_energy` needs a consistent model (`consistent = true`), since 𝒫 depends on h⁰ and z₀, which the dynamics
of a Fickian (`consistent = false`) model never use: for a Fickian model, `thermodynamic_operation(model,
potential_energy)` throws a `MethodError`. The other functions, such as π, m, h⁰ and μ, remain thermodynamic properties
of the state of a Fickian model, but have no energetic meaning for it. The energy reservoirs 𝒫 and ℐ are also returned
by [`potential_energy(model)`](@ref potential_energy) and [`internal_energy`](@ref), both of which refuse a
Fickian model.
"""
thermodynamic_operation(model, f) = thermodynamic_function_operation(model, f)

thermodynamic_operation(model, f::typeof(potential_energy)) =
    thermodynamic_operation(production_parameters(model).diffusivity, model, f)

thermodynamic_operation(::ThermodynamicDiffusivity, model, f::typeof(potential_energy)) =
    thermodynamic_function_operation(model, f)

#=
The energy reservoirs 𝒫 and ℐ of the heuristic split 𝒫 = ℐ + 𝒢 + g₀ z₀ (`ape_budget.jl`), with the gravitational energy
𝒢 in `bpe_budget.jl`, next to 𝒢★. Every term is evaluated with the (θ, S) and the height z of the cell centre, so that
the split holds in every cell to round-off, for any thermodynamics.
=#

"""
$(TYPEDSIGNATURES)

Return the potential energy 𝒫 = h⁰ - b (z - z₀) + g₀ z₀ of `model` at (Center, Center, Center), the field of
`potential_energy(θ, S, z, thermodynamics)` at the cell centres, equal to `thermodynamic_operation(model,
potential_energy)`. In every cell, to round-off and for any thermodynamics,

    𝒫 = ℐ + 𝒢 + g₀ z₀,

with ℐ the [`internal_energy`](@ref) and 𝒢 the [`gravitational_energy`](@ref), and, with the reference height z★ of
[`reference_height`](@ref), 𝒫 = 𝒜 + ℬ, the [`available_potential_energy`](@ref) and the
[`background_potential_energy`](@ref).

It needs a consistent model (`consistent = true`), since 𝒫 depends on h⁰ and z₀, which the dynamics of a Fickian
(`consistent = false`) model never use: for a Fickian model it throws a `MethodError`.

```jldoctest
julia> using Oceananigans, ThermoShenanigans

julia> grid = RectilinearGrid(size = (4, 4), x = (0, 1), z = (-1, 0), topology = (Periodic, Flat, Bounded));

julia> thermodynamics = CabbelingBoussinesqThermodynamics(thermal_expansion = 2e-4, haline_contraction = 0.8,
                                                          reference_temperature = 283, reference_salinity = 0.035,
                                                          reference_heat_capacity = 4000);

julia> components = thermodynamic_model_components(grid, thermodynamics; viscosity = 1e-6, thermal_diffusivity = 1e-7,
                                                   salt_diffusivity = 1e-9);

julia> model = NonhydrostaticModel(grid; tracers = (:θ, :S), components...);

julia> set!(model, θ = 284, S = 0.035);

julia> 𝒫 = Field(potential_energy(model));

julia> round(𝒫[1, 1, 1], digits = 8) # cₚ₀ (θ - T₀) - g α (θ - T₀) z in the bottom cell, z = -0.875
4000.00171616
```
"""
potential_energy(model) = potential_energy(production_parameters(model).diffusivity, model)

potential_energy(::ThermodynamicDiffusivity, model) = thermodynamic_function_operation(model, potential_energy)

# Counterpart of Oceanostics v0.21.7 `PotentialEnergyTendency` (`src/PotentialEnergyEquation.jl:465`)
# ∂ₜ𝒫 = π ∂ₜθ + m ∂ₜS
@inline potential_energy_tendencyᶜᶜᶜ(i, j, k, grid, buoyancy_formulation, fields, tendencies, walls) =
    thermodynamic_functionᶜᶜᶜ(i, j, k, grid, exner, buoyancy_formulation, fields) *
    θ_tendencyᶜᶜᶜ(i, j, k, grid, tendencies, walls) +
    thermodynamic_functionᶜᶜᶜ(i, j, k, grid, chemical_potential_analogue, buoyancy_formulation, fields) *
    salt_tendencyᶜᶜᶜ(i, j, k, grid, buoyancy_formulation, tendencies, walls)

"""
$(TYPEDSIGNATURES)

Return the tendency ∂ₜ𝒫 = π ∂ₜθ + m ∂ₜS of the [`potential_energy`](@ref) of `model` at (Center, Center, Center), with
∂ₜθ = G_θ + F_θ and ∂ₜS = G_S + F_S, G the model's tendencies and F the tendencies of the fluxes of its
`FluxBoundaryCondition`s. When σ_θ is the only forcing, in every cell

    ∂𝒫/∂t = π ADV_θ + m ADV_S - ∇·j_𝒫 + ε,    j_𝒫 = ℑπ j_θ + ℑm j_S,

with ADV_θ and ADV_S the tendencies of θ and S by the discrete advection, j_θ and j_S the [`θ_flux`](@ref) and
[`salt_flux`](@ref), whose wall values give j_𝒫·n = π₁ Q_θ + m₁ Q_S, and ε the [`viscous_dissipation`](@ref).

It reads the model's tendencies `Gⁿ`, which `time_step!`, `set!` and `update_state!` compute for the current state; it
is valid after any of them.

It needs a consistent model (`consistent = true`), since 𝒫 depends on h⁰ and z₀: for a Fickian (`consistent = false`)
model it throws a `MethodError`.
"""
potential_energy_tendency(model) = potential_energy_tendency(production_parameters(model).diffusivity, model)

potential_energy_tendency(::ThermodynamicDiffusivity, model) =
    center_operation(potential_energy_tendencyᶜᶜᶜ, model.grid, model.buoyancy.formulation, field_data(model.tracers),
                     model_tendencies(model), wall_conditions(model))

"""
$(TYPEDSIGNATURES)

Return the internal energy ℐ = h⁰ of `model` at (Center, Center, Center), the potential enthalpy of the state, equal to
`thermodynamic_operation(model, potential_enthalpy)`. In every cell, to round-off,

    𝒫 = ℐ + 𝒢 + g₀ z₀

for any thermodynamics, with 𝒫 the [`potential_energy`](@ref) and 𝒢 the
[`gravitational_energy`](@ref), and

    ℬ = ℐ + 𝒢★ + g₀ z₀

for any thermodynamics whose h⁰ and b are independent of z, as `AbstractBoussinesqThermodynamics` requires, with ℬ the
[`background_potential_energy`](@ref) and 𝒢★ the [`background_gravitational_energy`](@ref): ℬ evaluates h⁰ and b at the
reference height z★, ℐ and 𝒢★ at the height z of the cell. ℐ thus lies entirely in ℬ.

It needs a consistent model (`consistent = true`), since the dynamics of a Fickian (`consistent = false`) model never
use h⁰, which is then no energy reservoir: for a Fickian model it throws a `MethodError`.
`thermodynamic_operation(model, potential_enthalpy)` returns h⁰, as a thermodynamic property of the state, for either
model.

```jldoctest
julia> using Oceananigans, ThermoShenanigans

julia> grid = RectilinearGrid(size = (4, 4), x = (0, 1), z = (-1, 0), topology = (Periodic, Flat, Bounded));

julia> thermodynamics = CabbelingBoussinesqThermodynamics(thermal_expansion = 2e-4, haline_contraction = 0.8,
                                                          reference_temperature = 283, reference_salinity = 0.035,
                                                          reference_heat_capacity = 4000);

julia> components = thermodynamic_model_components(grid, thermodynamics; viscosity = 1e-6, thermal_diffusivity = 1e-7,
                                                   salt_diffusivity = 1e-9);

julia> model = NonhydrostaticModel(grid; tracers = (:θ, :S), components...);

julia> set!(model, θ = 284, S = 0.035);

julia> ℐ = Field(internal_energy(model));

julia> ℐ[1, 1, 1] # cₚ₀ (θ - T₀)
4000.0
```
"""
internal_energy(model) = internal_energy(production_parameters(model).diffusivity, model)

internal_energy(::ThermodynamicDiffusivity, model) = thermodynamic_function_operation(model, potential_enthalpy)

# Operation of the thermodynamic function f(θ, S, z, thermodynamics) of the model's tracers, at (Center, Center, Center)
thermodynamic_function_operation(model, f) =
    center_operation(thermodynamic_functionᶜᶜᶜ, model.grid, f, model.buoyancy.formulation, field_data(model.tracers))

# Operation of a production term in `production.jl`, at (Center, Center, Center)
production_operation(kernel, model) =
    center_operation(kernel, model.grid, model.clock, velocities_and_tracers(model), production_parameters(model))

"""
$(TYPEDSIGNATURES)

Return the kinetic energy dissipation ε = ∂ⱼuᵢ τᵢⱼ of `model` at (Center, Center, Center), formed from the viscous
fluxes at their native locations so that ∫ ε dV is exactly the discrete kinetic energy sink.
"""
viscous_dissipation(model) = production_operation(dissipationᶜᶜᶜ, model)

"""
$(TYPEDSIGNATURES)

Return the θ production σ_θ = [ε - j_θ·∇π - j_S·∇m] / π of `model` at (Center, Center, Center), the θ source that
makes the potential energy gain exactly the kinetic energy lost to viscosity. It is zero for a Fickian
(`consistent = false`) model, which has no θ source.
"""
θ_production(model) = production_operation(θ_productionᶜᶜᶜ, model)

"""
$(TYPEDSIGNATURES)

Return the buoyancy production σ_b = σ_b_heating + σ_b_nonlin of `model` at (Center, Center, Center), so that
Db/Dt = σ_b - ∇·j_b. See [`heating_buoyancy_production`](@ref) and [`nonlinear_buoyancy_production`](@ref).
"""
buoyancy_production(model) = production_operation(buoyancy_productionᶜᶜᶜ, model)

"""
$(TYPEDSIGNATURES)

Return the buoyancy production σ_b_heating = b_θ σ_θ of `model` by the θ source, at (Center, Center, Center). It is
zero for a Fickian (`consistent = false`) model, which has no θ source.
"""
heating_buoyancy_production(model) = production_operation(heating_buoyancy_productionᶜᶜᶜ, model)

"""
$(TYPEDSIGNATURES)

Return the nonlinear (cabbeling) buoyancy production σ_b_nonlin = j_θ·∇b_θ + j_S·∇b_S of `model` at
(Center, Center, Center), which vanishes for a linear equation of state.
"""
nonlinear_buoyancy_production(model) = production_operation(nonlinear_buoyancy_productionᶜᶜᶜ, model)

"""
$(TYPEDSIGNATURES)

Return the entropy production σ_η = ε / T + κ_θ cₚ |∇T|² / T² + κ_S (∂μ/∂S)_T |∇S + Γ_S 𝐤|² / T ≥ 0 of `model`
at (Center, Center, Center). It needs a consistent model (`consistent = true`), since it is the entropy production of
the consistent fluxes: for a Fickian (`consistent = false`) model it throws a `MethodError`.
"""
entropy_production(model) = entropy_production(production_parameters(model).diffusivity, model)

entropy_production(::ThermodynamicDiffusivity, model) = production_operation(entropy_productionᶜᶜᶜ, model)

"""
$(TYPEDSIGNATURES)

Return the conversion j_θ·∇π + j_S·∇m of potential energy by the diffusive fluxes of `model`, at
(Center, Center, Center). It needs a consistent model (`consistent = true`), since it is a term of the budget of 𝒫,
which depends on h⁰ and z₀: for a Fickian (`consistent = false`) model it throws a `MethodError`.
"""
potential_energy_conversion(model) = potential_energy_conversion(production_parameters(model).diffusivity, model)

potential_energy_conversion(::ThermodynamicDiffusivity, model) = production_operation(potential_energy_conversionᶜᶜᶜ, model)

# x, y and z components of a diffusive flux of the diffusivity of a model, its `ThermodynamicDiffusivity` or its
# Fickian `ScalarDiffusivity`, with the wall fluxes of the model
diffusive_flux_operations(f_x, f_y, f_z, model) =
    face_operations(f_x, f_y, f_z, model.grid, production_parameters(model).diffusivity, model.buoyancy.formulation,
                    field_data(model.tracers), wall_conditions(model))

"""
$(TYPEDSIGNATURES)

Return the diffusive θ flux j_θ of `model`, a `NamedTuple` of its `(x, y, z)` components at faces. At a wall it is the
θ flux that the model applies there: the flux of a `FluxBoundaryCondition` (zero by default), plus, for a Fickian
(`consistent = false`) model, the wall flux of its closure, that of a `ValueBoundaryCondition` or
`GradientBoundaryCondition`. With the tendency G_θ of `compute_tendencies!` and `compute_flux_bc_tendencies!`,
G_θ - σ_θ = - ∇·j_θ in every cell when u = 0. Only the explicit part of an implicit-explicit flux condition is included.
"""
θ_flux(model) = diffusive_flux_operations(total_θ_flux_x, total_θ_flux_y, total_θ_flux_z, model)

"""
$(TYPEDSIGNATURES)

Return the diffusive salt flux j_S of `model`, a `NamedTuple` of its `(x, y, z)` components at faces, with the wall
fluxes that the model applies, as for [`θ_flux`](@ref).
"""
salt_flux(model) = diffusive_flux_operations(total_salt_flux_x, total_salt_flux_y, total_salt_flux_z, model)

"""
$(TYPEDSIGNATURES)

Return the diffusive buoyancy flux j_b = b_θ j_θ + b_S j_S of `model`, a `NamedTuple` of its `(x, y, z)` components
at faces. At a wall it is b_θ₁ Q_θ + b_S₁ Q_S, with the coefficients of the boundary cell, to whose tendency the model
applies the wall fluxes Q_θ and Q_S of [`θ_flux`](@ref) and [`salt_flux`](@ref).
"""
buoyancy_flux(model) = diffusive_flux_operations(buoyancy_flux_x, buoyancy_flux_y, buoyancy_flux_z, model)

"""
$(TYPEDSIGNATURES)

Return the buoyancy work w b̃ of `model` at (Center, Center, Face), with b̃ the buoyancy force used by the model:
the conversion of potential into kinetic energy.
"""
buoyancy_work(model) = KernelFunctionOperation{Center, Center, Face}(buoyancy_workᶜᶜᶠ, model.grid, model.buoyancy.formulation,
                                                                     velocities_and_tracers(model))

# w b̃ at (Center, Center, Face)
@inline buoyancy_workᶜᶜᶠ(i, j, k, grid, buoyancy_formulation, fields) =
    @inbounds fields.w[i, j, k] * consistent_buoyancyᶜᶜᶠ(i, j, k, grid, buoyancy_formulation, fields)
