#=
Local available and background potential energy (Holliday & McIntyre 1981; Tailleux 2013; Hatton, Tailleux & Dubos):

    𝒜 = - b (z - z★) + 𝒲★,      𝒲★ = p★(z) - p★(z★),
    𝓑 = 𝒫(θ, S, z★) - 𝒲★,

so that 𝒜 + 𝓑 = 𝒫 = h⁰ - b (z - z₀) in every cell. The internal energy h⁰ lies entirely in 𝓑, and 𝒲★, the work
against the reference pressure to move a parcel from z★ to z, integrates to zero over the domain.
Following the heuristic split 𝒫 = ℐ + 𝒢 of 𝒫 into internal energy ℐ = h⁰ and gravitational energy 𝒢 = - b (z - z₀),
𝓑 = ℐ + 𝒢★ with 𝒢★ = - b (z★ - z₀) - 𝒲★. 𝒢★ is computed directly, since h⁰ is much larger than its changes.
=#

# 𝒲★ = p★(z) - p★(z★), with p★ evaluated through the same sorted column at both heights
@inline reference_workᶜᶜᶜ(i, j, k, grid, z★, reference_potential, sorted_buoyancy, slot_potential, bottom, slot_thickness) =
    @inbounds reference_potential[1, 1, k] -
              reference_potential_at(z★[i, j, k], sorted_buoyancy, slot_potential, bottom, slot_thickness)

# Υ = z★ - z, the displacement of a parcel to its reference height
@inline reference_displacementᶜᶜᶜ(i, j, k, grid, z★) = @inbounds z★[i, j, k] - znode(i, j, k, grid, Center(), Center(), Center())

# 𝒜 = - b (z - z★) + 𝒲★ = b Υ + 𝒲★
@inline available_potential_energyᶜᶜᶜ(i, j, k, grid, buoyancy, z★, reference_potential, sorted_buoyancy, slot_potential, bottom,
                                      slot_thickness) =
    @inbounds buoyancy[i, j, k] * reference_displacementᶜᶜᶜ(i, j, k, grid, z★) +
              reference_workᶜᶜᶜ(i, j, k, grid, z★, reference_potential, sorted_buoyancy, slot_potential, bottom, slot_thickness)

# 𝒢★ = - b (z★ - z₀) - 𝒲★, the gravitational part of 𝓑
@inline background_gravitational_energyᶜᶜᶜ(i, j, k, grid, buoyancy, surface_height, z★, reference_potential, sorted_buoyancy,
                                           slot_potential, bottom, slot_thickness) =
    @inbounds - buoyancy[i, j, k] * (z★[i, j, k] - surface_height) -
                reference_workᶜᶜᶜ(i, j, k, grid, z★, reference_potential, sorted_buoyancy, slot_potential, bottom, slot_thickness)

# 𝓑 = 𝒫(θ, S, z★ - z₀) - 𝒲★
@inline background_potential_energyᶜᶜᶜ(i, j, k, grid, buoyancy_formulation, C, z★, reference_potential, sorted_buoyancy,
                                       slot_potential, bottom, slot_thickness) =
    @inbounds potential_energy(θ_and_sᴬ(i, j, k, get_temperature_and_salinity(buoyancy_formulation, C)...)...,
                               z★[i, j, k] - buoyancy_formulation.surface_height, buoyancy_formulation.thermodynamics) -
              reference_workᶜᶜᶜ(i, j, k, grid, z★, reference_potential, sorted_buoyancy, slot_potential, bottom, slot_thickness)

# ℐ = h⁰, in the form f(θ, S, z, thermodynamics) evaluated by thermodynamic_functionᶜᶜᶜ
@inline internal_energy(θ, S, z, thermodynamics) = potential_enthalpy(θ, S, thermodynamics)

"""
    energetics(model)

Return a `NamedTuple` of the local energetics of a model built with [`thermodynamic_model_components`](@ref),
relative to the Lorenz reference state found by sorting the buoyancy:

* `z★`: the reference height of each cell, at (Center, Center, Center); computing it sorts the buoyancy
* `Υ`: the displacement z★ - z
* `p★`: the reference pressure p★(z) = ∫ b★ dz at cell centres, at (Nothing, Nothing, Center)
* `b★`: the reference buoyancy δz p★ / Δz at w-faces, at (Nothing, Nothing, Face)
* `𝒲★`: the work p★(z) - p★(z★) against the reference pressure
* `𝒜`: the available potential energy - b (z - z★) + 𝒲★ ≥ 0
* `𝓑`: the background potential energy 𝒫(θ, S, z★) - 𝒲★, so that 𝒜 + 𝓑 = 𝒫
* `ℐ`: the internal energy h⁰, which lies entirely in 𝓑
* `𝒢★`: the gravitational part - b (z★ - z₀) - 𝒲★ of 𝓑 = ℐ + 𝒢★

The diagnostics share one sort per time: wrap them in `Field` to compute them, e.g. `Field(energetics(model).𝒜)`.
The grid must be a `RectilinearGrid` regular in x, y and z.

```jldoctest
julia> using Oceananigans, ThermoShenanigans

julia> grid = RectilinearGrid(size = (4, 8), x = (0, 1), z = (-1, 0), topology = (Periodic, Flat, Bounded));

julia> thermodynamics = CabbelingBoussinesqThermodynamics(thermal_expansion = 2e-4, haline_contraction = 0.8, reference_temperature = 283,
                                                          reference_salinity = 0.035, reference_heat_capacity = 4000);

julia> components = thermodynamic_model_components(grid, thermodynamics; viscosity = 1e-6, thermal_diffusivity = 1e-7,
                                                   salt_diffusivity = 1e-7);

julia> model = NonhydrostaticModel(grid; tracers = (:θ, :S), components...);

julia> set!(model, θ = (x, z) -> 283 - z, S = 0.035);

julia> 𝒜 = Field(energetics(model).𝒜);

julia> round(maximum(𝒜), sigdigits = 3) # an unstable column has available potential energy
0.000751
```
"""
function energetics(model)
    grid, C = model.grid, merge(model.velocities, model.tracers)
    buoyancy_formulation = model.buoyancy.formulation
    buoyancy = center_operation(buoyancy_perturbationᶜᶜᶜ, grid, buoyancy_formulation, C)

    z★ = reference_height(buoyancy)
    state = z★.operand
    reference = (z★, state.reference_potential, state.sorted_buoyancy, state.slot_potential, state.bottom, state.slot_thickness)

    return (z★ = z★,
            Υ  = center_operation(reference_displacementᶜᶜᶜ, grid, z★),
            p★ = reference_profile(z★, state.reference_potential),
            b★ = reference_profile(z★, state.reference_buoyancy),
            𝒲★ = center_operation(reference_workᶜᶜᶜ, grid, reference...),
            𝒜  = center_operation(available_potential_energyᶜᶜᶜ, grid, state.buoyancy, reference...),
            𝓑  = center_operation(background_potential_energyᶜᶜᶜ, grid, buoyancy_formulation, C, reference...),
            ℐ  = center_operation(thermodynamic_functionᶜᶜᶜ, grid, internal_energy, buoyancy_formulation, C),
            𝒢★ = center_operation(background_gravitational_energyᶜᶜᶜ, grid, state.buoyancy, buoyancy_formulation.surface_height,
                                  reference...))
end
