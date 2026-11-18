#=
Local background potential energy ℬ = 𝒫(θ, S, z★) - 𝒲★ and its budget; the definitions of 𝒜, ℬ and 𝒢★ are in
`ape_budget.jl`.

Local background potential energy budget. ℬ = 𝒫 - 𝒜, and ∂𝒫/∂t = π G_θ + m G_S in every cell, where
- π ∇·j_θ + m ∇·j_S = ∇·j_𝒫 - (j_θ·∇π + j_S·∇m) with j_𝒫 = ℑπ j_θ + ℑm j_S on the faces, by summation by parts;
- π σ_θ = ε - (j_θ·∇π + j_S·∇m), the θ source of the first law.
So, with the APE budget, in every cell

    ∂ℬ/∂t = - Υ σ_b + εₐ - ∇·j_ℬ - ∇·(u ℬ) + ε - R★ + 𝒩_ℬ,    j_ℬ = p★ u + j_𝒫 - j_𝒜,

where j_𝒫 - j_𝒜 = (ℑπ - ℑΥ ℑb_θ) j_θ + (ℑm - ℑΥ ℑb_S) j_S discretises π(θ, z★) j_θ + m(z★) j_S, since
π(θ, S, z) - Υ b_θ = π(θ, S, z★). With p★ u in j_ℬ for a divergence-free velocity (∇·(p★ u) = ℑz(w b★)),

    𝒩_ℬ = π ADV_θ + m ADV_S - Υ ADV_b + ∇·(u ℬ) + ℑz(w b★)

is the production of ℬ by the error of the discrete advection of θ and S, as 𝒩 is for 𝒜. With centred second-order
advection and the energy-consistent b̃, Σ V (π ADV_θ + m ADV_S) = - Σ V w b̃, so that Σ V 𝒩_ℬ = - Σ V 𝒩: the
spurious mixing by the advection converts ℬ into 𝒜 (or back) with no net change of 𝒫.

The budget shares the terms Υ σ_b, εₐ and R★ with the APE budget. At a wall, p★ u = 0 and, with the wall values of
`ape_budget.jl`, j_ℬ·n = (π₁ - Υ₁ b_θ₁) Q_θ + (m₁ - Υ₁ b_S₁) Q_S = π(θ₁, S₁, z★₁) Q_θ + m(θ₁, S₁, z★₁) Q_S, so that the
budget holds in the boundary cells as well. Background fields are not included.
=#

# ℬ = 𝒫(θ, S, z★) - 𝒲★
@inline background_potential_energyᶜᶜᶜ(i, j, k, grid, buoyancy_formulation, fields, z★, reference_potential,
                                       parcel_reference_potential) =
    @inbounds potential_energy(θ_and_sᴬ(i, j, k, get_temperature_and_salinity(buoyancy_formulation, fields)...)...,
                               z★[i, j, k], buoyancy_formulation.thermodynamics) -
              reference_workᶜᶜᶜ(i, j, k, grid, reference_potential, parcel_reference_potential)

"""
$(TYPEDSIGNATURES)

Return the background potential energy ℬ = 𝒫(θ, S, z★) - 𝒲★ of `model` at (Center, Center, Center), so that
𝒜 + ℬ = 𝒫. The internal energy ℐ = h⁰, [`internal_energy`](@ref), lies entirely in ℬ = ℐ + 𝒢★ + g₀ z₀; see
[`background_gravitational_energy`](@ref). It needs a consistent model
(`consistent = true`), since ℬ depends on h⁰ and z₀: for a Fickian (`consistent = false`) model it throws a
`MethodError`.
"""
background_potential_energy(model, z★::ReferenceHeightField) =
    background_potential_energy(production_parameters(model).diffusivity, model, z★)

function background_potential_energy(::ThermodynamicDiffusivity, model, z★)
    state = z★.operand
    return center_operation(background_potential_energyᶜᶜᶜ, model.grid, model.buoyancy.formulation, state.tracers, z★,
                            state.reference_potential, state.parcel_reference_potential)
end

# 𝒢★ = - b (z★ - z₀) - 𝒲★, the gravitational part of ℬ
@inline background_gravitational_energyᶜᶜᶜ(i, j, k, grid, buoyancy, z₀, z★, reference_potential,
                                           parcel_reference_potential) =
    @inbounds - buoyancy[i, j, k] * (z★[i, j, k] - z₀) -
                reference_workᶜᶜᶜ(i, j, k, grid, reference_potential, parcel_reference_potential)

"""
$(TYPEDSIGNATURES)

Return the gravitational part 𝒢★ = - b (z★ - z₀) - 𝒲★ of the background potential energy ℬ = ℐ + 𝒢★ + g₀ z₀ of
`model`, at (Center, Center, Center), with z₀ the `reference_pressure_height` of its thermodynamics. It is computed
directly, since the internal energy ℐ = h⁰ is much larger than its changes. For a Fickian (`consistent = false`) model,
𝒢★ depends only on b and z★, and z₀ only sets the origin of z: with z₀ = 0, 𝒢★ = - b z★ - 𝒲★.
"""
function background_gravitational_energy(model, z★::ReferenceHeightField)
    state = z★.operand
    z₀ = reference_pressure_height(model.buoyancy.formulation.thermodynamics)
    return center_operation(background_gravitational_energyᶜᶜᶜ, model.grid, state.buoyancy, z₀, z★, state.reference_potential,
                            state.parcel_reference_potential)
end

#=
𝒢 = - b (z - z₀) at the height z of the cell centre, the gravitational part of 𝒫. Compared with `minus_bz_ccc` of
Oceanostics v0.21.7 (`src/PotentialEnergyEquation.jl:55`), - b z from `buoyancy_perturbationᶜᶜᶜ` and `Zᶜᶜᶜ`: here the
height is z - z₀, through `buoyancy_height` (`thermodynamic_buoyancy.jl`), which evaluates b as `buoyancy_perturbationᶜᶜᶜ`
does, at the same znode, so that 𝒢 - 𝒢★ and 𝒜 = b Υ + 𝒲★ use the same b.
=#
@inline gravitational_energyᶜᶜᶜ(i, j, k, grid, buoyancy_formulation, fields) =
    - thermodynamic_functionᶜᶜᶜ(i, j, k, grid, buoyancy_height, buoyancy_formulation, fields)

"""
$(TYPEDSIGNATURES)

Return the gravitational energy 𝒢 = - b (z - z₀) of `model` at (Center, Center, Center), with z the height of the cell
centre and z₀ the `reference_pressure_height` of its thermodynamics. In every cell, to round-off,

    𝒫 = ℐ + 𝒢 + g₀ z₀,      𝒜 = 𝒢 - 𝒢★,

with 𝒫 the [`potential_energy`](@ref), ℐ the [`internal_energy`](@ref), 𝒜 the [`available_potential_energy`](@ref) and
𝒢★ the [`background_gravitational_energy`](@ref) of the same state. Both hold for any thermodynamics: 𝒫, ℐ and 𝒢 are
evaluated at the (θ, S, z) of the cell, and 𝒜 = b (z★ - z) + 𝒲★ and 𝒢★ = - b (z★ - z₀) - 𝒲★ take the same b as 𝒢,
that of the cell at its own height, so that z₀ cancels.

z₀ only sets the origin of z: 𝒢 = - b z' with z' = z - z₀ the height above z₀, and 𝒜 = 𝒢 - 𝒢★ is independent of it.
𝒢 depends only on b, so that it is defined for a Fickian (`consistent = false`) model as well, whose dynamics never use
z₀; there, with z₀ = 0, 𝒢 = - b z.

```jldoctest
julia> using Oceananigans, ThermoShenanigans

julia> grid = RectilinearGrid(size = (4, 4), x = (0, 1), z = (-1, 0), topology = (Periodic, Flat, Bounded));

julia> thermodynamics = CabbelingBoussinesqThermodynamics(thermal_expansion = 2e-4, haline_contraction = 0.8,
                                                          reference_temperature = 283, reference_salinity = 0.035,
                                                          reference_heat_capacity = 4000);

julia> components = thermodynamic_model_components(grid, thermodynamics; viscosity = 1e-6, thermal_diffusivity = 1e-7,
                                                   salt_diffusivity = 1e-9, consistent = false);

julia> model = NonhydrostaticModel(grid; tracers = (:θ, :S), components...);

julia> set!(model, θ = 284, S = 0.035);

julia> 𝒢 = Field(gravitational_energy(model));

julia> round(𝒢[1, 1, 1], digits = 10) # - g α (θ - T₀) z in the bottom cell, z = -0.875, for this Fickian model
0.0017161637
```
"""
gravitational_energy(model) =
    center_operation(gravitational_energyᶜᶜᶜ, model.grid, model.buoyancy.formulation, field_data(model.tracers))

# Generated: potential_energy_flux_{x,y,z}, j_𝒫 = ℑπ j_θ + ℑm j_S at interior faces and π₁ Q_θ + m₁ Q_S at walls,
# the diffusive flux of 𝒫; bpe_flux_{x,y,z}, the components of j_ℬ = p★ u + j_𝒫 - j_𝒜 at faces
for (dir, average_to_face, boundary_cell_value, velocity_component) in ((:x, ℑxᶠᵃᵃ, boundary_cell_value_x, :u),
                                                                        (:y, ℑyᵃᶠᵃ, boundary_cell_value_y, :v),
                                                                        (:z, ℑzᵃᵃᶠ, boundary_cell_value_z, :w))
    masked_θ_flux_xyz         = Symbol(:masked_θ_flux_, dir)
    masked_salt_flux_xyz      = Symbol(:masked_salt_flux_, dir)
    wall_θ_flux_xyz           = Symbol(:wall_θ_flux_, dir)
    wall_salt_flux_xyz        = Symbol(:wall_salt_flux_, dir)
    potential_energy_flux_xyz = Symbol(:potential_energy_flux_, dir)
    ape_flux_xyz              = Symbol(:ape_flux_, dir)
    bpe_flux_xyz              = Symbol(:bpe_flux_, dir)

    @eval begin
        @inline $potential_energy_flux_xyz(i, j, k, grid, closure, buoyancy_formulation, fields, walls) =
            $average_to_face(i, j, k, grid, thermodynamic_functionᶜᶜᶜ, exner, buoyancy_formulation, fields) *
            $masked_θ_flux_xyz(i, j, k, grid, closure, buoyancy_formulation, fields) +
            $average_to_face(i, j, k, grid, thermodynamic_functionᶜᶜᶜ, chemical_potential_analogue, buoyancy_formulation, fields) *
            $masked_salt_flux_xyz(i, j, k, grid, closure, buoyancy_formulation, fields) +
            $boundary_cell_value(i, j, k, grid, thermodynamic_functionᶜᶜᶜ, exner, buoyancy_formulation, fields) *
            $wall_θ_flux_xyz(i, j, k, grid, closure, buoyancy_formulation, fields, walls) +
            $boundary_cell_value(i, j, k, grid, thermodynamic_functionᶜᶜᶜ, chemical_potential_analogue, buoyancy_formulation, fields) *
            $wall_salt_flux_xyz(i, j, k, grid, closure, buoyancy_formulation, fields, walls)

        #=
        At a wall, p★ u = 0 since u·n = 0, and j_𝒫 - j_𝒜 = (π₁ - Υ₁ b_θ₁) Q_θ + (m₁ - Υ₁ b_S₁) Q_S. Since b does not
        depend on z, π(θ, S, z) - Υ b_θ = π(θ, S, z★) and m(θ, S, z) - Υ b_S = m(θ, S, z★), so that this is
        π(θ₁, S₁, z★₁) Q_θ + m(θ₁, S₁, z★₁) Q_S; the general form is kept for thermodynamics in which b depends on z.
        =#
        @inline $bpe_flux_xyz(i, j, k, grid, z★, reference_potential, closure, buoyancy_formulation, fields, walls) =
            @inbounds fields.$velocity_component[i, j, k] *
                      $average_to_face(i, j, k, grid, reference_potentialᶜᶜᶜ, reference_potential) +
                      $potential_energy_flux_xyz(i, j, k, grid, closure, buoyancy_formulation, fields, walls) -
                      $ape_flux_xyz(i, j, k, grid, z★, closure, buoyancy_formulation, fields, walls)
    end
end

"""
$(TYPEDSIGNATURES)

Return the flux j_ℬ = p★ u + π(θ, z★) j_θ + m(z★) j_S of background potential energy of `model`, a `NamedTuple` of its
`(x, y, z)` components at faces. Its diffusive part is j_𝒫 - j_𝒜, with j_𝒫 = ℑπ j_θ + ℑm j_S the diffusive flux of
potential energy and j_𝒜 the [`ape_flux`](@ref). At a wall, where p★ u = 0, it is (π₁ - Υ₁ b_θ₁) Q_θ +
(m₁ - Υ₁ b_S₁) Q_S = π(θ₁, S₁, z★₁) Q_θ + m(θ₁, S₁, z★₁) Q_S, with the coefficients of the boundary cell and the wall
fluxes Q_θ and Q_S of the [`θ_flux`](@ref) and [`salt_flux`](@ref). It needs a consistent model
(`consistent = true`), since it is a term of the budget of ℬ: for a Fickian (`consistent = false`) model it throws a
`MethodError`.
"""
bpe_flux(model, z★::ReferenceHeightField) = bpe_flux(production_parameters(model).diffusivity, model, z★)

bpe_flux(diffusivity::ThermodynamicDiffusivity, model, z★) =
    face_operations(bpe_flux_x, bpe_flux_y, bpe_flux_z, model.grid, z★, z★.operand.reference_potential, diffusivity,
                    model.buoyancy.formulation, velocities_and_tracers(model), wall_conditions(model))

# ℬ for the advective operators, as `ape_data_operation` for 𝒜
function bpe_data_operation(z★)
    state = z★.operand
    return center_operation(background_potential_energyᶜᶜᶜ, z★.grid, state.buoyancy_formulation, state.tracers, z★.data,
                            state.reference_potential, state.parcel_reference_potential)
end

"""
$(TYPEDSIGNATURES)

Return the advection - ∇·(u ℬ) of background potential energy by `model`, with its tracer advection scheme, at
(Center, Center, Center). It needs a consistent model (`consistent = true`), since it is a term of the budget of ℬ:
for a Fickian (`consistent = false`) model it throws a `MethodError`.
"""
bpe_advection(model, z★::ReferenceHeightField) = bpe_advection(production_parameters(model).diffusivity, model, z★)

bpe_advection(::ThermodynamicDiffusivity, model, z★) =
    center_operation(with_reference_state, model.grid, z★, energy_advectionᶜᶜᶜ, model.advection,
                     velocities_and_tracers(model), bpe_data_operation(z★))

# m ∇·(u S), zero for constant salinity
@inline salt_advection_potential_energy_tendencyᶜᶜᶜ(i, j, k, grid, advection, buoyancy_formulation, fields) =
    thermodynamic_functionᶜᶜᶜ(i, j, k, grid, chemical_potential_analogue, buoyancy_formulation, fields) *
    div_Uc(i, j, k, grid, advection.S, fields, fields.S)

@inline salt_advection_potential_energy_tendencyᶜᶜᶜ(i, j, k, grid, advection, ::ConstantSalinityThermodynamicBuoyancy,
                                                    fields) = zero(grid)

# π ADV_θ + m ADV_S = - π ∇·(u θ) - m ∇·(u S), the tendency of 𝒫 by the model's advection of θ and S
@inline advective_potential_energy_tendencyᶜᶜᶜ(i, j, k, grid, advection, buoyancy_formulation, fields) =
    - thermodynamic_functionᶜᶜᶜ(i, j, k, grid, exner, buoyancy_formulation, fields) *
      div_Uc(i, j, k, grid, advection.θ, fields, fields.θ) -
    salt_advection_potential_energy_tendencyᶜᶜᶜ(i, j, k, grid, advection, buoyancy_formulation, fields)

# w b★ at (Center, Center, Face)
@inline reference_buoyancy_workᶜᶜᶠ(i, j, k, grid, fields, reference_buoyancy) =
    @inbounds fields.w[i, j, k] * reference_buoyancy[1, 1, k]

# 𝒩_ℬ = π ADV_θ + m ADV_S - Υ ADV_b + ∇·(u ℬ) + ℑz(w b★)
@inline spurious_bpe_productionᶜᶜᶜ(i, j, k, grid, z★, bpe, reference_buoyancy, advection, buoyancy_formulation, fields) =
    advective_potential_energy_tendencyᶜᶜᶜ(i, j, k, grid, advection, buoyancy_formulation, fields) -
    reference_displacementᶜᶜᶜ(i, j, k, grid, z★) *
    advective_buoyancy_tendencyᶜᶜᶜ(i, j, k, grid, advection, buoyancy_formulation, fields) -
    energy_advectionᶜᶜᶜ(i, j, k, grid, advection, fields, bpe) +
    ℑzᵃᵃᶜ(i, j, k, grid, reference_buoyancy_workᶜᶜᶠ, fields, reference_buoyancy)

"""
$(TYPEDSIGNATURES)

Return the production 𝒩_ℬ = π ADV_θ + m ADV_S - Υ ADV_b + ∇·(u ℬ) + ℑz(w b★) of background potential energy by the
error of the discrete advection of `model`, at (Center, Center, Center), with ADV_θ, ADV_S and ADV_b the tendencies
of θ, S and b by the discrete advection. 𝒩_ℬ vanishes in the continuum. With Oceananigans' default
`Centered(order = 2)` advection, its integral is minus that of the [`spurious_ape_production`](@ref) 𝒩. Then, in every
cell,

    ∂ℬ/∂t = - Υ σ_b + εₐ - ∇·j_ℬ - ∇·(u ℬ) + ε - R★ + 𝒩_ℬ

exactly for a divergence-free velocity, when no two cells share a buoyancy, with Υ σ_b the [`ape_production`](@ref),
εₐ the [`ape_dissipation`](@ref), j_ℬ the [`bpe_flux`](@ref), - ∇·(u ℬ) the [`bpe_advection`](@ref), ε the
[`viscous_dissipation`](@ref) and R★ the [`reference_tendency_work`](@ref). It needs a consistent model
(`consistent = true`), since it is a term of the budget of ℬ: for a Fickian (`consistent = false`) model it throws a
`MethodError`.
"""
spurious_bpe_production(model, z★::ReferenceHeightField) =
    spurious_bpe_production(production_parameters(model).diffusivity, model, z★)

spurious_bpe_production(::ThermodynamicDiffusivity, model, z★) =
    center_operation(spurious_bpe_productionᶜᶜᶜ, model.grid, z★, bpe_data_operation(z★), z★.operand.reference_buoyancy,
                     model.advection, model.buoyancy.formulation, velocities_and_tracers(model))

# No equivalent in Oceananigans or Oceanostics
#=
π - Υ b_θ = π(θ, S, z★) because b does not depend on z; the general form is kept for thermodynamics in which b depends
on z.
=#
@inline bpe_tendencyᶜᶜᶜ(i, j, k, grid, z★, reference_tendency, parcel_tendency, buoyancy_formulation, fields, tendencies,
                        walls) =
    (thermodynamic_functionᶜᶜᶜ(i, j, k, grid, exner, buoyancy_formulation, fields) -
     reference_displacementᶜᶜᶜ(i, j, k, grid, z★) * thermodynamic_functionᶜᶜᶜ(i, j, k, grid, ∂b∂θ, buoyancy_formulation, fields)) *
    θ_tendencyᶜᶜᶜ(i, j, k, grid, tendencies, walls) +
    (thermodynamic_functionᶜᶜᶜ(i, j, k, grid, chemical_potential_analogue, buoyancy_formulation, fields) -
     reference_displacementᶜᶜᶜ(i, j, k, grid, z★) * thermodynamic_functionᶜᶜᶜ(i, j, k, grid, ∂b∂S, buoyancy_formulation, fields)) *
    salt_tendencyᶜᶜᶜ(i, j, k, grid, buoyancy_formulation, tendencies, walls) -
    reference_workᶜᶜᶜ(i, j, k, grid, reference_tendency, parcel_tendency)

"""
$(TYPEDSIGNATURES)

Return the tendency ∂ₜℬ = (π - Υ b_θ) ∂ₜθ + (m - Υ b_S) ∂ₜS - R★[ḃ] of the [`background_potential_energy`](@ref) of
`model` at (Center, Center, Center), with ∂ₜθ = G_θ + F_θ and ∂ₜS = G_S + F_S, G the model's tendencies and F the
tendencies of the fluxes of its `FluxBoundaryCondition`s, and R★[ḃ] the reference term of
[`reference_tendency_work`](@ref) formed from ḃ = b_θ ∂ₜθ + b_S ∂ₜS. When σ_θ is the only forcing, in every cell

    ∂ℬ/∂t = - Υ σ_b + εₐ - ∇·j_ℬ - ∇·(u ℬ) + ε - R★ + 𝒩_ℬ,

exactly for a divergence-free velocity, when no two cells share a buoyancy, with Υ σ_b the [`ape_production`](@ref),
εₐ the [`ape_dissipation`](@ref), j_ℬ the [`bpe_flux`](@ref), - ∇·(u ℬ) the [`bpe_advection`](@ref), ε the
[`viscous_dissipation`](@ref), R★ the [`reference_tendency_work`](@ref) and 𝒩_ℬ the
[`spurious_bpe_production`](@ref).

It reads the model's tendencies `Gⁿ`, which `time_step!`, `set!` and `update_state!` compute for the current state; it
is valid after any of them.

It needs a consistent model (`consistent = true`), since it is the tendency of ℬ: for a Fickian (`consistent = false`)
model it throws a `MethodError`.
"""
bpe_tendency(model, z★::ReferenceHeightField) = bpe_tendency(production_parameters(model).diffusivity, model, z★)

function bpe_tendency(::ThermodynamicDiffusivity, model, z★)
    parcel_tendency = prognostic_reference_tendency(model, z★)
    # Computing Ṗ★(z★) computes z★, so z★ is passed as its data, and is sorted once
    return center_operation(bpe_tendencyᶜᶜᶜ, model.grid, z★.data, parcel_tendency.operand.reference_tendency,
                            parcel_tendency, model.buoyancy.formulation, field_data(model.tracers),
                            model_tendencies(model), wall_conditions(model))
end
