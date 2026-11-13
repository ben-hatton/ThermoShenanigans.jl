#=
Local available and background potential energy (Holliday & McIntyre 1981; Tailleux 2013; Hatton, Tailleux & Dubos):

    𝒜 = - b (z - z★) + 𝒲★,      𝒲★ = p★(z) - p★(z★),
    ℬ = 𝒫(θ, S, z★) - 𝒲★,

so that 𝒜 + ℬ = 𝒫 = h⁰ - b (z - z₀) + g₀ z₀ in every cell. The internal energy h⁰ lies entirely in ℬ, and 𝒲★, the
work against the reference pressure to move a parcel from z★ to z, integrates to zero over the domain.
Following the heuristic split 𝒫 = ℐ + 𝒢 + g₀ z₀ of 𝒫 into internal energy ℐ = h⁰, gravitational energy
𝒢 = - b (z - z₀) and a constant, ℬ = ℐ + 𝒢★ + g₀ z₀ with 𝒢★ = - b (z★ - z₀) - 𝒲★. 𝒢★ is computed directly, since h⁰
is much larger than its changes.
=#

# f(i, j, k, grid, args...), with the reference height z★ as an extra argument of its operation, so that computing the
# operation computes the reference state that `args` read
@inline with_reference_state(i, j, k, grid, z★, f::F, args...) where F = f(i, j, k, grid, args...)

# 𝒲★ = p★(z) - p★(z★), from the reference potential at the cell's level and at its reference height
@inline reference_workᶜᶜᶜ(i, j, k, grid, reference_potential, parcel_reference_potential) =
    @inbounds reference_potential[1, 1, k] - parcel_reference_potential[i, j, k]

# Υ = z★ - z, the displacement of a parcel to its reference height
@inline reference_displacementᶜᶜᶜ(i, j, k, grid, z★) =
    @inbounds z★[i, j, k] - znode(i, j, k, grid, Center(), Center(), Center())

# 𝒜 = - b (z - z★) + 𝒲★ = b Υ + 𝒲★
@inline available_potential_energyᶜᶜᶜ(i, j, k, grid, buoyancy, z★, reference_potential, parcel_reference_potential) =
    @inbounds buoyancy[i, j, k] * reference_displacementᶜᶜᶜ(i, j, k, grid, z★) +
              reference_workᶜᶜᶜ(i, j, k, grid, reference_potential, parcel_reference_potential)

# 𝒢★ = - b (z★ - z₀) - 𝒲★, the gravitational part of ℬ
@inline background_gravitational_energyᶜᶜᶜ(i, j, k, grid, buoyancy, z₀, z★, reference_potential,
                                           parcel_reference_potential) =
    @inbounds - buoyancy[i, j, k] * (z★[i, j, k] - z₀) -
                reference_workᶜᶜᶜ(i, j, k, grid, reference_potential, parcel_reference_potential)

# ℬ = 𝒫(θ, S, z★) - 𝒲★
@inline background_potential_energyᶜᶜᶜ(i, j, k, grid, buoyancy_formulation, fields, z★, reference_potential,
                                       parcel_reference_potential) =
    @inbounds potential_energy(θ_and_sᴬ(i, j, k, get_temperature_and_salinity(buoyancy_formulation, fields)...)...,
                               z★[i, j, k], buoyancy_formulation.thermodynamics) -
              reference_workᶜᶜᶜ(i, j, k, grid, reference_potential, parcel_reference_potential)

#=
Local APE budget. With Db/Dt = σ_b - ∇·j_b and b★(z★) = b,

    D𝒜/Dt = Υ σ_b - εₐ - ∇·j_𝒜 - w (b - b★(z)) + R★,
    εₐ = j_b·∇(z - z★),   j_𝒜 = Υ j_b,   R★ = Ṗ★(z) - Ṗ★(z★).

Discretely, with j_𝒜 = ℑ(Υ) j_b on the faces and εₐ the face contraction - A j_b δΥ averaged to the cell centres,
- Υ ∇·j_b = - ∇·j_𝒜 - εₐ holds exactly in every cell, as does b_θ ∇·j_θ + b_S ∇·j_S = ∇·j_b - σ_b_nonlin.

The model's discrete advection of θ and S changes b by ADV_b = - b_θ ∇·(u θ) - b_S ∇·(u S). When no two cells share a
buoyancy, z★ is constant in time, and the semi-discrete tendency of 𝒜 = b Υ + 𝒲★ in each cell is exactly

    ∂𝒜/∂t = Υ (ADV_b + σ_b - ∇·j_b) + R★
          = Υ σ_b - εₐ - ∇·j_𝒜 - ∇·(u 𝒜) - ℑz[w (b̃ - b★)] + 𝒩 + R★,

with Ṗ★ in R★ formed from the full discrete Ḃ = ADV_b + σ_b - ∇·j_b, and

    𝒩 = Υ ADV_b + ∇·(u 𝒜) + ℑz[w (b̃ - b★)],

the error of the discrete advection in the APE budget: the advective tendency of 𝒜 by the chain rule, less its
continuum form (after the discrete variance decay of Klingbeil et al. 2014, Ocean Modell. 81). In the continuum
𝒩 = 0. Over the domain, ∇·(u 𝒜) and w b★ integrate to zero, and with centred second-order advection and the
energy-consistent b̃, Σ V z ADV_b = Σ V w b̃, so that

    Σ V 𝒩 = Σ V z★ ADV_b = - (the change of ∫ ℬ dV by the discrete advection),

the spurious diapycnal mixing of Ilıcak et al. (2012, Ocean Modell. 45-46): Σ V 𝒩 < 0 when the discrete advection
mixes, and Σ V 𝒩 > 0 when the dispersive errors of centred advection unmix.
=#

# ADV_b = - b_θ ∇·(u θ) - b_S ∇·(u S), with the model's advection schemes
@inline advective_buoyancy_tendencyᶜᶜᶜ(i, j, k, grid, advection, buoyancy_formulation, fields) =
    - thermodynamic_functionᶜᶜᶜ(i, j, k, grid, HeightIndependent(∂b∂θ), buoyancy_formulation, fields) *
      div_Uc(i, j, k, grid, advection.θ, fields, fields.θ) -
    salt_advection_buoyancy_tendencyᶜᶜᶜ(i, j, k, grid, advection, buoyancy_formulation, fields)

# b_S ∇·(u S), zero for constant salinity
@inline salt_advection_buoyancy_tendencyᶜᶜᶜ(i, j, k, grid, advection, buoyancy_formulation, fields) =
    thermodynamic_functionᶜᶜᶜ(i, j, k, grid, HeightIndependent(∂b∂S), buoyancy_formulation, fields) *
    div_Uc(i, j, k, grid, advection.S, fields, fields.S)

@inline salt_advection_buoyancy_tendencyᶜᶜᶜ(i, j, k, grid, advection, ::ConstantSalinityThermodynamicBuoyancy, fields) =
    zero(grid)

# A j_b at faces
for (dir, A) in ((:x, Axᶠᶜᶜ), (:y, Ayᶜᶠᶜ), (:z, Azᶜᶜᶠ))
    buoyancy_flux_kernel = Symbol(:buoyancy_flux_, dir)
    area_buoyancy_flux = Symbol(:area_buoyancy_flux_, dir)
    @eval @inline $area_buoyancy_flux(i, j, k, grid, closure, buoyancy_formulation, fields) =
        $A(i, j, k, grid) * $buoyancy_flux_kernel(i, j, k, grid, closure, buoyancy_formulation, fields)
end

# Ḃ = σ_b - ∇·j_b, the non-advective buoyancy tendency
@inline function buoyancy_tendencyᶜᶜᶜ(i, j, k, grid, clock, fields, parameters)
    closure, buoyancy_formulation = parameters.diffusivity, parameters.buoyancy
    ∇_dot_j_b = (δxᶜᵃᵃ(i, j, k, grid, area_buoyancy_flux_x, closure, buoyancy_formulation, fields) +
                 δyᵃᶜᵃ(i, j, k, grid, area_buoyancy_flux_y, closure, buoyancy_formulation, fields) +
                 δzᵃᵃᶜ(i, j, k, grid, area_buoyancy_flux_z, closure, buoyancy_formulation, fields)) / Vᶜᶜᶜ(i, j, k, grid)
    return buoyancy_productionᶜᶜᶜ(i, j, k, grid, clock, fields, parameters) - ∇_dot_j_b
end

# Ḃ = ADV_b + σ_b - ∇·j_b, the full discrete buoyancy tendency
@inline total_buoyancy_tendencyᶜᶜᶜ(i, j, k, grid, advection, clock, fields, parameters) =
    advective_buoyancy_tendencyᶜᶜᶜ(i, j, k, grid, advection, parameters.buoyancy, fields) +
    buoyancy_tendencyᶜᶜᶜ(i, j, k, grid, clock, fields, parameters)

# Υ σ, the production of 𝒜 by the buoyancy source σ = production(i, j, k, grid, clock, fields, parameters)
@inline ape_productionᶜᶜᶜ(i, j, k, grid, z★, production::P, clock, fields, parameters) where P =
    reference_displacementᶜᶜᶜ(i, j, k, grid, z★) * production(i, j, k, grid, clock, fields, parameters)

for (dir, δ, ℑᶠ, ℑᶜ) in ((:x, δxᶠᵃᵃ, ℑxᶠᵃᵃ, ℑxᶜᵃᵃ), (:y, δyᵃᶠᵃ, ℑyᵃᶠᵃ, ℑyᵃᶜᵃ), (:z, δzᵃᵃᶠ, ℑzᵃᵃᶠ, ℑzᵃᵃᶜ))
    buoyancy_flux_kernel = Symbol(:buoyancy_flux_, dir)
    area_buoyancy_flux = Symbol(:area_buoyancy_flux_, dir)
    face_ape_dissipation = Symbol(:face_ape_dissipation_, dir)
    ape_flux_kernel = Symbol(:ape_flux_, dir)

    @eval begin
        # - A j_b δΥ = A j_b δ(z - z★)
        @inline $face_ape_dissipation(i, j, k, grid, z★, closure, buoyancy_formulation, fields) =
            - $area_buoyancy_flux(i, j, k, grid, closure, buoyancy_formulation, fields) *
              $δ(i, j, k, grid, reference_displacementᶜᶜᶜ, z★)

        # j_𝒜 = ℑ(Υ) j_b
        @inline $ape_flux_kernel(i, j, k, grid, z★, closure, buoyancy_formulation, fields) =
            $ℑᶠ(i, j, k, grid, reference_displacementᶜᶜᶜ, z★) * $buoyancy_flux_kernel(i, j, k, grid, closure, buoyancy_formulation, fields)
    end
end

# εₐ = j_b·∇(z - z★); by summation by parts Σ V εₐ = Σ V Υ ∇·j_b exactly
@inline ape_dissipationᶜᶜᶜ(i, j, k, grid, z★, closure, buoyancy_formulation, fields) =
    (ℑxᶜᵃᵃ(i, j, k, grid, face_ape_dissipation_x, z★, closure, buoyancy_formulation, fields) +
     ℑyᵃᶜᵃ(i, j, k, grid, face_ape_dissipation_y, z★, closure, buoyancy_formulation, fields) +
     ℑzᵃᵃᶜ(i, j, k, grid, face_ape_dissipation_z, z★, closure, buoyancy_formulation, fields)) / Vᶜᶜᶜ(i, j, k, grid)

# w (b̃ - b★) at (Center, Center, Face), the conversion of 𝒜 into kinetic energy
@inline available_buoyancy_workᶜᶜᶠ(i, j, k, grid, buoyancy_formulation, fields, reference_buoyancy) =
    @inbounds fields.w[i, j, k] * (consistent_buoyancyᶜᶜᶠ(i, j, k, grid, buoyancy_formulation, fields) -
                                   reference_buoyancy[1, 1, k])

# - ∇·(u e) of an energy density e, such as 𝒜 or ℬ, with the model's tracer advection scheme
@inline energy_advectionᶜᶜᶜ(i, j, k, grid, advection, fields, energy) = - div_Uc(i, j, k, grid, advection.θ, fields, energy)

# 𝒩 = Υ ADV_b + ∇·(u 𝒜) + ℑz[w (b̃ - b★)]
@inline spurious_ape_productionᶜᶜᶜ(i, j, k, grid, z★, ape, reference_buoyancy, advection, buoyancy_formulation, fields) =
    reference_displacementᶜᶜᶜ(i, j, k, grid, z★) *
    advective_buoyancy_tendencyᶜᶜᶜ(i, j, k, grid, advection, buoyancy_formulation, fields) -
    energy_advectionᶜᶜᶜ(i, j, k, grid, advection, fields, ape) +
    ℑzᵃᵃᶜ(i, j, k, grid, available_buoyancy_workᶜᶜᶠ, buoyancy_formulation, fields, reference_buoyancy)

#####
##### One function per diagnostic, after Oceanostics.jl, so that only the diagnostics asked for are compiled.
##### Those of the reference state take the reference height z★ returned by `reference_height(model)`: passing the
##### same z★ to each shares one computation of the reference state per time.
#####

"""
$(TYPEDSIGNATURES)

Return the displacement Υ = z★ - z of each cell to its reference height, at (Center, Center, Center).
"""
reference_displacement(z★::ReferenceHeightField) = center_operation(reference_displacementᶜᶜᶜ, z★.grid, z★)

"""
$(TYPEDSIGNATURES)

Return the reference potential p★(z) = ∫ b★ dz at the cell centres, as a `Field` at (Nothing, Nothing, Center) that
shares its data with the reference state of z★, and computes z★ when computed.
"""
reference_potential(z★::ReferenceHeightField) = reference_profile(z★, z★.operand.reference_potential, Nothing, Nothing, Center)

"""
$(TYPEDSIGNATURES)

Return the reference buoyancy b★ = δz p★ / Δz at the w-faces, as a `Field` at (Nothing, Nothing, Face) that shares
its data with the reference state of z★, and computes z★ when computed.
"""
reference_buoyancy(z★::ReferenceHeightField) = reference_profile(z★, z★.operand.reference_buoyancy, Nothing, Nothing, Face)

"""
$(TYPEDSIGNATURES)

Return the work 𝒲★ = p★(z) - p★(z★) against the reference pressure, at (Center, Center, Center), which integrates
to zero.
"""
function reference_work(z★::ReferenceHeightField)
    state = z★.operand
    return center_operation(with_reference_state, z★.grid, z★, reference_workᶜᶜᶜ, state.reference_potential,
                            state.parcel_reference_potential)
end

"""
$(TYPEDSIGNATURES)

Return the available potential energy 𝒜 = - b (z - z★) + 𝒲★ ≥ 0 at (Center, Center, Center).
"""
function available_potential_energy(z★::ReferenceHeightField)
    state = z★.operand
    return center_operation(available_potential_energyᶜᶜᶜ, z★.grid, state.buoyancy, z★, state.reference_potential,
                            state.parcel_reference_potential)
end

"""
$(TYPEDSIGNATURES)

Return the background potential energy ℬ = 𝒫(θ, S, z★) - 𝒲★ of `model` at (Center, Center, Center), so that
𝒜 + ℬ = 𝒫. The internal energy ℐ = h⁰, `thermodynamic_operation(model, HeightIndependent(potential_enthalpy))`,
lies entirely in ℬ = ℐ + 𝒢★ + g₀ z₀; see [`background_gravitational_energy`](@ref).
"""
function background_potential_energy(model, z★::ReferenceHeightField)
    state = z★.operand
    return center_operation(background_potential_energyᶜᶜᶜ, model.grid, model.buoyancy.formulation, state.tracers, z★,
                            state.reference_potential, state.parcel_reference_potential)
end

"""
$(TYPEDSIGNATURES)

Return the gravitational part 𝒢★ = - b (z★ - z₀) - 𝒲★ of the background potential energy ℬ = ℐ + 𝒢★ + g₀ z₀ of
`model`, at (Center, Center, Center), with z₀ the `reference_pressure_height` of its thermodynamics. It is computed directly, since the internal energy ℐ = h⁰ is much larger than its changes.
"""
function background_gravitational_energy(model, z★::ReferenceHeightField)
    state = z★.operand
    z₀ = reference_pressure_height(model.buoyancy.formulation.thermodynamics)
    return center_operation(background_gravitational_energyᶜᶜᶜ, model.grid, state.buoyancy, z₀, z★, state.reference_potential,
                            state.parcel_reference_potential)
end

#####
##### The local APE budget, ∂𝒜/∂t = Υ σ_b - εₐ - ∇·j_𝒜 - ∇·(u 𝒜) - ℑz[w (b̃ - b★)] + 𝒩 + R★. Boundary fluxes of
##### θ and S, and background fields, are not included.
#####

"""
$(TYPEDSIGNATURES)

Return the non-advective buoyancy tendency Ḃ = σ_b - ∇·j_b of `model` at (Center, Center, Center).
"""
buoyancy_tendency(model) = production_operation(buoyancy_tendencyᶜᶜᶜ, model)

"""
$(TYPEDSIGNATURES)

Return the buoyancy tendency ADV_b = - b_θ ∇·(u θ) - b_S ∇·(u S) of `model` due to its discrete advection of θ and S,
at (Center, Center, Center).
"""
advective_buoyancy_tendency(model) =
    center_operation(advective_buoyancy_tendencyᶜᶜᶜ, model.grid, model.advection, model.buoyancy.formulation,
                     velocities_and_tracers(model))

# 𝒜 for the advective operators, which read it in neighbouring cells: as `available_potential_energy`, but with the
# data of z★, so that its type stays small; the operations that hold it compute the reference state through z★
function ape_data_operation(z★)
    state = z★.operand
    return center_operation(available_potential_energyᶜᶜᶜ, z★.grid, state.buoyancy, z★.data, state.reference_potential,
                            state.parcel_reference_potential)
end

# Operation of Υ σ, for the buoyancy production kernel σ
ape_production_operation(production, model, z★) =
    center_operation(ape_productionᶜᶜᶜ, model.grid, z★, production, model.clock, velocities_and_tracers(model),
                     production_parameters(model))

"""
$(TYPEDSIGNATURES)

Return the production Υ σ_b of available potential energy of `model` by the buoyancy sources, at
(Center, Center, Center), the sum of [`heating_ape_production`](@ref) and [`nonlinear_ape_production`](@ref).
"""
ape_production(model, z★::ReferenceHeightField) = ape_production_operation(buoyancy_productionᶜᶜᶜ, model, z★)

"""
$(TYPEDSIGNATURES)

Return the production Υ σ_b_heating of available potential energy of `model` by the θ source, at
(Center, Center, Center).
"""
heating_ape_production(model, z★::ReferenceHeightField) =
    ape_production_operation(heating_buoyancy_productionᶜᶜᶜ, model, z★)

"""
$(TYPEDSIGNATURES)

Return the production Υ σ_b_nonlin of available potential energy of `model` by cabbeling, at
(Center, Center, Center), which vanishes for a linear equation of state.
"""
nonlinear_ape_production(model, z★::ReferenceHeightField) =
    ape_production_operation(nonlinear_buoyancy_productionᶜᶜᶜ, model, z★)

"""
$(TYPEDSIGNATURES)

Return the diffusive dissipation εₐ = j_b·∇(z - z★) of available potential energy of `model`, at
(Center, Center, Center). Its integral equals that of Υ ∇·j_b exactly, by summation by parts.
"""
ape_dissipation(model, z★::ReferenceHeightField) =
    center_operation(ape_dissipationᶜᶜᶜ, model.grid, z★, production_parameters(model).diffusivity,
                     model.buoyancy.formulation, field_data(model.tracers))

"""
$(TYPEDSIGNATURES)

Return the diffusive flux j_𝒜 = Υ j_b of available potential energy of `model`, a `NamedTuple` of its `(x, y, z)`
components at faces. In every cell, - Υ ∇·j_b = - ∇·j_𝒜 - εₐ exactly.
"""
ape_flux(model, z★::ReferenceHeightField) =
    face_operations(ape_flux_x, ape_flux_y, ape_flux_z, model.grid, z★, production_parameters(model).diffusivity,
                    model.buoyancy.formulation, field_data(model.tracers))

"""
$(TYPEDSIGNATURES)

Return the conversion w (b̃ - b★) of available potential energy into kinetic energy of `model`, at
(Center, Center, Face). For a divergence-free velocity, its integral equals that of the buoyancy work w b̃.
"""
available_buoyancy_work(model, z★::ReferenceHeightField) =
    KernelFunctionOperation{Center, Center, Face}(with_reference_state, model.grid, z★, available_buoyancy_workᶜᶜᶠ,
                                                  model.buoyancy.formulation, velocities_and_tracers(model),
                                                  z★.operand.reference_buoyancy)

"""
$(TYPEDSIGNATURES)

Return the advection - ∇·(u 𝒜) of available potential energy by `model`, with its tracer advection scheme, at
(Center, Center, Center).
"""
ape_advection(model, z★::ReferenceHeightField) =
    center_operation(with_reference_state, model.grid, z★, energy_advectionᶜᶜᶜ, model.advection,
                     velocities_and_tracers(model), ape_data_operation(z★))

"""
$(TYPEDSIGNATURES)

Return the production 𝒩 = Υ ADV_b + ∇·(u 𝒜) + ℑz[w (b̃ - b★)] of available potential energy by the error of the
discrete advection of `model`, at (Center, Center, Center), with ADV_b the [`advective_buoyancy_tendency`](@ref).
𝒩 vanishes in the continuum. Its integral is that of z★ ADV_b, minus the change of the background potential energy
by the discrete advection: negative for spurious mixing, positive for spurious unmixing. Then, in every cell,

    ∂𝒜/∂t = Υ σ_b - εₐ - ∇·j_𝒜 - ∇·(u 𝒜) - ℑz[w (b̃ - b★)] + 𝒩 + R★,

exactly when no two cells share a buoyancy, and the integrated budget closes with Oceananigans' default
`Centered(order = 2)` advection.
"""
spurious_ape_production(model, z★::ReferenceHeightField) =
    center_operation(spurious_ape_productionᶜᶜᶜ, model.grid, z★, ape_data_operation(z★), z★.operand.reference_buoyancy,
                     model.advection, model.buoyancy.formulation, velocities_and_tracers(model))

"""
$(TYPEDSIGNATURES)

Return the reference term R★ = Ṗ★(z) - Ṗ★(z★) of the APE budget of `model` at (Center, Center, Center), with Ṗ★ the
tendency of the reference potential due to the buoyancy tendency Ḃ = ADV_b + σ_b - ∇·j_b of `model`, including its
discrete advection ADV_b, which leaves the reference state unchanged only in the continuum. R★ integrates to zero.
"""
function reference_tendency_work(model, z★::ReferenceHeightField)
    parcel_tendency = reference_tendency(model, z★)
    # R★ has the form of 𝒲★; computing Ṗ★(z★) computes Ṗ★ at each level
    return center_operation(reference_workᶜᶜᶜ, z★.grid, parcel_tendency.operand.reference_tendency, parcel_tendency)
end

#=
Local kinetic energy budget. With 𝒦 = (ℑx u² + ℑy v² + ℑz w²) / 2 at the cell centres, the model's momentum equation
∂u/∂t = ADV_u - ∇p + b̃ 𝐤 - ∇·F, with p = p_NHS + p_HY′ and F the viscous fluxes of Oceananigans (F = - τ), gives

    ∂𝒦/∂t = A_𝒦 - ∇·j_𝒦 + ℑz[w (b̃ - b★)] - ε,    j_𝒦 = (p - p★) u + u·F,

exactly in every cell for a divergence-free velocity on a regular grid:

- A_𝒦 = ℑ(u·ADV_u) is the advective tendency of 𝒦. For centred second-order advection it is exactly the divergence
  of a discrete flux of 𝒦: in each direction u_{i+½} δ(ū ū) = δ(ū u₋ u₊ / 2) + u² δū / 2, and the last terms sum to
  u² ∇·u / 2 = 0. A_𝒦 then integrates to zero; for other schemes its integral is their numerical dissipation.
- The pressure and viscous terms follow by summation by parts, ℑ(u δF) = δ(u ℑF) - F δu, whose last term is the
  dissipation ε of `viscous_dissipation`, formed in the same way at the native locations of the viscous fluxes.
- The w equation holds the buoyancy work in the hydrostatic pressure anomaly, δz p_HY′ = b̃ Δz, and
  ∇·(p★ u) = ℑz(w b★) for a divergence-free velocity, with p★ at the u- and v-faces taken at their level and
  interpolated to the w-faces. So the pressure work becomes the conversion ℑz[w (b̃ - b★)] of 𝒜 into 𝒦.
=#

# 𝒦 = (ℑx u² + ℑy v² + ℑz w²) / 2
@inline kinetic_energyᶜᶜᶜ(i, j, k, grid, velocities) =
    ℑxᶜᵃᵃ(i, j, k, grid, half_ϕ², velocities.u) + ℑyᵃᶜᵃ(i, j, k, grid, half_ϕ², velocities.v) +
    ℑzᵃᵃᶜ(i, j, k, grid, half_ϕ², velocities.w)

# u ADV_u, v ADV_v and w ADV_w at the velocity points, with the model's momentum advection scheme
@inline u_advective_workᶠᶜᶜ(i, j, k, grid, advection, velocities) =
    @inbounds - velocities.u[i, j, k] * div_𝐯u(i, j, k, grid, advection, velocities, velocities.u)
@inline v_advective_workᶜᶠᶜ(i, j, k, grid, advection, velocities) =
    @inbounds - velocities.v[i, j, k] * div_𝐯v(i, j, k, grid, advection, velocities, velocities.v)
@inline w_advective_workᶜᶜᶠ(i, j, k, grid, advection, velocities) =
    @inbounds - velocities.w[i, j, k] * div_𝐯w(i, j, k, grid, advection, velocities, velocities.w)

# A_𝒦 = ℑ(u·ADV_u)
@inline ke_advectionᶜᶜᶜ(i, j, k, grid, advection, velocities) =
    ℑxᶜᵃᵃ(i, j, k, grid, u_advective_workᶠᶜᶜ, advection, velocities) +
    ℑyᵃᶜᵃ(i, j, k, grid, v_advective_workᶜᶠᶜ, advection, velocities) +
    ℑzᵃᵃᶜ(i, j, k, grid, w_advective_workᶜᶜᶠ, advection, velocities)

# p = p_NHS + p_HY′, where the hydrostatic pressure anomaly may be absent
@inline pressureᶜᶜᶜ(i, j, k, grid, nonhydrostatic_pressure, hydrostatic_pressure) =
    @inbounds nonhydrostatic_pressure[i, j, k] + hydrostatic_pressure[i, j, k]
@inline pressureᶜᶜᶜ(i, j, k, grid, nonhydrostatic_pressure, ::Nothing) = @inbounds nonhydrostatic_pressure[i, j, k]

# p★ at each level, at the cell centres
@inline reference_potentialᶜᶜᶜ(i, j, k, grid, reference_potential) = @inbounds reference_potential[1, 1, k]

# p - p★
@inline pressure_anomalyᶜᶜᶜ(i, j, k, grid, pressures, reference_potential) =
    pressureᶜᶜᶜ(i, j, k, grid, pressures...) - reference_potentialᶜᶜᶜ(i, j, k, grid, reference_potential)

# uᵢ Fᵢⱼ at the native location of the viscous flux Fᵢⱼ, with uᵢ interpolated there; args = (closure, K, clock, fields, b)
@inline u_F₁₁ᶠᶜᶜ(i, j, k, grid, args...) = @inbounds args[4].u[i, j, k] * ℑxᶠᵃᵃ(i, j, k, grid, viscous_flux_ux, args...)
@inline v_F₂₁ᶠᶠᶜ(i, j, k, grid, args...) = ℑxᶠᵃᵃ(i, j, k, grid, args[4].v) * viscous_flux_vx(i, j, k, grid, args...)
@inline w_F₃₁ᶠᶜᶠ(i, j, k, grid, args...) = ℑxᶠᵃᵃ(i, j, k, grid, args[4].w) * viscous_flux_wx(i, j, k, grid, args...)
@inline u_F₁₂ᶠᶠᶜ(i, j, k, grid, args...) = ℑyᵃᶠᵃ(i, j, k, grid, args[4].u) * viscous_flux_uy(i, j, k, grid, args...)
@inline v_F₂₂ᶜᶠᶜ(i, j, k, grid, args...) = @inbounds args[4].v[i, j, k] * ℑyᵃᶠᵃ(i, j, k, grid, viscous_flux_vy, args...)
@inline w_F₃₂ᶜᶠᶠ(i, j, k, grid, args...) = ℑyᵃᶠᵃ(i, j, k, grid, args[4].w) * viscous_flux_wy(i, j, k, grid, args...)
@inline u_F₁₃ᶠᶜᶠ(i, j, k, grid, args...) = ℑzᵃᵃᶠ(i, j, k, grid, args[4].u) * viscous_flux_uz(i, j, k, grid, args...)
@inline v_F₂₃ᶜᶠᶠ(i, j, k, grid, args...) = ℑzᵃᵃᶠ(i, j, k, grid, args[4].v) * viscous_flux_vz(i, j, k, grid, args...)
@inline w_F₃₃ᶜᶜᶠ(i, j, k, grid, args...) = @inbounds args[4].w[i, j, k] * ℑzᵃᵃᶠ(i, j, k, grid, viscous_flux_wz, args...)

# u·F, the viscous flux of 𝒦 (u·F = - u·τ), at the faces
@inline viscous_ke_flux_x(i, j, k, grid, args...) =
    u_F₁₁ᶠᶜᶜ(i, j, k, grid, args...) + ℑyᵃᶜᵃ(i, j, k, grid, v_F₂₁ᶠᶠᶜ, args...) + ℑzᵃᵃᶜ(i, j, k, grid, w_F₃₁ᶠᶜᶠ, args...)
@inline viscous_ke_flux_y(i, j, k, grid, args...) =
    ℑxᶜᵃᵃ(i, j, k, grid, u_F₁₂ᶠᶠᶜ, args...) + v_F₂₂ᶜᶠᶜ(i, j, k, grid, args...) + ℑzᵃᵃᶜ(i, j, k, grid, w_F₃₂ᶜᶠᶠ, args...)
@inline viscous_ke_flux_z(i, j, k, grid, args...) =
    ℑxᶜᵃᵃ(i, j, k, grid, u_F₁₃ᶠᶜᶠ, args...) + ℑyᵃᶜᵃ(i, j, k, grid, v_F₂₃ᶜᶠᶠ, args...) + w_F₃₃ᶜᶜᶠ(i, j, k, grid, args...)

#=
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
=#

for (dir, ℑᶠ, velocity, wall_mask) in ((:x, ℑxᶠᵃᵃ, :u, biharmonic_mask_x),
                                       (:y, ℑyᵃᶠᵃ, :v, biharmonic_mask_y),
                                       (:z, ℑzᵃᵃᶠ, :w, biharmonic_mask_z))
    viscous_ke_flux       = Symbol(:viscous_ke_flux_, dir)
    unmasked_ke_flux      = Symbol(:unmasked_ke_flux_, dir)
    ke_flux_kernel        = Symbol(:ke_flux_, dir)
    θ_flux_kernel         = Symbol(:masked_θ_flux_, dir)
    salt_flux_kernel      = Symbol(:masked_salt_flux_, dir)
    potential_energy_flux = Symbol(:potential_energy_flux_, dir)
    ape_flux_kernel       = Symbol(:ape_flux_, dir)
    unmasked_bpe_flux     = Symbol(:unmasked_bpe_flux_, dir)
    bpe_flux_kernel       = Symbol(:bpe_flux_, dir)

    @eval begin
        # j_𝒦 = (p - p★) u + u·F
        @inline $unmasked_ke_flux(i, j, k, grid, clock, fields, parameters, pressures, reference_potential) =
            @inbounds fields.$velocity[i, j, k] * $ℑᶠ(i, j, k, grid, pressure_anomalyᶜᶜᶜ, pressures, reference_potential) +
                      $viscous_ke_flux(i, j, k, grid, parameters.viscous_closure, parameters.closure_fields, clock, fields,
                                       parameters.buoyancy)

        # Zero at walls, where the velocity normal to the wall and the model's viscous fluxes vanish
        @inline $ke_flux_kernel(i, j, k, grid, clock, fields, parameters, pressures, reference_potential) =
            $wall_mask(i, j, k, grid, $unmasked_ke_flux, clock, fields, parameters, pressures, reference_potential)

        # j_𝒫 = ℑπ j_θ + ℑm j_S, the diffusive flux of 𝒫
        @inline $potential_energy_flux(i, j, k, grid, closure, buoyancy_formulation, fields) =
            $ℑᶠ(i, j, k, grid, thermodynamic_functionᶜᶜᶜ, exner, buoyancy_formulation, fields) *
            $θ_flux_kernel(i, j, k, grid, closure, buoyancy_formulation, fields) +
            $ℑᶠ(i, j, k, grid, thermodynamic_functionᶜᶜᶜ, chemical_potential_analogue, buoyancy_formulation, fields) *
            $salt_flux_kernel(i, j, k, grid, closure, buoyancy_formulation, fields)

        # j_ℬ = p★ u + j_𝒫 - j_𝒜
        @inline $unmasked_bpe_flux(i, j, k, grid, z★, reference_potential, closure, buoyancy_formulation, fields) =
            @inbounds fields.$velocity[i, j, k] * $ℑᶠ(i, j, k, grid, reference_potentialᶜᶜᶜ, reference_potential) +
                      $potential_energy_flux(i, j, k, grid, closure, buoyancy_formulation, fields) -
                      $ape_flux_kernel(i, j, k, grid, z★, closure, buoyancy_formulation, fields)

        @inline $bpe_flux_kernel(i, j, k, grid, z★, reference_potential, closure, buoyancy_formulation, fields) =
            $wall_mask(i, j, k, grid, $unmasked_bpe_flux, z★, reference_potential, closure, buoyancy_formulation, fields)
    end
end

# π ADV_θ + m ADV_S = - π ∇·(u θ) - m ∇·(u S), the tendency of 𝒫 by the model's advection of θ and S
@inline advective_potential_energy_tendencyᶜᶜᶜ(i, j, k, grid, advection, buoyancy_formulation, fields) =
    - thermodynamic_functionᶜᶜᶜ(i, j, k, grid, exner, buoyancy_formulation, fields) *
      div_Uc(i, j, k, grid, advection.θ, fields, fields.θ) -
    salt_advection_potential_energy_tendencyᶜᶜᶜ(i, j, k, grid, advection, buoyancy_formulation, fields)

# m ∇·(u S), zero for constant salinity
@inline salt_advection_potential_energy_tendencyᶜᶜᶜ(i, j, k, grid, advection, buoyancy_formulation, fields) =
    thermodynamic_functionᶜᶜᶜ(i, j, k, grid, chemical_potential_analogue, buoyancy_formulation, fields) *
    div_Uc(i, j, k, grid, advection.S, fields, fields.S)

@inline salt_advection_potential_energy_tendencyᶜᶜᶜ(i, j, k, grid, advection, ::ConstantSalinityThermodynamicBuoyancy,
                                                    fields) = zero(grid)

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

#####
##### The local KE budget, ∂𝒦/∂t = A_𝒦 - ∇·j_𝒦 + ℑz[w (b̃ - b★)] - ε, with the conversion ℑz[w (b̃ - b★)] of
##### `available_buoyancy_work` and the dissipation ε of `viscous_dissipation`. Forcing of the velocities, Coriolis and
##### background fields are not included.
#####

"""
$(TYPEDSIGNATURES)

Return the kinetic energy 𝒦 = (ℑx u² + ℑy v² + ℑz w²) / 2 of `model` at (Center, Center, Center). On a regular grid,
its integral is that of the kinetic energy of the velocities at their own locations.
"""
kinetic_energy(model) = center_operation(kinetic_energyᶜᶜᶜ, model.grid, field_data(model.velocities))

"""
$(TYPEDSIGNATURES)

Return the advective tendency A_𝒦 = ℑ(u·ADV_u) of the kinetic energy of `model`, with ADV_u the model's momentum
advection, at (Center, Center, Center). For a divergence-free velocity and Oceananigans' default `Centered(order = 2)`
advection, A_𝒦 is exactly the divergence of a discrete flux of 𝒦 in every cell, and integrates to zero; for other
schemes, its integral is their numerical dissipation of kinetic energy.
"""
ke_advection(model) =
    center_operation(ke_advectionᶜᶜᶜ, model.grid, model.advection.momentum, field_data(model.velocities))

# The data of the nonhydrostatic pressure and of the hydrostatic pressure anomaly, which may be `nothing`
pressure_data(model) = (model.pressures.pNHS.data, maybe_data(model.pressures.pHY′))
maybe_data(::Nothing) = nothing
maybe_data(field) = field.data

"""
$(TYPEDSIGNATURES)

Return the flux j_𝒦 = (p - p★) u - u·τ of kinetic energy of `model` by the pressure anomaly and the viscous stress,
a `NamedTuple` of its `(x, y, z)` components at faces, zero at walls. Here p = p_NHS + p_HY′, with p_NHS the
nonhydrostatic pressure of the model's last pressure correction: after `set!`, the potential of the projection of the
velocity. For a divergence-free velocity on a regular grid, in every cell

    ∂𝒦/∂t = A_𝒦 - ∇·j_𝒦 + ℑz[w (b̃ - b★)] - ε

exactly, with A_𝒦 the [`ke_advection`](@ref), ℑz[w (b̃ - b★)] the [`available_buoyancy_work`](@ref) and ε the
[`viscous_dissipation`](@ref).
"""
function ke_flux(model, z★::ReferenceHeightField)
    args = (model.clock, velocities_and_tracers(model), production_parameters(model), pressure_data(model),
            z★.operand.reference_potential)
    return (x = KernelFunctionOperation{Face, Center, Center}(with_reference_state, model.grid, z★, ke_flux_x, args...),
            y = KernelFunctionOperation{Center, Face, Center}(with_reference_state, model.grid, z★, ke_flux_y, args...),
            z = KernelFunctionOperation{Center, Center, Face}(with_reference_state, model.grid, z★, ke_flux_z, args...))
end

#####
##### The local BPE budget, ∂ℬ/∂t = - Υ σ_b + εₐ - ∇·j_ℬ - ∇·(u ℬ) + ε - R★ + 𝒩_ℬ, sharing the terms Υ σ_b, εₐ and
##### R★ with the APE budget. Boundary fluxes of θ and S, and background fields, are not included.
#####

# ℬ for the advective operators, as `ape_data_operation` for 𝒜
function bpe_data_operation(z★)
    state = z★.operand
    return center_operation(background_potential_energyᶜᶜᶜ, z★.grid, state.buoyancy_formulation, state.tracers, z★.data,
                            state.reference_potential, state.parcel_reference_potential)
end

"""
$(TYPEDSIGNATURES)

Return the flux j_ℬ = p★ u + π(θ, z★) j_θ + m(z★) j_S of background potential energy of `model`, a `NamedTuple` of its
`(x, y, z)` components at faces, zero at walls. Its diffusive part is j_𝒫 - j_𝒜, with j_𝒫 = ℑπ j_θ + ℑm j_S the
diffusive flux of potential energy and j_𝒜 the [`ape_flux`](@ref).
"""
bpe_flux(model, z★::ReferenceHeightField) =
    face_operations(bpe_flux_x, bpe_flux_y, bpe_flux_z, model.grid, z★, z★.operand.reference_potential,
                    production_parameters(model).diffusivity, model.buoyancy.formulation, velocities_and_tracers(model))

"""
$(TYPEDSIGNATURES)

Return the advection - ∇·(u ℬ) of background potential energy by `model`, with its tracer advection scheme, at
(Center, Center, Center).
"""
bpe_advection(model, z★::ReferenceHeightField) =
    center_operation(with_reference_state, model.grid, z★, energy_advectionᶜᶜᶜ, model.advection,
                     velocities_and_tracers(model), bpe_data_operation(z★))

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
[`viscous_dissipation`](@ref) and R★ the [`reference_tendency_work`](@ref).
"""
spurious_bpe_production(model, z★::ReferenceHeightField) =
    center_operation(spurious_bpe_productionᶜᶜᶜ, model.grid, z★, bpe_data_operation(z★), z★.operand.reference_buoyancy,
                     model.advection, model.buoyancy.formulation, velocities_and_tracers(model))
