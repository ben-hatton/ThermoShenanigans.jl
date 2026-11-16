#=
Local available and background potential energy (Holliday & McIntyre 1981; Tailleux 2013; Hatton, Tailleux & Dubos):

    𝒜 = - b (z - z★) + 𝒲★,      𝒲★ = p★(z) - p★(z★),
    ℬ = 𝒫(θ, S, z★) - 𝒲★,

so that 𝒜 + ℬ = 𝒫 = h⁰ - b (z - z₀) + g₀ z₀ in every cell. The internal energy h⁰ lies entirely in ℬ, and 𝒲★, the
work against the reference pressure to move a parcel from z★ to z, integrates to zero over the domain.
Following the heuristic split 𝒫 = ℐ + 𝒢 + g₀ z₀ of 𝒫 into internal energy ℐ = h⁰, gravitational energy
𝒢 = - b (z - z₀) and a constant, ℬ = ℐ + 𝒢★ + g₀ z₀ with 𝒢★ = - b (z★ - z₀) - 𝒲★. 𝒢★ is computed directly, since h⁰
is much larger than its changes.

This file holds the reference state diagnostics and the local APE budget; `ke_budget.jl` and `bpe_budget.jl` hold the
budgets of 𝒦 and ℬ, which share terms with it. For a Fickian (`consistent = false`) model, the APE budget is the only
potential-energy budget: 𝒜 and its sources depend only on b, z★ and the fluxes, while 𝒫 and ℬ depend on h⁰ and z₀,
which its dynamics never use.

One function per diagnostic, after Oceanostics.jl, so that only the diagnostics asked for are compiled. Those of the
reference state take the reference height z★ returned by `reference_height(model)`: passing the same z★ to each shares
one computation of the reference state per time.
=#

# f(i, j, k, grid, args...), with the reference height z★ as an extra argument of its operation, so that computing the
# operation computes the reference state that `args` read
@inline with_reference_state(i, j, k, grid, z★, f::F, args...) where F = f(i, j, k, grid, args...)

# Υ = z★ - z, the displacement of a parcel to its reference height
@inline reference_displacementᶜᶜᶜ(i, j, k, grid, z★) =
    @inbounds z★[i, j, k] - znode(i, j, k, grid, Center(), Center(), Center())

"""
$(TYPEDSIGNATURES)

Return the displacement Υ = z★ - z of each cell to its reference height, at (Center, Center, Center).
"""
reference_displacement(z★::ReferenceHeightField) = center_operation(reference_displacementᶜᶜᶜ, z★.grid, z★)

# p★ at each level, read at the cell centres by the fluxes of 𝒦 and ℬ
@inline reference_potentialᶜᶜᶜ(i, j, k, grid, reference_potential) = @inbounds reference_potential[1, 1, k]

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

# 𝒲★ = p★(z) - p★(z★), from the reference potential at the cell's level and at its reference height
@inline reference_workᶜᶜᶜ(i, j, k, grid, reference_potential, parcel_reference_potential) =
    @inbounds reference_potential[1, 1, k] - parcel_reference_potential[i, j, k]

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

# 𝒜 = - b (z - z★) + 𝒲★ = b Υ + 𝒲★
@inline available_potential_energyᶜᶜᶜ(i, j, k, grid, buoyancy, z★, reference_potential, parcel_reference_potential) =
    @inbounds buoyancy[i, j, k] * reference_displacementᶜᶜᶜ(i, j, k, grid, z★) +
              reference_workᶜᶜᶜ(i, j, k, grid, reference_potential, parcel_reference_potential)

"""
$(TYPEDSIGNATURES)

Return the available potential energy 𝒜 = - b (z - z★) + 𝒲★ ≥ 0 at (Center, Center, Center).
"""
function available_potential_energy(z★::ReferenceHeightField)
    state = z★.operand
    return center_operation(available_potential_energyᶜᶜᶜ, z★.grid, state.buoyancy, z★, state.reference_potential,
                            state.parcel_reference_potential)
end

#=
Local APE budget. With Db/Dt = σ_b - ∇·j_b and b★(z★) = b,

    D𝒜/Dt = Υ σ_b - εₐ - ∇·j_𝒜 - w (b - b★(z)) + R★,
    εₐ = j_b·∇(z - z★),   j_𝒜 = Υ j_b,   R★ = Ṗ★(z) - Ṗ★(z★).

Discretely, with j_𝒜 = ℑ(Υ) j_b on the faces and εₐ the face contraction - A j_b δΥ averaged to the cell centres,
- Υ ∇·j_b = - ∇·j_𝒜 - εₐ holds exactly in every cell, as does b_θ ∇·j_θ + b_S ∇·j_S = ∇·j_b - σ_b_nonlin.

Boundary fluxes. The fluxes j_θ and j_S of the diagnostics are the closure fluxes at interior faces, and at the walls
the fluxes Q_θ and Q_S that the model applies: the flux of a `FluxBoundaryCondition`, plus, for a Fickian model, its
closure's own wall flux, that of a Value or Gradient boundary condition. The model applies a wall flux to the tendency
of the boundary cell, so that at a wall face every face product takes the coefficients of the boundary cell (X₁), and
every face contraction j δX is zero:

    j_b·n = b_θ₁ Q_θ + b_S₁ Q_S,     j_𝒜·n = Υ₁ j_b·n,     j_𝒫·n = π₁ Q_θ + m₁ Q_S,

and σ_b_nonlin, εₐ, j_θ·∇π + j_S·∇m and σ_η have no wall term. The identities above then hold in the boundary cells
as well, and the budgets hold in every cell with these fluxes.

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

The budget is ∂𝒜/∂t = Υ σ_b - εₐ - ∇·j_𝒜 - ∇·(u 𝒜) - ℑz[w (b̃ - b★)] + 𝒩 + R★, for both closures, with the wall
fluxes of θ and S in j_𝒜 and in the Ḃ of R★. Background fields, θ forcings other than σ_θ and immersed boundaries are
not included.
=#

# b_S ∇·(u S), zero for constant salinity
@inline salt_advection_buoyancy_tendencyᶜᶜᶜ(i, j, k, grid, advection, buoyancy_formulation, fields) =
    thermodynamic_functionᶜᶜᶜ(i, j, k, grid, ∂b∂S, buoyancy_formulation, fields) *
    div_Uc(i, j, k, grid, advection.S, fields, fields.S)

@inline salt_advection_buoyancy_tendencyᶜᶜᶜ(i, j, k, grid, advection, ::ConstantSalinityThermodynamicBuoyancy, fields) =
    zero(grid)

# ADV_b = - b_θ ∇·(u θ) - b_S ∇·(u S), with the model's advection schemes
@inline advective_buoyancy_tendencyᶜᶜᶜ(i, j, k, grid, advection, buoyancy_formulation, fields) =
    - thermodynamic_functionᶜᶜᶜ(i, j, k, grid, ∂b∂θ, buoyancy_formulation, fields) *
      div_Uc(i, j, k, grid, advection.θ, fields, fields.θ) -
    salt_advection_buoyancy_tendencyᶜᶜᶜ(i, j, k, grid, advection, buoyancy_formulation, fields)

"""
$(TYPEDSIGNATURES)

Return the buoyancy tendency ADV_b = - b_θ ∇·(u θ) - b_S ∇·(u S) of `model` due to its discrete advection of θ and S,
at (Center, Center, Center).
"""
advective_buoyancy_tendency(model) =
    center_operation(advective_buoyancy_tendencyᶜᶜᶜ, model.grid, model.advection, model.buoyancy.formulation,
                     velocities_and_tracers(model))

# Generated: area_buoyancy_flux_{x,y,z}, A j_b at faces, with the wall fluxes
for (dir, face_area) in ((:x, Axᶠᶜᶜ), (:y, Ayᶜᶠᶜ), (:z, Azᶜᶜᶠ))
    buoyancy_flux_xyz      = Symbol(:buoyancy_flux_, dir)
    area_buoyancy_flux_xyz = Symbol(:area_buoyancy_flux_, dir)
    @eval @inline $area_buoyancy_flux_xyz(i, j, k, grid, closure, buoyancy_formulation, fields, walls) =
        $face_area(i, j, k, grid) * $buoyancy_flux_xyz(i, j, k, grid, closure, buoyancy_formulation, fields, walls)
end

#=
Ḃ = σ_b - ∇·j_b, the non-advective buoyancy tendency, formed here for both closures from the face fluxes j_b of their
diffusivity, which include the wall fluxes of `walls` (`wall_conditions(model)`), with σ_b = σ_b_nonlin for a Fickian
model, whose σ_θ is zero. In a boundary cell, - ∇·j_b includes b_θ₁ F_θ + b_S₁ F_S, with F_θ and F_S the tendencies of
θ and S by the wall fluxes.
=#
@inline function buoyancy_tendencyᶜᶜᶜ(i, j, k, grid, clock, fields, parameters, walls)
    closure, buoyancy_formulation = parameters.diffusivity, parameters.buoyancy
    ∇_dot_j_b = (δxᶜᵃᵃ(i, j, k, grid, area_buoyancy_flux_x, closure, buoyancy_formulation, fields, walls) +
                 δyᵃᶜᵃ(i, j, k, grid, area_buoyancy_flux_y, closure, buoyancy_formulation, fields, walls) +
                 δzᵃᵃᶜ(i, j, k, grid, area_buoyancy_flux_z, closure, buoyancy_formulation, fields, walls)) / Vᶜᶜᶜ(i, j, k, grid)
    return buoyancy_productionᶜᶜᶜ(i, j, k, grid, clock, fields, parameters) - ∇_dot_j_b
end

"""
$(TYPEDSIGNATURES)

Return the non-advective buoyancy tendency Ḃ = σ_b - ∇·j_b of `model` at (Center, Center, Center), with j_b the
[`buoyancy_flux`](@ref). In a boundary cell it includes b_θ F_θ + b_S F_S, the change of b by the tendencies F_θ and
F_S of the wall fluxes of θ and S that the model applies.
"""
buoyancy_tendency(model) =
    center_operation(buoyancy_tendencyᶜᶜᶜ, model.grid, model.clock, velocities_and_tracers(model),
                     production_parameters(model), wall_conditions(model))

# Ḃ = ADV_b + σ_b - ∇·j_b, the full discrete buoyancy tendency
@inline total_buoyancy_tendencyᶜᶜᶜ(i, j, k, grid, advection, clock, fields, parameters, walls) =
    advective_buoyancy_tendencyᶜᶜᶜ(i, j, k, grid, advection, parameters.buoyancy, fields) +
    buoyancy_tendencyᶜᶜᶜ(i, j, k, grid, clock, fields, parameters, walls)

# Υ σ, the production of 𝒜 by the buoyancy source σ = production(i, j, k, grid, clock, fields, parameters)
@inline ape_productionᶜᶜᶜ(i, j, k, grid, z★, production::P, clock, fields, parameters) where P =
    reference_displacementᶜᶜᶜ(i, j, k, grid, z★) * production(i, j, k, grid, clock, fields, parameters)

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
(Center, Center, Center). It is zero for a Fickian (`consistent = false`) model, which has no θ source.
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

#=
Generated: face_ape_dissipation_{x,y,z}, - A j_b δΥ = A j_b δ(z - z★) at faces, with the closure fluxes, which are zero
at walls: a wall flux enters the boundary cell, whose Υ it takes (j_𝒜 = Υ₁ j_b at a wall), and so has no part in εₐ
=#
for (dir, difference_to_face, face_area) in ((:x, δxᶠᵃᵃ, Axᶠᶜᶜ), (:y, δyᵃᶠᵃ, Ayᶜᶠᶜ), (:z, δzᵃᵃᶠ, Azᶜᶜᶠ))
    closure_buoyancy_flux_xyz = Symbol(:closure_buoyancy_flux_, dir)
    face_ape_dissipation_xyz  = Symbol(:face_ape_dissipation_, dir)

    @eval @inline $face_ape_dissipation_xyz(i, j, k, grid, z★, closure, buoyancy_formulation, fields) =
        - $face_area(i, j, k, grid) * $closure_buoyancy_flux_xyz(i, j, k, grid, closure, buoyancy_formulation, fields) *
          $difference_to_face(i, j, k, grid, reference_displacementᶜᶜᶜ, z★)
end

# εₐ = j_b·∇(z - z★); by summation by parts Σ V εₐ = Σ V Υ ∇·j_b exactly
@inline ape_dissipationᶜᶜᶜ(i, j, k, grid, z★, closure, buoyancy_formulation, fields) =
    (ℑxᶜᵃᵃ(i, j, k, grid, face_ape_dissipation_x, z★, closure, buoyancy_formulation, fields) +
     ℑyᵃᶜᵃ(i, j, k, grid, face_ape_dissipation_y, z★, closure, buoyancy_formulation, fields) +
     ℑzᵃᵃᶜ(i, j, k, grid, face_ape_dissipation_z, z★, closure, buoyancy_formulation, fields)) / Vᶜᶜᶜ(i, j, k, grid)

"""
$(TYPEDSIGNATURES)

Return the diffusive dissipation εₐ = j_b·∇(z - z★) of available potential energy of `model`, at
(Center, Center, Center). It has no wall term: a wall flux enters the boundary cell, whose Υ it takes in the
[`ape_flux`](@ref). By summation by parts, its integral equals that of Υ ∇·j_b - ∇·j_𝒜 exactly, and that of Υ ∇·j_b
when the wall fluxes vanish.
"""
ape_dissipation(model, z★::ReferenceHeightField) =
    center_operation(ape_dissipationᶜᶜᶜ, model.grid, z★, production_parameters(model).diffusivity,
                     model.buoyancy.formulation, field_data(model.tracers))

# Generated: ape_flux_{x,y,z}, j_𝒜 = ℑ(Υ) j_b at interior faces and Υ₁ j_b at walls, with Υ₁ of the boundary cell
for (dir, average_to_face, boundary_cell_value) in ((:x, ℑxᶠᵃᵃ, boundary_cell_value_x),
                                                    (:y, ℑyᵃᶠᵃ, boundary_cell_value_y),
                                                    (:z, ℑzᵃᵃᶠ, boundary_cell_value_z))
    closure_buoyancy_flux_xyz = Symbol(:closure_buoyancy_flux_, dir)
    wall_buoyancy_flux_xyz    = Symbol(:wall_buoyancy_flux_, dir)
    ape_flux_xyz              = Symbol(:ape_flux_, dir)

    @eval @inline $ape_flux_xyz(i, j, k, grid, z★, closure, buoyancy_formulation, fields, walls) =
        $average_to_face(i, j, k, grid, reference_displacementᶜᶜᶜ, z★) *
        $closure_buoyancy_flux_xyz(i, j, k, grid, closure, buoyancy_formulation, fields) +
        $boundary_cell_value(i, j, k, grid, reference_displacementᶜᶜᶜ, z★) *
        $wall_buoyancy_flux_xyz(i, j, k, grid, closure, buoyancy_formulation, fields, walls)
end

"""
$(TYPEDSIGNATURES)

Return the diffusive flux j_𝒜 = Υ j_b of available potential energy of `model`, a `NamedTuple` of its `(x, y, z)`
components at faces. At a wall it is Υ₁ j_b, with Υ₁ of the boundary cell and j_b the [`buoyancy_flux`](@ref) there,
so that its integrated divergence is the APE input by the wall fluxes. In every cell, - Υ ∇·j_b = - ∇·j_𝒜 - εₐ exactly.
"""
ape_flux(model, z★::ReferenceHeightField) =
    face_operations(ape_flux_x, ape_flux_y, ape_flux_z, model.grid, z★, production_parameters(model).diffusivity,
                    model.buoyancy.formulation, field_data(model.tracers), wall_conditions(model))

# w (b̃ - b★) at (Center, Center, Face), the conversion of 𝒜 into kinetic energy
@inline available_buoyancy_workᶜᶜᶠ(i, j, k, grid, buoyancy_formulation, fields, reference_buoyancy) =
    @inbounds fields.w[i, j, k] * (consistent_buoyancyᶜᶜᶠ(i, j, k, grid, buoyancy_formulation, fields) -
                                   reference_buoyancy[1, 1, k])

"""
$(TYPEDSIGNATURES)

Return the conversion w (b̃ - b★) of available potential energy into kinetic energy of `model`, at
(Center, Center, Face). For a divergence-free velocity, its integral equals that of the buoyancy work w b̃.
"""
available_buoyancy_work(model, z★::ReferenceHeightField) =
    KernelFunctionOperation{Center, Center, Face}(with_reference_state, model.grid, z★, available_buoyancy_workᶜᶜᶠ,
                                                  model.buoyancy.formulation, velocities_and_tracers(model),
                                                  z★.operand.reference_buoyancy)

# - ∇·(u e) of an energy density e, such as 𝒜 or ℬ, with the model's tracer advection scheme
@inline energy_advectionᶜᶜᶜ(i, j, k, grid, advection, fields, energy) = - div_Uc(i, j, k, grid, advection.θ, fields, energy)

# 𝒜 for the advective operators, which read it in neighbouring cells: as `available_potential_energy`, but with the
# data of z★, so that its type stays small; the operations that hold it compute the reference state through z★
function ape_data_operation(z★)
    state = z★.operand
    return center_operation(available_potential_energyᶜᶜᶜ, z★.grid, state.buoyancy, z★.data, state.reference_potential,
                            state.parcel_reference_potential)
end

"""
$(TYPEDSIGNATURES)

Return the advection - ∇·(u 𝒜) of available potential energy by `model`, with its tracer advection scheme, at
(Center, Center, Center).
"""
ape_advection(model, z★::ReferenceHeightField) =
    center_operation(with_reference_state, model.grid, z★, energy_advectionᶜᶜᶜ, model.advection,
                     velocities_and_tracers(model), ape_data_operation(z★))

# 𝒩 = Υ ADV_b + ∇·(u 𝒜) + ℑz[w (b̃ - b★)]
@inline spurious_ape_productionᶜᶜᶜ(i, j, k, grid, z★, ape, reference_buoyancy, advection, buoyancy_formulation, fields) =
    reference_displacementᶜᶜᶜ(i, j, k, grid, z★) *
    advective_buoyancy_tendencyᶜᶜᶜ(i, j, k, grid, advection, buoyancy_formulation, fields) -
    energy_advectionᶜᶜᶜ(i, j, k, grid, advection, fields, ape) +
    ℑzᵃᵃᶜ(i, j, k, grid, available_buoyancy_workᶜᶜᶠ, buoyancy_formulation, fields, reference_buoyancy)

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
discrete advection ADV_b, which leaves the reference state unchanged only in the continuum, and the wall fluxes of θ
and S in j_b, which change the buoyancy of the boundary cells and so the reference state at every level above their
reference heights. R★ integrates to zero.
"""
function reference_tendency_work(model, z★::ReferenceHeightField)
    parcel_tendency = reference_tendency(model, z★)
    # R★ has the form of 𝒲★; computing Ṗ★(z★) computes Ṗ★ at each level
    return center_operation(reference_workᶜᶜᶜ, z★.grid, parcel_tendency.operand.reference_tendency, parcel_tendency)
end
