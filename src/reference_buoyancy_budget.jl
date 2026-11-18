#=
Isopycnal budget of the reference buoyancy b★ (Nakamura 1996, J. Atmos. Sci. 53; Winters & D'Asaro 1996, J. Fluid
Mech. 317). With A the horizontal area of the domain, the isopycnal average and the isopycnal integral of a cell field
f are

    ⟨f⟩(z) = (1/A) ∬_{z★ = z} f |∇z★|⁻¹ dS,      𝓘[f](z) = ∫^z ⟨f⟩ dz′ = (1/A) ∫_{z★ < z} f dV.

Discretely, the reference tendency `reference_tendency(z★, f, arguments)` computes 𝓘[f] at each level into
`.operand.reference_tendency`, at (Nothing, Nothing, Center): it gathers f in the order of b★, takes its `cumsum!` with
the slot thickness h = V/A, and averages over the slots of each level (`sorting.jl`). With f the full discrete buoyancy
tendency Ḃ, it is Ṗ★ = ∂ₜp★.

At a given time, the permutation and the bands of tied buoyancies are fixed, so that 𝓘 is linear in f, the
interpolation within the bands included. For the split

    Ḃ = σ_b - ∇·j_b + ADV_b,      ∇·j_b = ∇·j_w + ∇·j_κ,

with σ_b the buoyancy production, j_b the buoyancy flux, wall fluxes included, j_w = b_θ₁ Q_θ + b_S₁ Q_S its part at
the wall faces, whose divergence is nonzero only in boundary cells, j_κ the closure's buoyancy flux, zero at the wall
faces, and ADV_b the buoyancy tendency of the discrete advection,

    Ṗ★ = 𝓘[σ_b] - 𝓘[∇·j_b] + 𝓘[ADV_b],      𝓘[∇·j_b] = 𝓘[∇·j_w] + 𝓘[∇·j_κ],

holds to round-off at every level, ties included.

With b★ᶜᶜᶠ = δz p★ᶜᶜᶜ / Δz, the reference buoyancy, and ∂z b★ = δz b★ᶜᶜᶠ / Δz at the cell centres, the diagnostics are
the isopycnal averages

    ⟨σ_b⟩ = δz 𝓘[σ_b] / Δz,   ⟨∇·j_b⟩ = δz 𝓘[∇·j_b] / Δz,   ⟨ADV_b⟩ = δz 𝓘[ADV_b] / Δz,   ⟨∇·j_w⟩ = δz 𝓘[∇·j_w] / Δz

at the w-faces, and the diffusivities K = - F / ∂z b★ of two diapycnal fluxes F = 𝓘[∇·j] at the cell centres,

    K_eff = - 𝓘[∇·j_κ] / ∂z b★,      K_num = - 𝓘[- ADV_b] / ∂z b★ = 𝓘[ADV_b] / ∂z b★.

𝓘[∇·j_κ] is the flux ⟨j_b·∇z★⟩ of the closure's buoyancy flux through the isopycnal z★ = z. For K_num, j is the
buoyancy flux of the discrete advection, whose divergence is - ADV_b and whose net flux through an isopycnal vanishes
in the continuum, where ⟨ADV_b⟩ = K_num = 0. K_eff > 0 for down-gradient fluxes: for a Fickian j_b = - κ ∇b with a
linear equation of state, K_eff = κ ⟨|∇b|²⟩ / (∂z b★)² ≥ κ (Winters & D'Asaro 1996). K_num can be negative, where
centred advection unmixes (Ilıcak et al. 2012, Ocean Modell. 45-46).

Taking δz / Δz of the split, at the interior faces k = 2, …, Nz, exactly,

    δz Ṗ★ / Δz = ⟨σ_b⟩ - ⟨∇·j_b⟩ + ⟨ADV_b⟩
               = ⟨σ_b⟩ - ⟨∇·j_w⟩ + δz[(K_eff + K_num) ∂z b★] / Δz.

When no two cells share a buoyancy, Ṗ★ᶜᶜᶜ is the exact time derivative of p★ᶜᶜᶜ (`sorting.jl`), so that the left side
is ∂ₜb★ᶜᶜᶠ. With ties, the split still holds, but Ṗ★ᶜᶜᶜ is formed by the interpolation within the bands and need not be
the time derivative of p★ᶜᶜᶜ.

On the boundary faces, b★ is the buoyancy of the boundary slot of the sorted column rather than δz p★ / Δz. The face
budget therefore holds at the interior faces only, and the four face diagnostics are zero on the boundary faces. Where
∂z b★ = 0, the two diffusivities are infinite or NaN.

When no two cells share a buoyancy, summation by parts over the slots, at whose centres z★ = bottom + h (n - ½), gives
the first moment of any cell field f,

    Σ V z★ f = z_top Σ V f - A Σₖ Δz 𝓘[f]ₖ,

which identifies each diagnostic with its part of Ḃ. With f = ADV_b, Σ V z★ ADV_b is the domain integral of
`spurious_ape_production` with centred second-order advection and the energy-consistent b̃ (`ape_budget.jl`), which
relates ⟨ADV_b⟩ and K_num to the spurious mixing of the discrete advection.
=#

# ⟨f⟩ = δz 𝓘[f] / Δz at the interior w-faces, zero on the boundary faces, where `ifelse` discards the halo values read
@inline isopycnal_averageᶜᶜᶠ(i, j, k, grid, isopycnal_integral) =
    ifelse((k == 1) | (k == size(grid, 3) + 1), zero(grid), ∂zᶜᶜᶠ(1, 1, k, grid, isopycnal_integral))

# Computing the reference tendency of f computes z★ and the isopycnal integral 𝓘[f] at each level, which the
# operation reads; it is passed through `with_reference_state` so that computing the operation computes it.

"""
$(TYPEDSIGNATURES)

Return the isopycnal average ⟨σ_b⟩ of the buoyancy production σ_b of `model` at the w-faces, as a
`KernelFunctionOperation` at (Nothing, Nothing, Face), for the reference height z★ returned by
[`reference_height`](@ref). The budgets

    ∂ₜb★ = ⟨σ_b⟩ − ⟨∇·j_b⟩ + ⟨ADV_b⟩
         = ⟨σ_b⟩ − ⟨∇·j_w⟩ + ∂z[(K_eff + K_num) ∂z b★]

hold exactly at the interior w-faces when no two cells share a buoyancy, with ⟨σ_b⟩ the
[`isopycnal_buoyancy_production`](@ref), ⟨∇·j_b⟩ the [`isopycnal_buoyancy_flux_divergence`](@ref), ⟨ADV_b⟩ the
[`isopycnal_advective_buoyancy_tendency`](@ref), ⟨∇·j_w⟩ the [`isopycnal_wall_buoyancy_flux_divergence`](@ref),
K_eff the [`effective_diapycnal_diffusivity`](@ref) and K_num the [`numerical_diapycnal_diffusivity`](@ref).

It is zero on the boundary faces, where b★ is the buoyancy of the boundary slot of the sorted column rather than
δz p★ / Δz.
"""
function isopycnal_buoyancy_production(model, z★::ReferenceHeightField)
    parameters, fields = production_parameters(model), velocities_and_tracers(model)
    parcel_tendency = reference_tendency(z★, buoyancy_productionᶜᶜᶜ, (model.clock, fields, parameters))
    return KernelFunctionOperation{Nothing, Nothing, Face}(with_reference_state, model.grid, parcel_tendency,
                                                          isopycnal_averageᶜᶜᶠ, parcel_tendency.operand.reference_tendency)
end

"""
$(TYPEDSIGNATURES)

Return the isopycnal average ⟨∇·j_b⟩ of the divergence of the [`buoyancy_flux`](@ref) j_b of `model`, wall fluxes
included, at the w-faces, as a `KernelFunctionOperation` at (Nothing, Nothing, Face). The budgets

    ∂ₜb★ = ⟨σ_b⟩ − ⟨∇·j_b⟩ + ⟨ADV_b⟩
         = ⟨σ_b⟩ − ⟨∇·j_w⟩ + ∂z[(K_eff + K_num) ∂z b★]

hold exactly at the interior w-faces when no two cells share a buoyancy, with ⟨σ_b⟩ the
[`isopycnal_buoyancy_production`](@ref), ⟨∇·j_b⟩ the [`isopycnal_buoyancy_flux_divergence`](@ref), ⟨ADV_b⟩ the
[`isopycnal_advective_buoyancy_tendency`](@ref), ⟨∇·j_w⟩ the [`isopycnal_wall_buoyancy_flux_divergence`](@ref),
K_eff the [`effective_diapycnal_diffusivity`](@ref) and K_num the [`numerical_diapycnal_diffusivity`](@ref).

It is zero on the boundary faces, where b★ is the buoyancy of the boundary slot of the sorted column rather than
δz p★ / Δz.
"""
function isopycnal_buoyancy_flux_divergence(model, z★::ReferenceHeightField)
    parameters, fields = production_parameters(model), velocities_and_tracers(model)
    parcel_tendency = reference_tendency(z★, flux_divergenceᶜᶜᶜ, (buoyancy_flux_x, buoyancy_flux_y, buoyancy_flux_z,
                                                                  parameters.diffusivity, parameters.buoyancy, fields,
                                                                  wall_conditions(model)))
    return KernelFunctionOperation{Nothing, Nothing, Face}(with_reference_state, model.grid, parcel_tendency,
                                                          isopycnal_averageᶜᶜᶠ, parcel_tendency.operand.reference_tendency)
end

"""
$(TYPEDSIGNATURES)

Return the isopycnal average ⟨ADV_b⟩ of the [`advective_buoyancy_tendency`](@ref) ADV_b of `model`, the change of b by
the discrete advection of θ and S, at the w-faces, as a `KernelFunctionOperation` at (Nothing, Nothing, Face).
⟨ADV_b⟩ is zero in the continuum, where advection only rearranges the fluid. The budgets

    ∂ₜb★ = ⟨σ_b⟩ − ⟨∇·j_b⟩ + ⟨ADV_b⟩
         = ⟨σ_b⟩ − ⟨∇·j_w⟩ + ∂z[(K_eff + K_num) ∂z b★]

hold exactly at the interior w-faces when no two cells share a buoyancy, with ⟨σ_b⟩ the
[`isopycnal_buoyancy_production`](@ref), ⟨∇·j_b⟩ the [`isopycnal_buoyancy_flux_divergence`](@ref), ⟨ADV_b⟩ the
[`isopycnal_advective_buoyancy_tendency`](@ref), ⟨∇·j_w⟩ the [`isopycnal_wall_buoyancy_flux_divergence`](@ref),
K_eff the [`effective_diapycnal_diffusivity`](@ref) and K_num the [`numerical_diapycnal_diffusivity`](@ref).

It is zero on the boundary faces, where b★ is the buoyancy of the boundary slot of the sorted column rather than
δz p★ / Δz.
"""
function isopycnal_advective_buoyancy_tendency(model, z★::ReferenceHeightField)
    parameters, fields = production_parameters(model), velocities_and_tracers(model)
    parcel_tendency = reference_tendency(z★, advective_buoyancy_tendencyᶜᶜᶜ, (model.advection, parameters.buoyancy, fields))
    return KernelFunctionOperation{Nothing, Nothing, Face}(with_reference_state, model.grid, parcel_tendency,
                                                          isopycnal_averageᶜᶜᶠ, parcel_tendency.operand.reference_tendency)
end

"""
$(TYPEDSIGNATURES)

Return the isopycnal average ⟨∇·j_w⟩ of the divergence of the wall part j_w = b_θ₁ Q_θ + b_S₁ Q_S of the buoyancy flux
of `model`, with Q_θ and Q_S its wall fluxes of θ and S, nonzero only in boundary cells, at the w-faces, as a
`KernelFunctionOperation` at (Nothing, Nothing, Face). It is zero for a model without wall fluxes. The budgets

    ∂ₜb★ = ⟨σ_b⟩ − ⟨∇·j_b⟩ + ⟨ADV_b⟩
         = ⟨σ_b⟩ − ⟨∇·j_w⟩ + ∂z[(K_eff + K_num) ∂z b★]

hold exactly at the interior w-faces when no two cells share a buoyancy, with ⟨σ_b⟩ the
[`isopycnal_buoyancy_production`](@ref), ⟨∇·j_b⟩ the [`isopycnal_buoyancy_flux_divergence`](@ref), ⟨ADV_b⟩ the
[`isopycnal_advective_buoyancy_tendency`](@ref), ⟨∇·j_w⟩ the [`isopycnal_wall_buoyancy_flux_divergence`](@ref),
K_eff the [`effective_diapycnal_diffusivity`](@ref) and K_num the [`numerical_diapycnal_diffusivity`](@ref).

It is zero on the boundary faces, where b★ is the buoyancy of the boundary slot of the sorted column rather than
δz p★ / Δz.
"""
function isopycnal_wall_buoyancy_flux_divergence(model, z★::ReferenceHeightField)
    parameters, fields = production_parameters(model), velocities_and_tracers(model)
    parcel_tendency = reference_tendency(z★, flux_divergenceᶜᶜᶜ, (wall_buoyancy_flux_x, wall_buoyancy_flux_y,
                                                                  wall_buoyancy_flux_z, parameters.diffusivity,
                                                                  parameters.buoyancy, fields, wall_conditions(model)))
    return KernelFunctionOperation{Nothing, Nothing, Face}(with_reference_state, model.grid, parcel_tendency,
                                                          isopycnal_averageᶜᶜᶠ, parcel_tendency.operand.reference_tendency)
end

# K = - F / ∂z b★, the diffusivity of a diapycnal flux F, with ∂z b★ = δz b★ / Δz
@inline diapycnal_diffusivityᶜᶜᶜ(i, j, k, grid, isopycnal_integral, reference_buoyancy) =
    @inbounds - isopycnal_integral[1, 1, k] / ∂zᶜᶜᶜ(1, 1, k, grid, reference_buoyancy)

"""
$(TYPEDSIGNATURES)

Return the effective diapycnal diffusivity K_eff = −⟨j_b·∇z★⟩ / ∂z b★ of `model` at the cell centres, as a
`KernelFunctionOperation` at (Nothing, Nothing, Center), with ⟨j_b·∇z★⟩ = (1/A) ∫_{z★ < z} ∇·j_b dV the flux of the
closure's buoyancy flux j_b through the isopycnal z★ = z, and ∂z b★ = δz b★ / Δz. For a Fickian flux j_b = −κ ∇b,
K_eff = κ ⟨|∇b|²⟩ / (∂z b★)² ≥ κ (Winters & D'Asaro 1996). The budgets

    ∂ₜb★ = ⟨σ_b⟩ − ⟨∇·j_b⟩ + ⟨ADV_b⟩
         = ⟨σ_b⟩ − ⟨∇·j_w⟩ + ∂z[(K_eff + K_num) ∂z b★]

hold exactly at the interior w-faces when no two cells share a buoyancy, with ⟨σ_b⟩ the
[`isopycnal_buoyancy_production`](@ref), ⟨∇·j_b⟩ the [`isopycnal_buoyancy_flux_divergence`](@ref), ⟨ADV_b⟩ the
[`isopycnal_advective_buoyancy_tendency`](@ref), ⟨∇·j_w⟩ the [`isopycnal_wall_buoyancy_flux_divergence`](@ref),
K_eff the [`effective_diapycnal_diffusivity`](@ref) and K_num the [`numerical_diapycnal_diffusivity`](@ref).

It is infinite or NaN where ∂z b★ = 0.
"""
function effective_diapycnal_diffusivity(model, z★::ReferenceHeightField)
    parameters, fields = production_parameters(model), velocities_and_tracers(model)
    parcel_tendency = reference_tendency(z★, flux_divergenceᶜᶜᶜ, (closure_buoyancy_flux_x, closure_buoyancy_flux_y,
                                                                  closure_buoyancy_flux_z, parameters.diffusivity,
                                                                  parameters.buoyancy, fields))
    return KernelFunctionOperation{Nothing, Nothing, Center}(with_reference_state, model.grid, parcel_tendency,
                                                            diapycnal_diffusivityᶜᶜᶜ, parcel_tendency.operand.reference_tendency,
                                                            z★.operand.reference_buoyancy)
end

"""
$(TYPEDSIGNATURES)

Return the numerical diapycnal diffusivity K_num = (1/A) ∫_{z★ < z} ADV_b dV / ∂z b★ of `model` at the cell centres,
as a `KernelFunctionOperation` at (Nothing, Nothing, Center), with ADV_b the [`advective_buoyancy_tendency`](@ref), so
that ⟨ADV_b⟩ = ∂z(K_num ∂z b★). K_num is zero in the continuum, and negative where the advection unmixes (Ilıcak et al.
2012). The budgets

    ∂ₜb★ = ⟨σ_b⟩ − ⟨∇·j_b⟩ + ⟨ADV_b⟩
         = ⟨σ_b⟩ − ⟨∇·j_w⟩ + ∂z[(K_eff + K_num) ∂z b★]

hold exactly at the interior w-faces when no two cells share a buoyancy, with ⟨σ_b⟩ the
[`isopycnal_buoyancy_production`](@ref), ⟨∇·j_b⟩ the [`isopycnal_buoyancy_flux_divergence`](@ref), ⟨ADV_b⟩ the
[`isopycnal_advective_buoyancy_tendency`](@ref), ⟨∇·j_w⟩ the [`isopycnal_wall_buoyancy_flux_divergence`](@ref),
K_eff the [`effective_diapycnal_diffusivity`](@ref) and K_num the [`numerical_diapycnal_diffusivity`](@ref).

It is infinite or NaN where ∂z b★ = 0.
"""
function numerical_diapycnal_diffusivity(model, z★::ReferenceHeightField)
    parameters, fields = production_parameters(model), velocities_and_tracers(model)
    parcel_tendency = reference_tendency(z★, advective_buoyancy_flux_divergenceᶜᶜᶜ, (model.advection, parameters.buoyancy,
                                                                                     fields))
    return KernelFunctionOperation{Nothing, Nothing, Center}(with_reference_state, model.grid, parcel_tendency,
                                                            diapycnal_diffusivityᶜᶜᶜ, parcel_tendency.operand.reference_tendency,
                                                            z★.operand.reference_buoyancy)
end
